"""Quantization config and linear method for EXL3 checkpoints."""
import os
from typing import Any

import torch
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import LinearBase, LinearMethodBase
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from vllm.model_executor.parameter import BasevLLMParameter


from .ops import GEMM_MAX_ROWS as _GEMM_MAX_ROWS, exl3_K  # noqa: F401  (also registers orcasaq2::shard_gemm)

logger = init_logger(__name__)

# vLLM fuses these; each constituent is its own EXL3 tensor, so a fused layer carries a list
# of shards. Order matters: it must match the order vLLM concatenates outputs in.
SHARD_IDS = {
    "qkv_proj": ["q", "k", "v"],
    "gate_up_proj": [0, 1],
    "in_proj_qkvz": [0, 1],
}


class Exl3ShardParam(BasevLLMParameter):
    """One placeholder parameter standing in for a list of per-shard EXL3 tensors.

    vLLM wants a Parameter of known shape per weight name; EXL3 shard shapes (and K) are only
    known once the file is read, and differ between shards of a fused layer. So the Parameter
    itself is a 1-element dummy and the real tensors land in `shards`, keyed by the loader's
    shard id (slot 0 for an unfused layer).

    It carries its OWN weight_loader. vLLM's generic path is
    `param.weight_loader(param, loaded_weight, shard_id)`, and if that resolves to the layer's
    legacy loader it asserts the placeholder's shape against the incoming tensor and fails;
    installing our own function claims that path outright.
    """

    def __new__(cls, **kwargs):
        return super().__new__(cls, data=torch.empty(0))

    def __init__(self, suffix: str):
        super().__init__(data=torch.empty(0), weight_loader=_capture)
        # slot -> tensor. A dict rather than a fixed list because the number of EXL3 tensors
        # behind a fused layer is not its output-partition count: vLLM builds in_proj_qkvz
        # with four partitions (q, k, v, z) while the pack has two tensors (in_proj_qkv, in_proj_z).
        self.shards: dict[int, torch.Tensor] = {}
        self.suffix = suffix

    def slot(self, shard_id):
        if shard_id is None:
            return 0
        if isinstance(shard_id, int):
            return shard_id
        if isinstance(shard_id, tuple):           # consecutive-range form
            return shard_id[0]
        for ids in SHARD_IDS.values():
            if shard_id in ids:
                return ids.index(shard_id)
        raise ValueError(f"unknown shard id {shard_id!r}")

    # The BasevLLMParameter hooks, for whichever layers take vLLM's newer path.
    def load_column_parallel_weight(self, loaded_weight, **kw):
        _capture(self, loaded_weight, kw.get("shard_id", kw.get("loaded_shard_id")))

    load_row_parallel_weight = load_column_parallel_weight
    load_merged_column_weight = load_column_parallel_weight
    load_qkv_weight = load_column_parallel_weight


def _capture(param, loaded_weight, shard_id=None, **kw):
    """Stash a shard; shapes are whatever the checkpoint says, no validation to do."""
    param.shards[param.slot(shard_id)] = loaded_weight


class Exl3Config(QuantizationConfig):
    """`quant_method: "exl3"` — exllamav3 trellis quantization, any per-tensor bitrate."""

    def __init__(self, bits: float, codebook: str, head_bits: float, full_config: dict):
        super().__init__()
        self.bits = bits
        self.codebook = codebook
        self.head_bits = head_bits
        self.full_config = full_config
        # which module prefixes actually carry a trellis, from the pack's own manifest
        ts = full_config.get("tensor_storage") or {}
        self.quantized_prefixes = {
            k for k, v in ts.items()
            if any(n.endswith(".trellis") for n in (v.get("stored_tensors") or {}))
        }

    def __repr__(self):
        return (f"Exl3Config(bits={self.bits}, codebook={self.codebook!r}, "
                f"head_bits={self.head_bits}, quantized={len(self.quantized_prefixes)})")

    @classmethod
    def get_name(cls):
        return "exl3"

    @classmethod
    def get_supported_act_dtypes(cls):
        return [torch.half, torch.bfloat16]

    @classmethod
    def get_min_capability(cls):
        return 80

    @staticmethod
    def get_config_filenames():
        return ["quantization_config.json"]

    @classmethod
    def from_config(cls, config: dict[str, Any]):
        return cls(bits=float(config.get("bits", 0.0)),
                   codebook=config.get("codebook", "mul1"),
                   head_bits=float(config.get("head_bits", 16)),
                   full_config=config)

    def get_quant_method(self, layer: torch.nn.Module, prefix: str) -> QuantizeMethodBase | None:
        if not isinstance(layer, LinearBase):
            from vllm.model_executor.layers.vocab_parallel_embedding import (
                ParallelLMHead,
                VocabParallelEmbedding,
            )
            if isinstance(layer, ParallelLMHead):
                # A recipe with head_bits below 16 makes lm_head an ordinary EXL3 tensor, and
                # ParallelLMHead is exempt from implementing embedding(), so the linear method
                # is already correct. It falls back to a dense weight when the pack has one.
                return Exl3LinearMethod(self, prefix)
            if isinstance(layer, VocabParallelEmbedding):
                from .heads import Exl3EmbeddingMethod
                return Exl3EmbeddingMethod()
            return None
        # A fused layer is quantized iff every constituent is; the manifest is authoritative
        # when present, otherwise assume every LinearBase in a decoder layer is quantized.
        leaf = prefix.rsplit(".", 1)[-1]
        names = [prefix.replace(leaf, str(s)) for s in SHARD_IDS.get(leaf, [])] or [prefix]
        if self.quantized_prefixes:
            cand = [p for p in _expand(prefix, leaf)]
            if not any(c in self.quantized_prefixes for c in cand):
                from vllm.model_executor.layers.quantization.unquant import (
                    UnquantizedLinearMethod,
                )
                return UnquantizedLinearMethod()
        return Exl3LinearMethod(self, prefix)

    def get_scaled_act_names(self):
        return []


def _expand(prefix: str, leaf: str) -> list[str]:
    """Fused vLLM name -> the HF names its shards came from."""
    mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
        "in_proj_qkvz": ["in_proj_qkv", "in_proj_z"],
    }
    if leaf in mapping:
        return [prefix[: -len(leaf)] + m for m in mapping[leaf]]
    return [prefix]


class Exl3LinearMethod(LinearMethodBase):
    def __init__(self, quant_config: Exl3Config, prefix: str = ""):
        self.quant_config = quant_config
        self.prefix = prefix

    def create_weights(self, layer, input_size_per_partition, output_partition_sizes,
                       input_size, output_size, params_dtype, **extra_weight_attrs):
        if input_size != input_size_per_partition or sum(output_partition_sizes) != output_size:
            raise NotImplementedError(
                "orcasaq2 does not implement tensor parallelism yet; serve with -tp 1 "
                "rather than silently producing wrong numbers"
            )

        # An EXL3 pack is not uniformly quantized: exllamav3 leaves some projections in
        # bf16 (linear_attn.in_proj_a/b, the vision tower's qkv, anything given 16 bits), and
        # they arrive as a plain `weight`. Which mode a layer is in is only knowable from what
        # the loader delivers, so register placeholders for both and decide afterwards.
        for suffix in ("trellis", "suh", "svh", "mcg", "mul1", "weight"):
            layer.register_parameter(suffix, Exl3ShardParam(suffix=suffix))
        layer.exl3_out_total = sum(output_partition_sizes)


    def process_weights_after_loading(self, layer) -> None:
        tr, suh, svh = layer.trellis.shards, layer.suh.shards, layer.svh.shards
        mcg, mul1 = layer.mcg.shards, layer.mul1.shards
        dense = layer.weight.shards
        slots = sorted(tr) or sorted(dense)          # shard order == vLLM's concat order
        # The loader may hand these over on CPU; pin everything to this worker's device
        # rather than discovering it mid-forward as "mat2 is on cpu".
        dev = next((t.device for t in list(tr.values()) + list(dense.values())
                    if t.device.type == "cuda"),
                   torch.device("cuda", torch.cuda.current_device()))
        def finish():
            for p in ("trellis", "suh", "svh", "mcg", "mul1", "weight"):
                if hasattr(layer, p):
                    delattr(layer, p)
        if not tr:
            # unquantized projection (exllamav3 leaves in_proj_a/b and anything at 16 bits
            # dense): keep it as a plain weight, exactly as vLLM would have
            if not dense:
                raise RuntimeError(f"{self.prefix}: no trellis and no dense weight loaded")
            layer.exl3_shards = None
            w = torch.cat([dense[i] for i in slots], 0) if len(slots) > 1 else dense[slots[0]]
            layer.exl3_dense = torch.nn.Parameter(w.contiguous().to(dev), requires_grad=False)
            finish(); return
        if dense:
            raise RuntimeError(f"{self.prefix}: {len(tr)} trellis and {len(dense)} dense shards; "
                               "a fused layer must be all one kind or all the other")
        layer.exl3_dense = None
        layer.exl3_shards = [
            dict(trellis=tr[i].contiguous().to(dev), suh=suh[i].to(dev, torch.half).contiguous(),
                 svh=svh[i].to(dev, torch.half).contiguous(),
                 mcg=mcg.get(i) is not None, mul1=mul1.get(i) is not None,
                 K=exl3_K(tr[i]), ic=suh[i].numel(), oc=svh[i].numel())
            for i in slots
        ]
        got = sum(s["oc"] for s in layer.exl3_shards)
        if got != layer.exl3_out_total:
            raise RuntimeError(f"{self.prefix}: shards give {got} outputs, layer wants "
                               f"{layer.exl3_out_total}")
        finish()

    def apply(self, layer, x: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        if layer.exl3_shards is None:
            out = torch.nn.functional.linear(x, layer.exl3_dense.to(x.dtype))
            return out + bias if bias is not None else out
        rows = x.numel() // x.shape[-1]
        xf = x.reshape(rows, x.shape[-1]).half()
        if not xf.is_contiguous():
            xf = xf.contiguous()          # the kernels read A as contiguous rows
        outs = []
        for s in layer.exl3_shards:
            # One custom op per shard. The extension calls themselves are invisible to Dynamo
            # (raw pybind11, no source file), so calling them here directly makes vLLM's
            # torch.compile fail the engine at startup -- see orcasaq2/ops.py.
            outs.append(torch.ops.orcasaq2.shard_gemm(
                xf, s["trellis"], s["suh"], s["svh"], s["oc"], float(s["K"]), s["mcg"], s["mul1"]))
        out = (outs[0] if len(outs) == 1 else torch.cat(outs, dim=-1)).to(x.dtype)
        out = out.reshape(*x.shape[:-1], out.shape[-1])
        return out + bias if bias is not None else out
