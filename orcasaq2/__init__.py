"""Serve exllamav3 (EXL3) trellis-quantized checkpoints in vLLM.

An EXL3 checkpoint stores each projection self-contained: a packed tail-biting trellis plus
its own incoherence vectors, and the bitrate is implied by the trellis width. Two consequences
shape this plugin:

  * Shards of a fused vLLM layer (q/k/v inside qkv_proj, gate/up inside gate_up_proj) are
    separate EXL3 tensors with different shapes -- and, under a mixed-precision allocation,
    different K. There is no single tensor to allocate, so each shard is kept as its own
    buffer and `apply` runs one GEMM per shard and concatenates. This mirrors how the
    upstream SGLang integration handles the same problem.
  * Nothing about the shapes is known until the weights arrive, so create_weights registers
    a placeholder and a loader that captures each shard as it is loaded.

Registered through vLLM's own `register_quantization_config`, so no monkeypatching: a
checkpoint whose config.json says `quant_method: "exl3"` is picked up automatically.
Activate by installing this package into the serving environment; the exllamav3 wheel must
match the environment's torch (its extension is ABI-linked).

Status: correctness path only. `apply` reconstructs each shard's dense weight with the
exllamav3 kernel and runs a bf16 GEMM. The fused `exl3_gemm` decode kernel is wired but off
by default until the reconstruct path is verified token-for-token (ORCA_EXL3_GEMM=1 to try it).
Tensor parallelism is not implemented; TP>1 raises rather than serving wrong numbers.
"""
import os

import torch

BLOCK = 128
_HAD = {}


def _had(device, dtype):
    key = (str(device), dtype)
    if key not in _HAD:
        h = torch.tensor([[1.0]])
        while h.shape[0] < BLOCK:
            h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
        _HAD[key] = (h / BLOCK ** 0.5).to(device=device, dtype=dtype)
    return _HAD[key]


def _t128(x):
    """Blockwise size-128 Hadamard along the last dim (exllamav3's preapply_had_r)."""
    s = x.shape
    return (x.reshape(*s[:-1], s[-1] // BLOCK, BLOCK) @ _had(x.device, x.dtype)).reshape(s)


def reconstruct_weight(trellis, suh, svh, mcg, mul1):
    """EXL3 deploy weight (out, in) in fp16, matching exl3.py::get_weight_tensor."""
    from exllamav3.ext import exllamav3_ext as ext

    ic, oc = suh.numel(), svh.numel()
    k = trellis.shape[-1] / 16
    k = int(k) if float(k).is_integer() else k   # mul1 half rates are not integers
    w = torch.empty((ic, oc), dtype=torch.half, device=trellis.device)
    ext.reconstruct(w, trellis, k, mcg is not None, mul1 is not None)
    w = w.float()
    w = _t128(w.t()).t() * suh.float().unsqueeze(1)
    w = _t128(w) * svh.float().unsqueeze(0)
    return w.t().contiguous().half()


def register():
    from vllm.logger import init_logger
    from vllm.model_executor.layers.quantization import register_quantization_config

    logger = init_logger("vllm.orcasaq2")

    # Fail here, not in the middle of a forward pass. The trellis kernels live in the
    # exllamav3 extension, which every GEMM reaches for lazily; without it the first
    # request dies with a bare ModuleNotFoundError from inside a custom op, hours after
    # anyone could have acted on it.
    try:
        from exllamav3.ext import exllamav3_ext  # noqa: F401
    except Exception as e:
        raise RuntimeError(
            "orcasaq2 needs the exllamav3 CUDA extension, and importing it failed: "
            f"{type(e).__name__}: {e}\n"
            "It is not on PyPI -- the wheel is ABI-linked to both torch and the CUDA major "
            "version, so pick the release asset matching this environment:\n"
            "  python -c \"import torch; print(torch.__version__, torch.version.cuda)\"\n"
            "  https://github.com/turboderp-org/exllamav3/releases  "
            "(cu128.* for CUDA 12.x, cu132.* for CUDA 13.x)"
        ) from e

    from .config import Exl3Config  # noqa: F401  (imports register the class)

    try:
        register_quantization_config("exl3")(Exl3Config)
    except ValueError:
        return  # already registered in this process
    _patch_embedding_layers(logger)
    logger.info("orcasaq2 armed: quant_method 'exl3' is now loadable")


def _patch_embedding_layers(logger):
    """Give VocabParallelEmbedding and ParallelLMHead the quant config the model withholds.

    vLLM consults quant_config only for layers that are constructed with one, and
    qwen3_next.py builds both `embed_tokens` and `lm_head` without passing it:

        self.embed_tokens = VocabParallelEmbedding(self.vocab_size, config.hidden_size)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size, prefix=...)

    So a pack whose lm_head is a 6-bit EXL3 tensor, or whose embedding is int8 rows, cannot
    load however the plugin is written -- the loader goes looking for `lm_head.weight` and
    `embed_tokens.weight` and fails on the names it finds instead. Editing vLLM in place would
    push the fix onto every user's site-packages, so wrap the two constructors instead and
    supply the current config when the caller omitted one. Layers of other quant methods are
    untouched: Exl3Config.get_quant_method returns a method only for EXL3 tensors, and falls
    back to a dense weight when the pack has one.
    """
    from vllm.model_executor.layers import vocab_parallel_embedding as vpe
    from .config import Exl3Config

    if getattr(vpe.VocabParallelEmbedding, "_orca_patched", False):
        return
    import inspect

    def _current_exl3_config():
        from vllm.config import get_current_vllm_config
        try:
            cfg = get_current_vllm_config()
        except Exception:
            return None
        qc = getattr(cfg, "quant_config", None) or \
             getattr(getattr(cfg, "model_config", None), "quant_config", None)
        return qc if isinstance(qc, Exl3Config) else None

    for cls in (vpe.VocabParallelEmbedding, vpe.ParallelLMHead):
        def make(orig_init):
            sig = inspect.signature(orig_init)

            def __init__(self, *args, **kwargs):
                # ParallelLMHead forwards quant_config to its parent POSITIONALLY, and the
                # parent is patched too, so injecting a keyword blindly collides. Bind against
                # the real signature and only fill the slot when it is genuinely empty.
                try:
                    bound = sig.bind_partial(self, *args, **kwargs)
                except TypeError:
                    return orig_init(self, *args, **kwargs)
                if bound.arguments.get("quant_config") is None:
                    qc = _current_exl3_config()
                    if qc is not None:
                        bound.arguments["quant_config"] = qc
                        return orig_init(*bound.args, **bound.kwargs)
                return orig_init(self, *args, **kwargs)
            return __init__

        cls.__init__ = make(cls.__init__)
    vpe.VocabParallelEmbedding._orca_patched = True
    logger.info("orcasaq2: embed_tokens / lm_head will receive the EXL3 quant config")
