"""The output head and the embedding table, for vLLM's VocabParallelEmbedding family.

Two different problems wearing the same base class:

* `lm_head` is a matmul. exllamav3 packs it as an ordinary EXL3 trellis tensor when the
  recipe asks for it (`head_bits`), so it needs no new code at all -- `Exl3LinearMethod`
  already does exactly the right thing, and ParallelLMHead is explicitly exempted from
  having to implement `embedding()`. All that was missing was the dispatch.

* `embed_tokens` is a lookup. exllamav3 refuses to quantize it ("No quant scheme for
  Embedding") and keeps it in system RAM instead (`caps["prefer_cpu"]`), so its native engine
  never pays VRAM for it. vLLM does: VocabParallelEmbedding is a GPU tensor, and for this
  248,320-token vocabulary that is 2.54 GB in bf16. int8 per row halves it, and the ablation
  measured that as free (embed+head at int8 scored KLD 0.0338 against 0.0341 for bf16).
"""
import torch
from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase

from .config import Exl3ShardParam


class Exl3EmbeddingMethod(QuantizeMethodBase):
    """int8 rows with a per-row scale, dequantized on the GATHERED rows.

    Dequantizing the whole table at load would hand the memory straight back. The gather
    output is [num_tokens, 5120], so scaling there costs nothing and the table stays int8.
    """

    def create_weights(self, layer, input_size_per_partition, output_partition_sizes,
                       input_size, output_size, params_dtype, **extra_weight_attrs):
        # A pack may carry either form. Which one it is, is only knowable from what the loader
        # delivers, so register both and decide afterwards. `extra_weight_attrs` is deliberately
        # not applied: it carries vLLM's vocab-sharding weight_loader, which would assert
        # against our empty placeholders.
        for suffix in ("qweight", "scales", "weight"):
            layer.register_parameter(suffix, Exl3ShardParam(suffix=suffix))
        layer.exl3_params_dtype = params_dtype

    def process_weights_after_loading(self, layer):
        q = layer.qweight.shards.get(0)
        s = layer.scales.shards.get(0)
        w = layer.weight.shards.get(0)
        dev = next((t.device for t in (q, w) if t is not None and t.device.type == "cuda"),
                   torch.device("cuda", torch.cuda.current_device()))
        for p in ("qweight", "scales", "weight"):
            if hasattr(layer, p):
                delattr(layer, p)
        if q is None:
            if w is None:
                raise RuntimeError("embed_tokens: neither an int8 table nor a dense one loaded")
            layer.exl3_embed_q = None
            layer.exl3_embed = torch.nn.Parameter(w.to(dev), requires_grad=False)
            return
        if s is None:
            raise RuntimeError("embed_tokens: qweight without scales")
        layer.exl3_embed = None
        layer.exl3_embed_q = torch.nn.Parameter(q.to(dev), requires_grad=False)
        layer.exl3_embed_s = torch.nn.Parameter(
            s.to(dev, layer.exl3_params_dtype).reshape(-1, 1), requires_grad=False)

    def embedding(self, layer, input_: torch.Tensor) -> torch.Tensor:
        if layer.exl3_embed_q is None:
            return layer.exl3_embed[input_]
        # index directly rather than through F.embedding: the table is int8, and the scale is
        # applied to the gathered rows so the table itself is never materialised in bf16.
        rows = layer.exl3_embed_q[input_].to(layer.exl3_embed_s.dtype)
        return rows * layer.exl3_embed_s[input_]

    def apply(self, layer, x: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        if layer.exl3_embed_q is None:
            w = layer.exl3_embed
        else:
            w = layer.exl3_embed_q.to(x.dtype) * layer.exl3_embed_s
        out = torch.nn.functional.linear(x, w.to(x.dtype))
        return out + bias if bias is not None else out
