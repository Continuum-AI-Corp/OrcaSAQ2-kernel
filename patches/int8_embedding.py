"""Teach exllamav3's native engine to read an int8-packed embedding table.

Why this exists. exllamav3 will not quantize an embedding ("No quant scheme for Embedding"),
so a pack it writes always carries `embed_tokens.weight` in bf16 -- 2.54 GB for a 248,320-token
vocabulary. On a 16 GB card that is a sixth of the budget spent on a lookup table. Packing it
as int8 rows with a per-row scale costs, measured on this model, a weight-domain relative error
of 0.0090 and ΔKLD within the noise floor, and gives back 1.27 GB.

Where it hooks in. Every read of the table in exllamav3's Embedding module goes through
`self.embedding(ids)` -- both the plain path and the multimodal indexed-embedding path. So the
whole patch is: at load time, if the checkpoint has `qweight`/`scales` instead of `weight`,
install a module with the same call signature that gathers int8 rows and scales them. None of
the surrounding masking logic changes.

The scale is applied to the GATHERED rows, not to the table. Dequantizing the table at load
would hand the 1.27 GB straight back; the gather output is [tokens, hidden], which is nothing.

    python patches/int8_embedding.py --check     # verify it applies to the installed exllamav3
    import orca_exl3.patches.int8_embedding as p; p.apply()   # or call it from your own loader
"""
import torch
from torch import nn


class Int8EmbeddingTable(nn.Module):
    """Drop-in for nn.Embedding over an int8 table with per-row scales."""

    def __init__(self, qweight: torch.Tensor, scales: torch.Tensor, out_dtype: torch.dtype):
        super().__init__()
        self.register_buffer("qweight", qweight, persistent=False)
        self.register_buffer("scales", scales.reshape(-1, 1), persistent=False)
        self.out_dtype = out_dtype

    @property
    def weight(self):
        # Only get_tensors()/weights_numel() and the TP producer path touch .weight. Materialise
        # on demand so those keep working; the serving path never reaches here.
        return self.qweight.to(self.out_dtype) * self.scales

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        # index directly rather than through F.embedding: the table is int8, and torch.embedding
        # asks for a float weight. Both casts happen on the GATHERED rows, never on the table.
        return self.qweight[ids].to(self.out_dtype) * self.scales[ids].to(self.out_dtype)


def apply():
    """Wrap exllamav3.modules.embedding.Embedding.load so it accepts either storage form."""
    from exllamav3.modules import embedding as _emb

    if getattr(_emb.Embedding, "_orca_int8_patched", False):
        return False
    orig_load = _emb.Embedding.load

    def load(self, device, **kwargs):
        stc = self.config.stc
        qkey, skey = self.key + ".qweight", self.key + ".scales"
        has_q = False
        try:
            has_q = stc.has_tensor(qkey) if hasattr(stc, "has_tensor") else False
        except Exception:
            has_q = False
        if not has_q:
            try:
                stc.get_tensor(qkey, "cpu")
                has_q = True
            except Exception:
                has_q = False
        if not has_q:
            return orig_load(self, device, **kwargs)
        self.device = device
        # no_defer is load-bearing. Inside model.load() exllamav3 runs a DEFERRED load: get_tensor
        # returns an unfilled (zeroed) tensor immediately, queues the read, and fills that same
        # tensor object later. Anything that copies it in between -- a .to(dtype), a .clone(), a
        # non-view reshape -- keeps the zeros, and the fill lands somewhere nobody reads. That
        # failure is silent: the table loads, the model runs, and every token embeds to 0, which
        # comes out as a stream of "!". Reading eagerly costs a one-off load-time pause and makes
        # the tensors safe to touch.
        q = stc.get_tensor(qkey, device, no_defer=True)
        s = stc.get_tensor(skey, device, allow_bf16=True, no_defer=True)
        self._numel = q.numel()
        self.embedding = Int8EmbeddingTable(q, s, self.out_dtype or torch.half)
        return None

    _emb.Embedding.load = load
    _emb.Embedding._orca_int8_patched = True
    return True


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    ok = apply()
    print("已应用" if ok else "已经打过补丁")
    if a.check:
        from exllamav3.modules import embedding as _e
        print("Embedding.load ->", _e.Embedding.load.__module__, _e.Embedding.load.__qualname__)
