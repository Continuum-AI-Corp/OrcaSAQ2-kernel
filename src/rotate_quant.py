#!/usr/bin/env python3
"""GPU implementation of the reverse-engineered Bonsai 2 quantizer.

Numerically equivalent to the numpy reference in bonsai_quant.py, but fast enough to
run over a 27B checkpoint. Everything happens in float32 on the GPU.

    fold  : W -> W @ T,   T = diag(signs) @ blockdiag(H_1024) / sqrt(1024)
    assign: per 128-column group, zero the 42 smallest |w|, sign() the other 86
    scale : s = mean(|w|) over the 86 retained, stored fp16
"""
import torch

ZEROS_PER_GROUP = 42        # floor(128/3) -- the balanced / maximum-entropy ternary point
GROUP = 128
BLOCK = 1024


def fwht(x, block=BLOCK, signs=None, inverse=False):
    """Vectorised natural-order Sylvester FWHT with 1/sqrt(block) normalisation."""
    shape = x.shape
    x = x.float()
    if signs is not None and not inverse:
        x = x * signs
    x = x.reshape(-1, block)
    h = 1
    while h < block:
        x = x.view(-1, block // (2 * h), 2, h)
        a = x[:, :, 0, :]
        b = x[:, :, 1, :]
        x = torch.stack((a + b, a - b), dim=2)
        h *= 2
    x = x.reshape(shape) / (block ** 0.5)
    if signs is not None and inverse:
        x = x * signs
    return x


def ternarize(Wf, zeros=ZEROS_PER_GROUP, group=GROUP):
    """Balanced ternary assignment + least-squares scale, on the rotated weights."""
    rows, cols = Wf.shape
    G = Wf.reshape(rows, cols // group, group)
    absG = G.abs()
    kth = absG.kthvalue(zeros, dim=-1, keepdim=True).values
    keep = absG > kth
    t = torch.where(keep, torch.sign(G), torch.zeros((), device=G.device, dtype=G.dtype))
    n = keep.sum(-1).clamp(min=1)
    s = (absG * keep).sum(-1) / n
    return t.reshape(rows, cols), s.half()


def simulate(W, signs, block=BLOCK, zeros=ZEROS_PER_GROUP, rotate=True):
    """Quantize, then map back to the original basis so a stock model can run it."""
    Wf = fwht(W, block, signs) if rotate else W.float()
    t, s = ternarize(Wf, zeros)
    Wq = t * s.float().repeat_interleave(GROUP, dim=1)
    return fwht(Wq, block, signs, inverse=True) if rotate else Wq


def pack(t, s):
    """{-1,0,1} -> MLX affine 2-bit words (16 codes per uint32, lane i at bit 2i)."""
    rows, cols = t.shape
    c = (t + 1).to(torch.int64).reshape(rows, cols // 16, 16)
    sh = 2 * torch.arange(16, device=t.device, dtype=torch.int64)
    return (c << sh).sum(-1), s


def codes_of(words, cols):
    """uint32 words (held as int64) -> code array in {0,1,2,3}."""
    rows = words.shape[0]
    out = torch.empty((rows, words.shape[1], 16), dtype=torch.int8, device=words.device)
    for l in range(16):
        out[:, :, l] = ((words >> (2 * l)) & 3).to(torch.int8)
    return out.reshape(rows, cols)
