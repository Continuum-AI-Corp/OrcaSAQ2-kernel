#!/usr/bin/env python3
"""Reference implementation of the Bonsai 2 quantizer, reverse-engineered.

    fold : W  ->  W @ T,  T = diag(signs) @ blockdiag(H_1024) / sqrt(1024)
    assign: per 128-column group, zero the 42 smallest |w|, sign() the other 86
    scale : s = mean(|w|) over the 86 retained  (== the least-squares scale for +-1 codes)
    pack  : MLX affine 2-bit, code = t + 1, biases = -scales

`simulate` returns the quantized weight mapped back to the original basis, so a stock
model runs it with no runtime transform:  x @ (Wq @ T^T)^T  ==  fwht(x) @ Wq^T.
"""
import numpy as np

ZEROS_PER_GROUP = 42           # = floor(128 / 3): the balanced / max-entropy ternary point
GROUP = 128


def fwht(x, block, signs=None, inverse=False):
    """Row-wise (x * signs) @ blockdiag(H_block) / sqrt(block), natural Sylvester order."""
    shape = x.shape
    x = x.astype(np.float32)
    if signs is not None and not inverse:
        x = x * signs
    x = x.reshape(-1, block).copy()
    h = 1
    while h < block:
        for i in range(0, block, h * 2):
            a = x[:, i:i + h].copy()
            b = x[:, i + h:i + 2 * h].copy()
            x[:, i:i + h] = a + b
            x[:, i + h:i + 2 * h] = a - b
        h *= 2
    x = (x / np.sqrt(block)).reshape(shape)
    if signs is not None and inverse:
        x = x * signs
    return x


def ternarize(Wf, zeros=ZEROS_PER_GROUP, group=GROUP):
    """Balanced ternary assignment + least-squares scale. Wf is already in the rotated basis."""
    rows, cols = Wf.shape
    G = Wf.reshape(rows, cols // group, group)
    order = np.argsort(np.abs(G), axis=-1, kind="stable")
    t = np.sign(G).astype(np.float32)
    np.put_along_axis(t, order[..., :zeros], 0.0, axis=-1)
    keep = t != 0
    s = np.where(keep, np.abs(G), 0.0).sum(-1) / np.maximum(keep.sum(-1), 1)
    return t.reshape(rows, cols), s.astype(np.float16)


def pack_mlx(t, s):
    """t in {-1,0,1} -> MLX affine 2-bit (uint32 words, fp16 scales, biases = -scales)."""
    rows, cols = t.shape
    c = (t + 1).astype(np.uint32).reshape(rows, cols // 16, 16)
    words = np.bitwise_or.reduce(c << (2 * np.arange(16, dtype=np.uint32)), axis=-1)
    return words.astype("<u4"), s, (-s.astype(np.float32)).astype(np.float16)


def quantize(W, signs, block=1024, zeros=ZEROS_PER_GROUP):
    """W [out, in] float -> (words, scales, biases) exactly as the pack stores them."""
    Wf = fwht(W, block, signs)
    t, s = ternarize(Wf, zeros)
    return pack_mlx(t, s)


def simulate(W, signs, block=1024, zeros=ZEROS_PER_GROUP):
    """Quantize and map back to the original basis, for drop-in evaluation."""
    Wf = fwht(W, block, signs)
    t, s = ternarize(Wf, zeros)
    Wq = t * np.repeat(s.astype(np.float32), GROUP, axis=1)
    return fwht(Wq, block, signs, inverse=True)
