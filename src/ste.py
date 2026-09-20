#!/usr/bin/env python3
"""Differentiable Bonsai quantizer: the balanced-42 ternary assignment with a
straight-through estimator, so latent weights can be trained through it.

Forward  : W -> fold -> balanced-42 ternary + LS scale -> unfold
Backward : identity onto the latent weights (STE)

Both the fold and the unfold are exact linear maps, so the only non-differentiable step is
the assignment itself, which the STE passes through.
"""
import torch
import bonsai_quant_torch as bqt


class _STE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Wf, zeros):
        t, s = bqt.ternarize(Wf, zeros)
        return t * s.float().repeat_interleave(bqt.GROUP, dim=1)

    @staticmethod
    def backward(ctx, g):
        return g, None


def quantize_ste(W, signs, zeros=bqt.ZEROS_PER_GROUP, block=1024):
    """Simulated-quantized weight in the ORIGINAL basis, differentiable w.r.t. W."""
    Wf = bqt.fwht(W, block, signs)
    Wq = _STE.apply(Wf, zeros)
    return bqt.fwht(Wq, block, signs, inverse=True)


class QuantLinearWrapper(torch.nn.Module):
    """Wrap an nn.Linear so its weight is the STE-quantized view of a trainable latent."""

    def __init__(self, linear, signs, zeros=bqt.ZEROS_PER_GROUP):
        super().__init__()
        self.latent = torch.nn.Parameter(linear.weight.detach().float().clone())
        self.register_buffer("signs", signs.float())
        self.bias = linear.bias
        self.zeros = zeros

    def forward(self, x):
        W = quantize_ste(self.latent, self.signs, self.zeros)
        return torch.nn.functional.linear(x, W.to(x.dtype), self.bias)

    @torch.no_grad()
    def export(self):
        Wf = bqt.fwht(self.latent.data, 1024, self.signs)
        return bqt.ternarize(Wf, self.zeros)
