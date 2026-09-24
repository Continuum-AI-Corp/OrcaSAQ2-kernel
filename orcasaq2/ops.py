"""The EXL3 shard GEMM, wrapped as a PyTorch custom op.

vLLM compiles the model with torch.compile. Dynamo cannot trace a raw pybind11 function --
it has no source file to inline -- so the engine died at startup with

    torch._dynamo.exc.Unsupported: Attempted to call function marked as skipped
      module: exllamav3_ext, qualname: ...had_r_128
      skip reason: cannot determine source file for exllamav3_ext

Serving with --enforce-eager avoids it and is how this plugin was first checked, but that
gives up CUDA graphs, which is most of the decode throughput. A custom op is the supported
way to tell Dynamo "this is opaque, here is the shape it returns": it stays one node in the
graph and everything around it still compiles.

The whole shard is one op rather than one op per extension call, so the row-count dispatch
below stays ordinary eager Python at runtime instead of becoming a graph-level branch.
"""
import os
import torch
from torch import Tensor

# exllamav3's own dispatch point (exl3.py: AUTO_RECONSTRUCT_THRESHOLD). At or below it the
# fused kernel reads the packed trellis directly; above it, rebuilding the weight once and
# handing it to cuBLAS is cheaper than one fused GEMM per row.
GEMM_MAX_ROWS = int(os.environ.get("ORCA_EXL3_GEMM_MAX_ROWS", "144"))


def exl3_K(trellis: Tensor):
    """Steps per 16-weight tile. The mul1 half rates (1.5/2.5/3.5) are NOT integers: a 3.5-bit
    tensor packs 56 int16 per tile, and 56 // 16 silently gives 3, which the kernel rejects
    with "packed dimension 2 is incorrect size". exllamav3 keeps the float (exl3.py:67) and
    the kernel accepts it."""
    k = trellis.shape[-1] / 16
    return int(k) if float(k).is_integer() else k


@torch.library.custom_op("orcasaq2::shard_gemm", mutates_args=())
def shard_gemm(x: Tensor, trellis: Tensor, suh: Tensor, svh: Tensor,
               oc: int, k: float, mcg: bool, mul1: bool) -> Tensor:
    # k is declared float because the mul1 half rates are 1.5/2.5/3.5; the extension takes
    # either, but a custom-op schema has to pick one type, so narrow it back here.
    from exllamav3.ext import exllamav3_ext as ext
    if float(k).is_integer(): k = int(k)
    rows, ic = x.shape
    y = torch.empty((rows, oc), dtype=torch.half, device=x.device)
    if rows <= GEMM_MAX_ROWS:
        # (A, trellis, C, suh, A_had_scratch, svh, kernel, mcg, mul1, acc_mode);
        # kernel -1 asks the extension for its preferred kernel for this shape.
        ext.exl3_gemm(x, trellis, y, suh, torch.empty_like(x), svh, -1, mcg, mul1, 0)
    else:
        # Reconstruct into a SCRATCH buffer, freed on exit -- keeping it resident would
        # undo the compression.
        xh = torch.empty_like(x)
        ext.had_r_128(x, xh, suh, None, 1.0)
        w = torch.empty((ic, oc), dtype=torch.half, device=x.device)
        ext.reconstruct(w, trellis, k, mcg, mul1)
        ext.hgemm_recon(xh, w, y)
        del w, xh
        ext.had_r_128(y, y, None, svh, 1.0)
    return y


@shard_gemm.register_fake
def _shard_gemm_fake(x: Tensor, trellis: Tensor, suh: Tensor, svh: Tensor,
                     oc: int, k: float, mcg: bool, mul1: bool) -> Tensor:
    return x.new_empty((x.shape[0], oc), dtype=torch.half)
