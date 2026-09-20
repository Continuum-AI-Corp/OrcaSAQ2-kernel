#!/usr/bin/env python3
"""How much does the Hadamard rotation cost, and is it worth it?

The rotation only pays for itself if the CUDA kernel is used. A naive PyTorch FWHT
(10 stacked butterfly stages, each a full read/write of the tensor) is slower than the
bf16 matmul it is supposed to accelerate, which inverts the whole argument for rotating.

    python bench/bench_rotation.py [--dim 5120] [--block 1024]
"""
import argparse, math, time
import torch


def ref_fwht(x, block):
    """Reference PyTorch FWHT -- correct, and deliberately kept as the slow baseline."""
    shape = x.shape
    x = x.reshape(-1, block).clone()
    h = 1
    while h < block:
        x = x.view(-1, block // (2 * h), 2, h)
        a, b = x[:, :, 0, :], x[:, :, 1, :]
        x = torch.stack((a + b, a - b), dim=2)
        h *= 2
    return x.reshape(shape) / math.sqrt(block)


def bench(fn, n=200):
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    t = time.time()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t) / n * 1e6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", type=int, default=5120)
    ap.add_argument("--out", type=int, default=17408)
    ap.add_argument("--block", type=int, default=1024)
    ap.add_argument("--speedup-4bit", type=float, default=3.0,
                    help="how much faster a 4-bit matmul is than bf16 on this card")
    a = ap.parse_args()
    from fast_hadamard_transform import hadamard_transform

    dev = "cuda"
    torch.manual_seed(0)
    signs = torch.where(torch.rand(a.dim, device=dev) < 0.5, -1.0, 1.0).to(torch.bfloat16)
    W = torch.randn(a.out, a.dim, device=dev, dtype=torch.bfloat16)

    # correctness first: the kernel must match the reference exactly
    x32 = torch.randn(4, a.block, device=dev, dtype=torch.float32)
    s32 = torch.where(torch.rand(a.block, device=dev) < 0.5, -1.0, 1.0)
    err = (ref_fwht(x32 * s32, a.block)
           - hadamard_transform(x32 * s32, scale=1 / math.sqrt(a.block))).abs().max().item()
    print(f"correctness: max |kernel - reference| = {err:.2e}"
          f"  {'OK' if err < 1e-3 else 'MISMATCH'}\n")

    print(f"dim={a.dim} out={a.out} block={a.block}, assuming 4-bit matmul is "
          f"{a.speedup_4bit:.1f}x faster than bf16\n")
    print(f"{'batch':>6} {'kernel':>9} {'pytorch':>10} {'bf16 mm':>9} "
          f"{'4bit+rot':>10} {'speedup':>8}")
    for bt in (1, 4, 16, 64, 256):
        x = torch.randn(bt, a.dim, device=dev, dtype=torch.bfloat16)
        tk = bench(lambda: hadamard_transform((x * signs).reshape(-1, a.block),
                                              scale=1 / math.sqrt(a.block)))
        tp = bench(lambda: ref_fwht((x * signs).float(), a.block), n=50)
        tm = bench(lambda: x @ W.T)
        q = tm / a.speedup_4bit + tk
        print(f"{bt:>6} {tk:>8.1f}us {tp:>9.1f}us {tm:>8.1f}us {q:>9.1f}us "
              f"{tm/q:>7.2f}x")
    print("\nThe pytorch column is why this repo exists: without the CUDA kernel the "
          "rotation costs more than the quantization saves.")


main()
