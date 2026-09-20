# OrcaSAQ2 kernel

Rotation-aware 4-bit quantization for LLMs: GPTQ in a Hadamard-rotated basis, packed into
the standard GPTQ layout so stock vLLM loads it, with the runtime rotation supplied by a
CUDA kernel.

The rotation is what makes low-bit quantization hold up — it flattens the outliers that
otherwise dominate each group's scale. The catch is that it has to be applied to the
activations at inference, and if you do that in plain PyTorch it costs more than the
quantization saves.

## Measured on one H200, Qwen3.8-27B shapes (dim 5120 → 17408)

| batch | CUDA kernel | PyTorch FWHT | bf16 matmul | 4-bit + rotation | speedup |
|---:|---:|---:|---:|---:|---:|
| 1 | 17.5 µs | 221.2 µs | 48.6 µs | 33.7 µs | **1.44×** |
| 4 | 13.5 µs | 200.7 µs | 49.3 µs | 29.9 µs | **1.65×** |
| 16 | 13.6 µs | 200.5 µs | 50.5 µs | 30.5 µs | **1.66×** |
| 64 | 12.7 µs | 201.3 µs | 55.2 µs | 31.1 µs | **1.77×** |
| 256 | 12.5 µs | 201.1 µs | 78.5 µs | 38.7 µs | **2.03×** |

Two things to read off this table.

**The kernel is 12–17× faster than the PyTorch implementation**, and that difference decides
whether rotating is viable at all: at 200 µs the rotation is *slower* than the bf16 matmul
it is meant to accelerate, so a rotated 4-bit model would run slower than the bf16 one it
replaces. At 13 µs it costs a quarter of the matmul and the quantization wins.

**Speedup grows with batch** (1.44× → 2.03×) because the rotation's cost is launch-bound,
not bandwidth-bound — it barely moves from batch 1 to batch 256 while the matmul grows.
Concurrent serving therefore gets a better deal than single-stream decode.

Quality, same recipe, measured against the bf16 source on wikitext-2
(4 × 1024 tokens, KL divergence over the full 248,320-token vocabulary):

| build | bits/weight | KLD | Top-1 agreement |
|---|---:|---:|---:|
| GPTQ int4 g128 + rotation + clipping, 8-bit head | 4.63 | **0.0195** | 93.8% |
| the same without rotation | 4.63 | ~0.042 | — |

## What is here

```
src/
  rotate_quant.py        rotation + quantization, GPU (the working implementation)
  rotate_quant_numpy.py  the same as a readable numpy reference
  gptq.py                GPTQ with Hadamard rotation and per-group MSE clipping
  pack_gptq.py           pack the result into the GPTQ layout vLLM reads
  ste.py                 straight-through estimator, for quantization-aware training
bench/
  bench_rotation.py      kernel vs PyTorch vs matmul; run this first on a new card
  bench_pack.py          pack/unpack round trip and zero-point snapping
```

## Install

The rotation kernel is Dao-AILab's; it is not vendored here.

```bash
export CUDA_HOME=/usr/local/cuda TORCH_CUDA_ARCH_LIST="9.0"   # 9.0 = H100/H200
git clone https://github.com/Dao-AILab/fast-hadamard-transform
pip install --no-build-isolation ./fast-hadamard-transform
```

Two build notes, both of which cost us time:

- The PyPI sdist is incomplete (`csrc/*.cpp` missing) — build from the git checkout.
- If `nvcc` reports `cannot execute 'cc1plus'`, gcc and its C++ frontend are at different
  versions. Fix the toolchain rather than forcing `CC=g++`:
  `sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 100 --slave /usr/bin/g++ g++ /usr/bin/g++-11`

Verify before trusting anything:

```bash
python bench/bench_rotation.py     # must print max |kernel - reference| = 0.00e+00
python bench/bench_pack.py         # must print round trip: codes True  zeros True
```

## The one subtlety in packing

vLLM stores the zero point as a 4-bit **code** and dequantizes as `(q - zcode) * scale`.
GPTQ's own zero point is an arbitrary float, so rounding it at export time is lossy — we
measured 9.3% weight error and 0.106 output error doing it that way.

Snapping the zero onto the integer grid **before** quantizing, so GPTQ compensates for the
snap in the columns it has not reached yet, brings the round-trip error to exactly zero and
halves the output error to 0.050. `pack_gptq.py` does this; `bench/bench_pack.py` checks it.

## Reusing this on another model

Most of it is model-agnostic:

| piece | change needed |
|---|---|
| Hadamard kernel | none — sizes 128 … 32768 all supported |
| `rotate_quant.py` | none — plain tensor ops |
| `gptq.py` | none |
| `pack_gptq.py` | the `TARGETS` layer-name list, and how the model builds rotary embeddings |
| vLLM-side rotation | which layers to hook |

For a new dense architecture that is a list of layer names. Mixture-of-experts needs extra
work on the expert weights; the kernel and the quantization logic still do not change.

## Deployed in vLLM

The rotation runs in production through a `vllm.general_plugins` entry point
(`bonsai-quant/scripts/orca_rotate`), which is the only hook that reaches vLLM's EngineCore and
worker processes. Each layer's sign vector is resolved once in `process_weights_after_loading`
and parked on the layer; `apply()` then calls a `torch.library` custom op wrapping this kernel,
so the forward stays `fullgraph`-compilable and CUDA-graph-capturable (verified bit-identical to
eager). vLLM picks the Machete 4-bit kernel on Hopper. Serving fidelity against the pack's own
bytes: mean |Δ log p| = 0.022 nat over 6k tokens, for both the stock and the abliterated pack.

## What the rotation costs in a real decode step

Measured on the live deployment (400 rotated layers, batch 1, H200):

| | per token |
|---|---|
| rotation, eager (mul + Hadamard per layer, 800 launches) | 7.04 ms |
| rotation, inside a CUDA graph | **3.69 ms** |
| reading all 4-bit weights at 4.8 TB/s (the matmul floor) | 2.67 ms |
| whole decode step as served | ~21 ms (47 tok/s) |

So at batch 1 the rotation costs more than the entire 4-bit weight read, and ~17 ms of the
step is neither: the GDN recurrent kernels, attention, sampling and framework overhead of this
hybrid architecture, which bf16 pays identically. Quantization can only shrink the matmul
share, which is why the batch-1 gain is ~1.3× as deployed and would be ~1.6× with the rotation
made free; at batch 32+ the matmul share grows and so does the gap.

## The obvious next improvement

Two steps, both pack-time or kernel-level, neither touching the quantized integers:

1. **Fold the sign vector away** (removes 400 elementwise launches). For inputs that come out of
   an RMSNorm (`qkv_proj`, `gate_up_proj`, `in_proj_qkvz`) fold `s` into the norm weight γ.
   For `down_proj`, fold `s` into `up_proj`'s output rows (quantize `-w` for flipped rows at pack
   time; the uint4b8 grid is not sign-symmetric, so re-quantize rather than negate codes). For
   `o_proj`, fold into `v_proj`'s output rows -- attention mixes positions, not channels, so
   `attn(V·s) = attn(V)·s`. For the GDN `out_proj`, flip the matching columns of the delta-rule
   state and fold into the gated norm's γ.
2. **Fuse the Hadamard into the preceding RMSNorm kernel** for the norm-fed inputs, as QuaRot
   does; `o_proj` / `out_proj` / `down_proj` keep a standalone transform. That leaves ~200
   launches per token, inside the graph, i.e. under 1 ms.
