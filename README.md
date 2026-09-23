# Orca QTIP-SAQ — serving kernels and loaders

Run EXL3 trellis-quantized checkpoints of **Qwen3.8-27B** on vLLM or on exllamav3's own engine.

The weights are a QTIP-style trellis code with a searched mixed-precision allocation. This
repository is only the part you need to *run* them: a vLLM quantization plugin, a patch that
teaches exllamav3 to read the packed embedding, tuned launch presets, and a Docker image.

## Why a plugin at all

An EXL3 checkpoint stores each projection self-contained — a packed tail-biting trellis plus
its own incoherence vectors — and the bitrate is implied by the trellis width. Three
consequences shape everything here:

* Shards of a fused vLLM layer (q/k/v inside `qkv_proj`, gate/up inside `gate_up_proj`) are
  separate tensors with different shapes and, under a mixed allocation, different bitwidths.
  There is no single tensor to allocate, so each shard is kept as its own buffer.
* Nothing about the shapes is known until the weights arrive, so `create_weights` registers
  placeholders and captures each shard as it loads.
* vLLM's Qwen3.5 model file constructs `embed_tokens` and `lm_head` **without** passing a
  quantization config, so a packed head could never load however the plugin was written. The
  plugin wraps those two constructors at registration time rather than asking you to edit your
  own site-packages.

llama.cpp is not an option for these weights: it reads only its own i-quant codebooks, so a
GGUF would mean re-quantizing with a measurably weaker quantizer.

## vLLM

Install the plugin into whatever environment already has vLLM, then launch with a preset:

```bash
pip install -e .                                   # registers via vllm.general_plugins
PRESET=16gb bash presets/serve.sh /path/to/checkpoint 8000 orca
```

`PRESET` takes `16gb`, `16gb-mtp`, `80gb`, or a path to your own env file. Preset values are
**defaults** — anything already in your environment wins, so `PRESET=16gb MAXLEN=8192 bash
presets/serve.sh ...` does what it says. `DRYRUN=1` prints the assembled command and exits,
which is the quickest way to see what a preset actually does.

The script finds its interpreter in this order: `$PYTHON`, a `.venv-vllm`/`.venv` beside the
repo, then `python3` on PATH. If the one it picks has no vLLM it says so in one line instead
of failing somewhere deeper. It also probes for the CUDA toolkit flashinfer needs to JIT
against, rather than assuming a path.

### Docker

```bash
bash docker/build.sh
docker run --gpus all -v /path/to/checkpoint:/model:ro -p 8000:8000 \
  -e PRESET=16gb -e API_KEY=... orca-exl3-vllm:0.30.0
```

The entrypoint sets the container defaults (`/model`, `0.0.0.0`) and then execs the same
`presets/serve.sh`, so every knob behaves identically on both paths and no option can exist on
one and not the other.

`presets/serve.sh` binds to **127.0.0.1 by default** on a host. vLLM's own default is `0.0.0.0`, which on
a host with a public IP means an unauthenticated inference endpoint on the open internet the
moment the server starts. Exposing it is `BIND=0.0.0.0`, and you should set `API_KEY` and a
firewall whitelist first.

### Reasoning and tool calling

All three presets turn both parsers on:

```
REASONING_PARSER=qwen3     # -> --reasoning-parser qwen3
TOOL_PARSER=qwen3_xml      # -> --enable-auto-tool-choice --tool-call-parser qwen3_xml
```

Without the reasoning parser the thinking block arrives inside `content` and every harness has
to strip it; with it, it comes back in a separate `reasoning` field. `--tool-call-parser` is
inert on its own, so `TOOL_PARSER` sets `--enable-auto-tool-choice` with it. Use **qwen3_xml**,
not `hermes`: this model's chat template emits the XML form,
`<tool_call><function=name><parameter=k>v</parameter></function></tool_call>`, and the hermes
parser expects JSON inside the tag, so it would silently return the call as plain text.

## exllamav3 (native)

Lighter than vLLM — no paged-KV manager, no torch.compile — and the better choice for a single
user on a consumer card. It reads the checkpoint directly, except for the int8 embedding:

```python
import patches.int8_embedding as p; p.apply()   # before Config.from_directory
```

Without the patch, a checkpoint whose embedding is packed will fail with
`Required tensor model.language_model.embed_tokens.weight not found`.

## Measured serving numbers

All on one card, this checkpoint, same prompt. `16 GB` rows are a 15.5 GiB cap on a larger
card, which is what a 5080 leaves after the driver.

| engine | card | MTP | KV pool | single stream |
|---|---|---|---:|---:|
| vLLM | 16 GB | off | 59,753 tok | 64.7 tok/s |
| vLLM | 16 GB | on | 35,617 tok | **111.2 tok/s** |
| vLLM | H200 | on | 814,188 tok | 112.0 tok/s |
| exllamav3 | 24 GB+ | off | — | — |
| exllamav3 | 24 GB+ | on | — | 1.37x of off |

**MTP is a separate download.** It buys 1.72x on vLLM and costs 40% of the context, because the
draft head needs its own KV out of the same pool. On exllamav3 it is a *net* +10.1 GB — the MTP
head loads as an independent component with a full cache of its own — so it does not fit a
16 GB card at all. Take the MTP variant only if your engine and card suit it.

## What is in here

```
orca_exl3/         vLLM plugin (pip installable)
patches/           exllamav3 int8-embedding patch
presets/serve.sh   the launcher -- the only place arguments are built
presets/*.env      launch configs, every value measured
docker/            serving image; its entrypoint execs presets/serve.sh
```

## License

Apache-2.0. EXL3 is a streamlined variant of [QTIP](https://github.com/Cornell-RelaxML/qtip)
(Cornell RelaxML, [arXiv 2406.11235](https://arxiv.org/abs/2406.11235)); the trellis kernels
come from [exllamav3](https://github.com/turboderp-org/exllamav3).
