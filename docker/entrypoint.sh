#!/usr/bin/env bash
# Serve an EXL3 pack. Every knob is an env var so one image covers an 80 GB datacenter card
# and a 16 GB consumer card:
#
#   MODEL          checkpoint path inside the container (default /model)
#   SERVED_NAME    model id clients use (default exl3)
#   MAXLEN         context length (default 32768)
#   GPU_FRAC       fraction of VRAM vLLM may use (default 0.90)
#   MAXSEQS        concurrent sequences (default 64; set 1 on a 16 GB card)
#   KV_DTYPE       auto | fp8_e4m3 — fp8 halves KV (64 -> 32 KB/token) at no measurable
#                  quality cost and is the first thing to reach for when context will not fit
#   CPU_OFFLOAD_GB keep this many GB of weights in host RAM, streamed each step. Last resort:
#                  roughly +40 ms/token per GB at PCIe 4.0 x16.
#   KV_OFFLOAD_GB  three-tier GPU/CPU/disk KV via LMCache. Helps many sessions or long reusable
#                  prefixes; does NOT shrink one sequence's active attention window.
#   REASONING_PARSER  e.g. qwen3 — splits the thinking block out of `content`
#   TOOL_PARSER    e.g. qwen3_xml — also sets --enable-auto-tool-choice, which the
#                  parser flag is inert without. Must match what the chat template
#                  emits: Qwen3.5 writes the XML form, so qwen3_xml, not hermes.
#   EXTRA          anything else appended verbatim
set -euo pipefail

# flashinfer JITs on first start and needs a toolkit; find whichever one is installed.
for d in /usr/local/cuda "$(python3 -c 'import os,nvidia;print(os.path.join(os.path.dirname(nvidia.__file__),"cu13"))' 2>/dev/null || true)" \
         "$(python3 -c 'import os,nvidia;print(os.path.join(os.path.dirname(nvidia.__file__),"cuda_nvcc"))' 2>/dev/null || true)"; do
  if [ -n "${d:-}" ] && [ -x "$d/bin/nvcc" ]; then
    export CUDA_HOME="$d" TRITON_PTXAS_PATH="$d/bin/ptxas" PATH="$d/bin:$PATH"
    echo "CUDA_HOME=$CUDA_HOME ($("$d/bin/nvcc" --version | tail -1))"
    break
  fi
done
[ -n "${CUDA_HOME:-}" ] || echo "WARNING: no nvcc found; flashinfer JIT will fail if vLLM needs it"

ARGS=(--model "${MODEL:-/model}" --served-model-name "${SERVED_NAME:-exl3}"
      --host 0.0.0.0 --port "${PORT:-8000}"
      --max-model-len "${MAXLEN:-32768}"
      --gpu-memory-utilization "${GPU_FRAC:-0.90}"
      --max-num-seqs "${MAXSEQS:-64}"
      --enable-prefix-caching --no-enable-log-requests)
[ "${KV_DTYPE:-auto}" != "auto" ] && ARGS+=(--kv-cache-dtype "$KV_DTYPE")
[ -n "${CPU_OFFLOAD_GB:-}" ] && ARGS+=(--cpu-offload-gb "$CPU_OFFLOAD_GB")
[ -n "${KV_OFFLOAD_GB:-}" ] && ARGS+=(--kv-offloading-size "$KV_OFFLOAD_GB" --kv-offloading-backend lmcache)
[ -n "${API_KEY:-}" ] && ARGS+=(--api-key "$API_KEY")
[ -n "${REASONING_PARSER:-}" ] && ARGS+=(--reasoning-parser "$REASONING_PARSER")
[ -n "${TOOL_PARSER:-}" ] && ARGS+=(--enable-auto-tool-choice --tool-call-parser "$TOOL_PARSER")

echo "+ vllm serve ${ARGS[*]} ${EXTRA:-}"
exec python3 -m vllm.entrypoints.openai.api_server "${ARGS[@]}" ${EXTRA:-}
