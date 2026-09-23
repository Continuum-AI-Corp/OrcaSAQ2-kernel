#!/usr/bin/env bash
# Serve an EXL3 pack with every knob that matters on a memory-constrained card exposed.
#
# The usual vLLM throughput advice (max-num-batched-tokens 16k-64k, max-num-seqs 256-512)
# assumes KV is plentiful. Here it is not: on 16 GB this model leaves ~2.4 GiB for KV, so
# CUDA-graph capture, block tables and activation peaks all compete with the context length
# the user actually gets. On this class of card the same knobs move the other way.
set -u
cd "$(dirname "$0")/.."
CU=$PWD/.venv-vllm/lib/python3.12/site-packages/nvidia/cu13
export CUDA_HOME=$CU TRITON_PTXAS_PATH=$CU/bin/ptxas PATH=$PWD/.venv-vllm/bin:$CU/bin:$PATH
export CC=/usr/bin/gcc-11 CXX=/usr/bin/g++-11 NVCC_PREPEND_FLAGS="-ccbin /usr/bin/g++-11"
export VLLM_USE_FLASHINFER_SAMPLER=0
MODEL=${1:?model dir}; PORT=${2:-8000}; NAME=${3:-orca}

# BIND defaults to loopback on purpose. vLLM's own default is 0.0.0.0, which on a box whose
# eth0 carries a public IP and whose iptables policy is ACCEPT means an unauthenticated
# inference endpoint on the open internet the moment the server starts. Exposing it has to be
# a deliberate act, and the firewall has to be in place first.
ARGS=(--model "$MODEL" --served-model-name "$NAME" --port "$PORT" --host "${BIND:-127.0.0.1}"
      --max-model-len "${MAXLEN:-16384}"
      --gpu-memory-utilization "${GPU_FRAC:-0.95}"
      --max-num-seqs "${MAXSEQS:-8}"
      --no-enable-log-requests)
[ "${PREFIX_CACHE:-1}" = "1" ] && ARGS+=(--enable-prefix-caching)
[ -n "${KV_DTYPE:-}" ] && ARGS+=(--kv-cache-dtype "$KV_DTYPE")
[ -n "${MAXBATCHTOK:-}" ] && ARGS+=(--max-num-batched-tokens "$MAXBATCHTOK")
[ -n "${OLEVEL:-}" ] && ARGS+=(-O"$OLEVEL")
[ -n "${CGSIZES:-}" ] && ARGS+=(--cuda-graph-sizes "$CGSIZES")
[ "${EAGER:-0}" = "1" ] && ARGS+=(--enforce-eager)
[ -n "${SWAP:-}" ] && ARGS+=(--swap-space "$SWAP")
# KV offload: a second and third tier under the GPU pool. It does NOT shrink one sequence's
# active attention window -- that still has to fit in VRAM -- but it lets evicted blocks and
# reusable prefixes live in host RAM instead of being recomputed, which is what actually hurts
# on a card this size.
[ -n "${KV_OFFLOAD_GB:-}" ] && ARGS+=(--kv-offloading-size "$KV_OFFLOAD_GB"
                                      --kv-offloading-backend "${KV_OFFLOAD_BACKEND:-native}")
[ -n "${CPU_OFFLOAD_GB:-}" ] && ARGS+=(--cpu-offload-gb "$CPU_OFFLOAD_GB")
[ -n "${MAMBA_BLOCK:-}" ] && ARGS+=(--mamba-block-size "$MAMBA_BLOCK")
[ -n "${BLOCK_SIZE:-}" ] && ARGS+=(--block-size "$BLOCK_SIZE")
[ -n "${API_KEY:-}" ] && ARGS+=(--api-key "$API_KEY")
[ -n "${REASONING_PARSER:-}" ] && ARGS+=(--reasoning-parser "$REASONING_PARSER")
# Tool calling has to be switched on explicitly: --tool-call-parser alone is inert without
# --enable-auto-tool-choice, so they are one knob here rather than two. The parser must match
# what the chat template emits -- this model's template writes the Qwen3 XML form
# (<tool_call><function=..><parameter=..>), which is qwen3_xml, not the JSON-in-tag hermes form.
[ -n "${TOOL_PARSER:-}" ] && ARGS+=(--enable-auto-tool-choice --tool-call-parser "$TOOL_PARSER")
echo "+ vllm ${ARGS[*]} ${EXTRA:-}"
exec ./.venv-vllm/bin/python -m vllm.entrypoints.openai.api_server "${ARGS[@]}" ${EXTRA:-}
