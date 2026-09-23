#!/usr/bin/env bash
# Serve an EXL3 pack with every knob that matters on a memory-constrained card exposed.
#
# The usual vLLM throughput advice (max-num-batched-tokens 16k-64k, max-num-seqs 256-512)
# assumes KV is plentiful. Here it is not: on 16 GB this model leaves ~2.4 GiB for KV, so
# CUDA-graph capture, block tables and activation peaks all compete with the context length
# the user actually gets. On this class of card the same knobs move the other way.
#
#   PRESET=16gb bash presets/serve.sh /path/to/checkpoint 8000 orca
#
# This is the only place launch arguments are built. The Docker entrypoint sets a few
# container defaults and then execs this same script, so an option cannot be present on one
# path and missing on the other.
set -u
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$HERE/.." && pwd)

# PRESET takes a name in presets/ ("16gb") or a path to any env file.
#
# A preset supplies DEFAULTS, not overrides: anything already in the environment wins. Sourcing
# the file outright (set -a; . file) would mean `PRESET=16gb MAXLEN=8192 serve.sh ...` and
# `docker run -e PRESET=16gb -e MAXLEN=8192` silently ignore the second variable -- and worse,
# that a preset's BIND would override the container default of 0.0.0.0 and leave the server
# listening on loopback, unreachable from its own published port.
load_preset() {
  local line key
  while IFS= read -r line || [ -n "$line" ]; do
    line=${line#"${line%%[![:space:]]*}"}                 # strip leading blanks
    case "$line" in ''|'#'*) continue;; esac
    line=${line#export }
    case "$line" in *=*) key=${line%%=*};; *) continue;; esac
    case "$key" in ''|*[!A-Za-z0-9_]*) continue;; esac
    [ -n "$(eval "printf %s \"\${$key+set}\"")" ] && continue   # already set: caller wins
    eval "export $line"
  done < "$1"
}
if [ -n "${PRESET:-}" ]; then
  for f in "$PRESET" "$HERE/$PRESET.env" "$HERE/$PRESET"; do
    [ -f "$f" ] && { load_preset "$f"; echo "preset: $f"; PRESET_OK=1; break; }
  done
  [ -n "${PRESET_OK:-}" ] || { echo "no such preset: $PRESET (have: $(ls "$HERE"/*.env 2>/dev/null | xargs -n1 basename 2>/dev/null | tr '\n' ' '))" >&2; exit 2; }
fi

MODEL=${1:-${MODEL:-}}; PORT=${2:-${PORT:-8000}}; NAME=${3:-${SERVED_NAME:-orca}}
[ -n "$MODEL" ] || { echo "usage: [PRESET=16gb] $0 <checkpoint-dir> [port] [served-name]" >&2; exit 2; }

# Interpreter: an explicit PYTHON wins, then a virtualenv beside the repo, then PATH. Nothing
# here may assume a venv exists -- pip install -e . into an existing vLLM environment is the
# normal case, and hardcoding a venv path made this script work on exactly one machine.
if [ -z "${PYTHON:-}" ]; then
  for c in "$ROOT/.venv-vllm/bin/python" "$ROOT/.venv/bin/python"; do
    [ -x "$c" ] && { PYTHON=$c; break; }
  done
  PYTHON=${PYTHON:-$(command -v python3 || true)}
fi
[ -n "${PYTHON:-}" ] && [ -x "$PYTHON" ] || { echo "no python found; set PYTHON=/path/to/python" >&2; exit 2; }
# find_spec rather than import: this answers "is vLLM installed here" in milliseconds and
# turns the otherwise cryptic failure that follows into one line.
"$PYTHON" -c 'import importlib.util as u,sys; sys.exit(0 if u.find_spec("vllm") else 1)' 2>/dev/null || {
  echo "ERROR: $PYTHON has no vllm. Install vLLM there, then 'pip install -e .' for the plugin," >&2
  echo "       or point PYTHON= at the interpreter that does." >&2; exit 2; }
case "$PYTHON" in */bin/python*) export PATH="$(dirname "$PYTHON"):$PATH";; esac

# flashinfer JIT-compiles kernels on first start and needs an nvcc. Probe for one instead of
# naming a path: which CUDA toolkit is present depends on how vLLM was installed, and an
# extension built against one CUDA major will not load against the other, so torch's own CUDA
# version decides the order. `nvidia` is a PEP 420 namespace package -- its __file__ is None
# and only __path__ is usable, which is what makes the obvious one-liner here fail silently.
if [ -z "${CUDA_HOME:-}" ]; then
  CUDA_HOME=$("$PYTHON" - <<'PROBE' 2>/dev/null || true
import os
try:
    import torch
    major = (torch.version.cuda or "").split(".")[0]
except Exception:
    major = ""
order = ["cu12", "cu13"] if major == "12" else ["cu13", "cu12"]
cands = []
try:
    import nvidia
    for base in list(nvidia.__path__):
        cands += [os.path.join(base, d) for d in order + ["cuda_nvcc"]]
except Exception:
    pass
cands += ["/usr/local/cuda"]
for c in cands:
    if os.path.exists(os.path.join(c, "bin", "nvcc")):
        print(c)
        break
PROBE
)
  [ -n "${CUDA_HOME:-}" ] && export CUDA_HOME
fi
if [ -n "${CUDA_HOME:-}" ]; then
  export TRITON_PTXAS_PATH="$CUDA_HOME/bin/ptxas" PATH="$CUDA_HOME/bin:$PATH"
else
  echo "note: no nvcc found; fine unless vLLM needs a flashinfer JIT build" >&2
fi
# Host-compiler pin for that JIT: nvcc refuses host compilers newer than it supports, and on
# a box whose default is gcc-12 that surfaces as a compile error deep in the build. Only used
# when gcc-11 is actually installed, and never overrides a CC the caller set.
if [ -z "${CC:-}" ] && [ -x /usr/bin/gcc-11 ] && [ -x /usr/bin/g++-11 ]; then
  export CC=/usr/bin/gcc-11 CXX=/usr/bin/g++-11 NVCC_PREPEND_FLAGS="-ccbin /usr/bin/g++-11"
fi
# flashinfer's sampling kernels are the most fragile of its JIT targets and vLLM's torch
# sampler is equivalent.
export VLLM_USE_FLASHINFER_SAMPLER=${VLLM_USE_FLASHINFER_SAMPLER:-0}

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
# on a card this size. The native backend ships with vLLM; lmcache is a separate install.
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
echo "+ $PYTHON -m vllm.entrypoints.openai.api_server ${ARGS[*]} ${EXTRA:-}"
[ -n "${DRYRUN:-}" ] && exit 0
# EXTRA is deliberately unquoted: it carries whole flags, including --speculative-config's
# JSON, which contains no spaces.
exec "$PYTHON" -m vllm.entrypoints.openai.api_server "${ARGS[@]}" ${EXTRA:-}
