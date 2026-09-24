#!/usr/bin/env bash
# Container entrypoint. It sets the handful of defaults that differ inside a container and
# then execs presets/serve.sh, which is the single place launch arguments are built. Keeping
# a second argument list here is how the image ended up without --block-size (a measured +41%
# KV) and with an --kv-offloading-backend the image does not ship.
#
#   docker run --gpus all -v /path/to/checkpoint:/model:ro -p 8000:8000 \
#     -e PRESET=16gb -e API_KEY=... <image>
#
# Every knob presets/serve.sh documents works as -e here. The ones worth knowing:
#
#   PRESET         16gb | 16gb-mtp | 80gb — measured configs; start here
#   MODEL          checkpoint path inside the container (default /model)
#   SERVED_NAME    model id clients use (default exl3)
#   PORT           listen port (default 8000)
#   API_KEY        bearer token. vLLM only guards /v1 /v2 /inference /cohere with it --
#                  /health and /metrics stay open, so the key is not the boundary
#   MAXLEN GPU_FRAC MAXSEQS MAXBATCHTOK BLOCK_SIZE KV_DTYPE PREFIX_CACHE
#   KV_OFFLOAD_GB  GPU/CPU KV tiering. KV_OFFLOAD_BACKEND defaults to native, which ships with
#                  vLLM; lmcache is a separate install and is not in this image
#   CPU_OFFLOAD_GB weights streamed from host RAM. Last resort: roughly +40 ms/token per GB
#                  at PCIe 4.0 x16
#   REASONING_PARSER / TOOL_PARSER   see presets/serve.sh
#   EXTRA          anything else appended verbatim
set -u

# Inside a container this is the container's own interface, not the host's. The published
# port (-p) and the host firewall are the boundary here, so binding wide is correct -- unlike
# on a host, where serve.sh defaults to loopback for good reason.
export BIND=${BIND:-0.0.0.0}
export MODEL=${MODEL:-/model} SERVED_NAME=${SERVED_NAME:-exl3} PORT=${PORT:-8000}

[ -d "$MODEL" ] || { echo "ERROR: no checkpoint at $MODEL -- mount one with -v /host/pack:/model:ro" >&2; exit 2; }

exec /opt/orcasaq2/presets/serve.sh "$MODEL" "$PORT" "$SERVED_NAME"
