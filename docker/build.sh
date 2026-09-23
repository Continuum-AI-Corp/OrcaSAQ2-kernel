#!/usr/bin/env bash
# Build the serving image. Run from the repo root: bash docker/build.sh [tag]
set -euo pipefail
cd "$(dirname "$0")/.."
TAG=${1:-orca-exl3-vllm:0.30.0}
# the daemon socket is root-owned on hosts where the user is not in the docker group
DOCKER=docker; docker info >/dev/null 2>&1 || DOCKER="sudo docker"
$DOCKER build -f docker/Dockerfile -t "$TAG" --build-arg VLLM_TAG="${VLLM_TAG:-v0.30.0}" .
echo "built $TAG"
echo
echo "80 GB card:   docker run --gpus all -v /data/models/<pack>:/model:ro -p 8000:8000 $TAG"
echo "16 GB card:   docker run --gpus all -v /data/models/<pack>:/model:ro -p 8000:8000 \\"
echo "                -e MAXSEQS=1 -e GPU_FRAC=0.95 -e MAXLEN=32768 -e KV_DTYPE=fp8_e4m3 $TAG"
echo "16 GB, 128k:  add -e CPU_OFFLOAD_GB=4   (slow: ~+160 ms/token, but it runs)"
