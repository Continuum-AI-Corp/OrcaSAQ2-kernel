#!/usr/bin/env bash
# Build the serving image. Run from the repo root: bash docker/build.sh [tag]
set -euo pipefail
cd "$(dirname "$0")/.."
TAG=${1:-orcasaq2-vllm:0.30.0}
# the daemon socket is root-owned on hosts where the user is not in the docker group
DOCKER=docker; docker info >/dev/null 2>&1 || DOCKER="sudo docker"
$DOCKER build -f docker/Dockerfile -t "$TAG" --build-arg VLLM_TAG="${VLLM_TAG:-v0.30.0}" .
echo "built $TAG"
echo
echo "Presets carry the measured values; prefer them over naming knobs by hand."
echo
echo "  80 GB card:  docker run --gpus all -v /data/models/<pack>:/model:ro -p 8000:8000 \\"
echo "                 -e PRESET=80gb $TAG"
echo "  16 GB card:  ... -e PRESET=16gb $TAG"
echo "  16 GB + MTP: ... -e PRESET=16gb-mtp $TAG      (1.72x faster, -40% context)"
echo
echo "Add -e API_KEY=... to require a bearer token. Note vLLM guards only /v1 /v2 /inference"
echo "/cohere with it -- /health and /metrics stay open, so publish the port accordingly."
echo "16 GB, 128k:  add -e CPU_OFFLOAD_GB=4   (slow: ~+160 ms/token, but it runs)"
