# vLLM serving image for EXL3 (exllamav3 trellis) quantized checkpoints.
#
# Everything here exists because of something that broke once:
#   * the exllamav3 CUDA extension is ABI-linked to torch AND to a CUDA major version, so the
#     wheel is resolved from the base image's own torch at build time rather than pinned
#   * flashinfer JIT-compiles kernels on first start and needs an nvcc matching torch's CUDA
#     minor version. The upstream image ships its own CUDA 13.0 toolkit AND a pip
#     nvidia-cuda-nvcc 13.4.92 that can shadow it on PATH; the 13.4 front end emits PTX 9.4
#     which the 13.0 ptxas rejects, so the pip one is pinned down to torch's version here.
#   * its sampling kernels are JIT-built too and are the most fragile of them; vLLM's torch
#     sampler is equivalent, so it is disabled by default
ARG VLLM_TAG=v0.30.0
FROM vllm/vllm-openai:${VLLM_TAG}

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends g++ ninja-build curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Resolve and install the exllamav3 wheel and nvcc that match this image's torch.
ARG EXL3_VERSION=1.5.1
RUN python3 - <<'PY'
import json, os, re, subprocess, sys, urllib.request
import torch
tv = torch.__version__.split("+")[0]
minor = ".".join(tv.split(".")[:2]) + ".0"          # wheels are tagged torch2.13.0, not 2.13.1
cuda = torch.version.cuda or ""
major = cuda.split(".")[0]
# exllamav3 publishes cu128.* for CUDA 12.x and cu132.* for CUDA 13.x; an extension built
# against one CUDA major will not load against the other.
cu = "cu128" if major == "12" else "cu132"
ver = os.environ.get("EXL3_VERSION", "1.5.1")
name = f"exllamav3-{ver}+{cu}.torch{minor}-cp312-cp312-linux_x86_64.whl"
url = f"https://github.com/turboderp-org/exllamav3/releases/download/v{ver}/" + name.replace("+", "%2B")
print(f"torch {tv} cuda {cuda} -> {name}", flush=True)
# pip reads the package metadata out of the filename, so keep the real one
path = "/tmp/" + name
urllib.request.urlretrieve(url, path)
subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-cache-dir", path])
os.remove(path)
# flashinfer's JIT needs nvcc at torch's CUDA minor version, not the newest available
pkg = "nvidia-cuda-nvcc-cu12" if major == "12" else "nvidia-cuda-nvcc"
spec = f"{pkg}=={cuda}.*"
try:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-cache-dir", spec])
except subprocess.CalledProcessError:
    print(f"WARNING: no {spec}; leaving the image's own toolkit in place", flush=True)
PY

# The plugin, installed so vLLM discovers it through the vllm.general_plugins entry point.
# pyproject declares readme = "README.md", so that file has to come along or the build fails.
COPY pyproject.toml README.md /opt/orcasaq2/
COPY orcasaq2 /opt/orcasaq2/orcasaq2
RUN python3 -m pip install --no-cache-dir /opt/orcasaq2 && rm -rf /opt/orcasaq2/*.egg-info

# The presets and the launcher, so the container and the host build arguments from the same
# script rather than from two copies that drift.
COPY presets /opt/orcasaq2/presets

# CUDA_HOME is set at runtime by serve.sh, which locates whichever toolkit is present.
ENV VLLM_USE_FLASHINFER_SAMPLER=0
COPY docker/entrypoint.sh /usr/local/bin/serve-exl3
RUN chmod +x /usr/local/bin/serve-exl3 /opt/orcasaq2/presets/serve.sh

EXPOSE 8000
ENTRYPOINT ["/usr/local/bin/serve-exl3"]
