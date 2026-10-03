#!/usr/bin/env bash
#
# First-boot guest provisioning for the LLM VM. Runs as root via
# vm-setup.service. Idempotent: safe to re-run.
#
set -uo pipefail
export HOME=/root
export DEBIAN_FRONTEND=noninteractive

mkdir -p /var/lib/vm-setup /opt/llm /opt/models
LOG=/var/log/vm-setup.log
exec > >(tee -a "$LOG") 2>&1

log() { printf '\n\033[1;36m[setup %(%H:%M:%S)T]\033[0m %s\n' -1 "$*"; }
die() { printf '\033[1;31m[setup]\033[0m %s\n' "$*"; exit 1; }

log "starting guest provisioning"

# ---------------------------------------------------------------- packages
log "installing build tools"
apt-get update -y
apt-get install -y --no-install-recommends \
  build-essential cmake git ca-certificates curl wget \
  python3 python3-venv python3-pip \
  libgomp1 htop tmux less jq procps pciutils \
  || die "apt install failed"

# ---------------------------------------------------------------- swap
# 10 GB RAM is tight for 14B models; swap keeps the OOM killer away.
if [ ! -f /swapfile ]; then
  log "creating 8G swapfile"
  fallocate -l 8G /swapfile 2>/dev/null || dd if=/dev/zero of=/swapfile bs=1M count=8192 status=none
  chmod 600 /swapfile
  mkswap /swapfile >/dev/null
  grep -q '^/swapfile ' /etc/fstab || echo '/swapfile none swap sw 0 0' >> /etc/fstab
fi
swapon /swapfile 2>/dev/null || true

# ---------------------------------------------------------------- llama.cpp
# Built with -DGGML_NATIVE=ON so the guest uses the host CPU's full instruction
# set (AVX2/AVX-512/FMA) for the quantised matmuls.
if [ ! -x /usr/local/bin/llama-cli ]; then
  log "cloning + compiling llama.cpp (native CPU, this takes a few minutes)"
  rm -rf /opt/llama.cpp
  git clone --depth 1 https://github.com/ggml-org/llama.cpp /opt/llama.cpp || die "git clone failed"
  cmake -S /opt/llama.cpp -B /opt/llama.cpp/build \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_NATIVE=ON \
    -DGGML_OPENMP=ON \
    -DLLAMA_CURL=OFF \
    -DLLAMA_BUILD_TESTS=OFF \
    -DLLAMA_BUILD_EXAMPLES=OFF \
    -DLLAMA_BUILD_SERVER=ON \
    || die "cmake configure failed"
  cmake --build /opt/llama.cpp/build --config Release -j"$(nproc)" \
    --target llama-cli llama-server llama-bench \
    || die "llama.cpp build failed"
  for b in llama-cli llama-server llama-bench; do
    ln -sf "/opt/llama.cpp/build/bin/$b" "/usr/local/bin/$b"
  done
else
  log "llama.cpp already built"
fi

# ---------------------------------------------------------------- hf downloader
if [ ! -x /opt/llm/venv/bin/python ]; then
  log "creating python venv with huggingface_hub"
  python3 -m venv /opt/llm/venv
  /opt/llm/venv/bin/pip install -q --upgrade pip
  /opt/llm/venv/bin/pip install -q huggingface_hub || die "pip install failed"
fi

# ---------------------------------------------------------------- scripts
chmod 0755 /opt/llm/*.sh /opt/llm/hf_download.py \
  /usr/local/bin/run /usr/local/bin/serve /usr/local/bin/models /usr/local/bin/download-models 2>/dev/null || true

# ---------------------------------------------------------------- models
log "downloading models (this is the long part)"
/opt/llm/download-models.sh
dl_status=$?

# ---------------------------------------------------------------- done
mkdir -p /var/lib/vm-setup
if [ "$dl_status" -ne 0 ]; then
  log "WARNING: some models failed to download. Re-run 'download-models' inside the VM."
fi
: > /var/lib/vm-setup/ready
sync
log "provisioning complete - run 'models' to list models, then 'run <alias>'"
exit 0
