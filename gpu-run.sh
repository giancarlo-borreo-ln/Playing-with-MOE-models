#!/usr/bin/env bash
#
# gpu-run.sh - run the same GGUF models on the host NVIDIA GPU via Docker.
#
# Uses ghcr.io/ggml-org/llama.cpp:full-cuda with --gpus all. Per-model GPU flags
# (-ngl etc.) live in guest/models.conf and are tuned for a 6 GB RTX 3060.
#
# Usage:
#   ./gpu-run.sh setup            # one-time: install nvidia-container-toolkit + configure docker
#   ./gpu-run.sh doctor           # check docker + GPU access
#   ./gpu-run.sh pull             # pull the CUDA image
#   ./gpu-run.sh download [m...]  # download all (or selected) models to ./models
#   ./gpu-run.sh list             # list models and download status
#   ./gpu-run.sh run <alias> [-- extra llama-cli flags]
#   ./gpu-run.sh serve <alias>    # OpenAI API + web UI on :8080
#   ./gpu-run.sh bench [m...]     # GPU benchmark
#   ./gpu-run.sh stop             # stop the server container
#   ./gpu-run.sh logs             # follow server logs
#
# Env: MODELS_DIR (./models), PORT (8080), IMAGE, CONTAINER (llm-gpu), HF_ENDPOINT
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MANIFEST="$SCRIPT_DIR/guest/models.conf"
MODELS_DIR="${MODELS_DIR:-$SCRIPT_DIR/models}"
IMAGE="${IMAGE:-ghcr.io/ggml-org/llama.cpp:full-cuda}"
CONTAINER="${CONTAINER:-llm-gpu}"
PORT="${PORT:-8080}"
HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"
GPU_ARGS=(--gpus all)

c_cy="\033[1;36m"; c_rd="\033[1;31m"; c_yl="\033[1;33m"; c_rs="\033[0m"
log() { printf "${c_cy}[gpu]${c_rs} %s\n" "$*"; }
warn(){ printf "${c_yl}[gpu]${c_rs} %s\n" "$*" >&2; }
die() { printf "${c_rd}[gpu]${c_rs} %s\n" "$*" >&2; exit 1; }

lookup() { grep -E "^${1}\|" "$MANIFEST" | head -n1; }

need_docker() {
  command -v docker >/dev/null 2>&1 || die "docker is not installed"
  docker info >/dev/null 2>&1 || die "cannot talk to the docker daemon (is it running? are you in the docker group?)"
}

gpu_works() {
  docker run --rm "${GPU_ARGS[@]}" --entrypoint nvidia-smi "$IMAGE" -L >/dev/null 2>&1
}

list_models() {
  local alias company size repo file cpu gpu mark
  printf '%-14s %-10s %-6s %-5s %s\n' ALIAS COMPANY SIZE HAVE FILE
  while IFS='|' read -r alias company size repo file cpu gpu; do
    case "$alias" in ''|'#'*) continue ;; esac
    mark="no"; [ -f "$MODELS_DIR/$file" ] && mark="yes"
    printf '%-14s %-10s %-6s %-5s %s\n' "$alias" "$company" "$size" "$mark" "$file"
  done < "$MANIFEST"
}

cmd_setup() {
  log "installing NVIDIA container toolkit (this modifies the host system)"
  if ! command -v nvidia-ctk >/dev/null 2>&1; then
    if apt-cache show nvidia-container-toolkit >/dev/null 2>&1; then
      sudo apt-get update
      sudo apt-get install -y nvidia-container-toolkit
    else
      curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
        | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
      curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
        | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
        | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null
      sudo apt-get update
      sudo apt-get install -y nvidia-container-toolkit
    fi
  fi
  sudo nvidia-ctk runtime configure --runtime=docker
  sudo systemctl restart docker
  log "done. Verify with: $0 doctor"
}

cmd_doctor() {
  need_docker
  echo "docker:  $(docker --version)"
  command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || warn "no nvidia-smi"
  log "pulling/checking CUDA image (first run downloads a few GB) ..."
  docker pull "$IMAGE" >/dev/null
  if gpu_works; then
    log "GPU access from Docker: OK"
    docker run --rm "${GPU_ARGS[@]}" --entrypoint nvidia-smi "$IMAGE" -L
  else
    warn "GPU access from Docker FAILED."
    warn "Run '$0 setup' (installs nvidia-container-toolkit and configures docker),"
    warn "then log out/in (docker group) and retry."
  fi
}

cmd_pull() { need_docker; docker pull "$IMAGE"; }

cmd_download() {
  mkdir -p "$MODELS_DIR"
  local sel=("$@") alias company size repo file cpu gpu in_sel s
  printf '%-14s %-10s %-6s %s\n' ALIAS COMPANY SIZE FILE
  while IFS='|' read -r alias company size repo file cpu gpu; do
    case "$alias" in ''|'#'*) continue ;; esac
    in_sel=0
    if [ "${#sel[@]}" -eq 0 ]; then
      in_sel=1
    else
      for s in "${sel[@]}"; do
        if [ "$s" = "all" ] || [ "$s" = "$alias" ]; then in_sel=1; break; fi
      done
    fi
    [ "$in_sel" = 1 ] || continue
    if [ -f "$MODELS_DIR/$file" ]; then
      printf '%-14s %-10s %-6s [have] %s\n' "$alias" "$company" "$size" "$file"
      continue
    fi
    printf '%-14s %-10s %-6s [get ] %s\n' "$alias" "$company" "$size" "$file"
    if curl -L --fail -C - --progress-bar \
         -o "$MODELS_DIR/$file.part" \
         "$HF_ENDPOINT/$repo/resolve/main/$file"; then
      mv "$MODELS_DIR/$file.part" "$MODELS_DIR/$file"
    else
      warn "download failed for $alias (partial kept, re-run to resume)"
    fi
  done < "$MANIFEST"
}

require_gpu() {
  need_docker
  gpu_works || die "GPU not usable from Docker. Run '$0 setup', then '$0 doctor'."
}

cmd_run() {
  local alias="${1:-}"; shift || true
  [ -n "$alias" ] || { list_models; return 0; }
  require_gpu
  local line company file cpu gpu
  line="$(lookup "$alias")"
  [ -n "$line" ] || { printf 'Unknown model: %s\n' "$alias" >&2; list_models >&2; return 1; }
  IFS='|' read -r _ company _ _ file cpu gpu <<<"$line"
  [ -f "$MODELS_DIR/$file" ] || die "model not downloaded: ./gpu-run.sh download $alias"
  local args=("$@")
  log "running $alias ($company) on GPU ..."
  exec docker run --rm -it "${GPU_ARGS[@]}" --ulimit memlock=-1:-1 \
    -v "$MODELS_DIR:/models:ro" --entrypoint llama-cli "$IMAGE" \
    -m "/models/$file" $gpu "${args[@]}"
}

cmd_serve() {
  local alias="${1:-}"; shift || true
  [ -n "$alias" ] || die "usage: serve <alias>"
  require_gpu
  local line company file cpu gpu
  line="$(lookup "$alias")"
  [ -n "$line" ] || { printf 'Unknown model: %s\n' "$alias" >&2; list_models >&2; return 1; }
  IFS='|' read -r _ company _ _ file cpu gpu <<<"$line"
  [ -f "$MODELS_DIR/$file" ] || die "model not downloaded: ./gpu-run.sh download $alias"
  docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
  log "starting $alias ($company) on GPU ..."
  docker run -d --name "$CONTAINER" "${GPU_ARGS[@]}" --ulimit memlock=-1:-1 \
    -p "${PORT}:8080" -v "$MODELS_DIR:/models:ro" --entrypoint llama-server "$IMAGE" \
    -m "/models/$file" $gpu -a "$alias" --host 0.0.0.0 --port 8080 "$@" >/dev/null
  log "OpenAI API : http://localhost:${PORT}/v1  (model name: $alias)"
  log "Web UI     : http://localhost:${PORT}"
  log "logs       : ./gpu-run.sh logs       stop: ./gpu-run.sh stop"
}

cmd_bench() {
  require_gpu
  local want=("$@") args=() line file a
  if [ "${#want[@]}" -eq 0 ]; then
    while IFS='|' read -r a _ _ _ _ _ _; do
      case "$a" in ''|'#'*) continue ;; esac
      [ -f "$MODELS_DIR/$(lookup "$a" | cut -d'|' -f5)" ] && want+=("$a")
    done < "$MANIFEST"
  fi
  for a in "${want[@]}"; do
    line="$(lookup "$a")" || continue
    [ -n "$line" ] || continue
    file="$(cut -d'|' -f5 <<<"$line")"
    [ -f "$MODELS_DIR/$file" ] || continue
    args+=(-m "/models/$file" $(cut -d'|' -f7 <<<"$line"))
  done
  [ "${#args[@]}" -gt 0 ] || die "no downloaded models to benchmark"
  docker run --rm -it "${GPU_ARGS[@]}" --ulimit memlock=-1:-1 \
    -v "$MODELS_DIR:/models:ro" --entrypoint llama-bench "$IMAGE" "${args[@]}"
}

cmd_stop()  { need_docker; docker rm -f "$CONTAINER" >/dev/null 2>&1 && log "stopped $CONTAINER" || log "not running"; }
cmd_logs()  { need_docker; docker logs -f "$CONTAINER"; }
cmd_help()  { awk 'NR>1 && /^#/ {sub(/^# ?/,""); print; next} NR>1 {exit}' "$0"; }

cmd="${1:-list}"
[ "$#" -gt 0 ] && shift || true
case "$cmd" in
  setup)     cmd_setup ;;
  doctor)    cmd_doctor ;;
  pull)      cmd_pull ;;
  download)  cmd_download "$@" ;;
  list|"")   list_models ;;
  run)       cmd_run "$@" ;;
  serve)     cmd_serve "$@" ;;
  bench)     cmd_bench "$@" ;;
  stop)      cmd_stop ;;
  logs)      cmd_logs ;;
  help|-h|--help) cmd_help ;;
  *) die "unknown command: $cmd (try: $0 help)" ;;
esac
