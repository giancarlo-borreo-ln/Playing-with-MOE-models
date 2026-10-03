#!/usr/bin/env bash
#
# In-VM LLM runner (CPU, llama.cpp built with -march=native).
# Per-model flags come from the shared manifest /opt/llm/models.conf.
#
#   run                     list models
#   run <alias>             interactive chat
#   run <alias> -p "hi"     one-shot (any llama-cli flag passes through)
#   run doctor|bench|...    shortcuts to the commands below
#   serve <alias>           OpenAI-compatible HTTP API on :8080
#   models                  list models + download status
#   download-models [a...]  fetch all (or selected) models
#   bench [a...]            CPU benchmark for downloaded models
#   doctor                  runtime / CPU / memory info
#
set -uo pipefail

CONF="${LLM_CONF:-/opt/llm/models.conf}"
MODEL_DIR="${LLM_MODEL_DIR:-/opt/models}"
LLAMA_CLI="${LLAMA_CLI:-llama-cli}"
LLAMA_SERVER="${LLAMA_SERVER:-llama-server}"
LLAMA_BENCH="${LLAMA_BENCH:-llama-bench}"

die() { printf '\033[1;31m%s\033[0m\n' "$*" >&2; exit 1; }

lookup() { grep -E "^${1}\|" "$CONF" | head -n1; }

list_models() {
  local alias company size repo file cpu gpu mark
  printf '%-14s %-10s %-6s %-5s %s\n' ALIAS COMPANY SIZE HAVE FILE
  while IFS='|' read -r alias company size repo file cpu gpu; do
    case "$alias" in ''|'#'*) continue ;; esac
    mark="no"; [ -f "$MODEL_DIR/$file" ] && mark="yes"
    printf '%-14s %-10s %-6s %-5s %s\n' "$alias" "$company" "$size" "$mark" "$file"
  done < "$CONF"
}

cmd_run() {
  local alias="${1:-}"; shift || true
  case "$alias" in
    "")            list_models; return 0 ;;
    doctor)        cmd_doctor; return 0 ;;
    bench)         cmd_bench "$@"; return $? ;;
    list|models)   list_models; return 0 ;;
    download|pull) cmd_download "$@"; return $? ;;
    help|-h|--help) cmd_help; return 0 ;;
  esac

  local line company file cpu gpu path
  line="$(lookup "$alias")"
  [ -n "$line" ] || { printf 'Unknown model: %s\n' "$alias" >&2; list_models >&2; return 1; }
  IFS='|' read -r _ company _ _ file cpu gpu <<<"$line"
  path="$MODEL_DIR/$file"
  [ -f "$path" ] || { printf 'Model %s is not downloaded. Run: download-models %s\n' "$alias" "$alias" >&2; return 1; }

  if [ "$#" -eq 0 ]; then
    printf '\033[1;36m%s\033[0m (%s, CPU) - interactive chat. Ctrl+D to quit.\n\n' "$alias" "$company"
    exec "$LLAMA_CLI" -m "$path" $cpu
  else
    exec "$LLAMA_CLI" -m "$path" $cpu "$@"
  fi
}

cmd_serve() {
  local alias="${1:-}"; shift || true
  [ -n "$alias" ] || die "usage: serve <alias>"
  local line company file cpu gpu path
  line="$(lookup "$alias")"
  [ -n "$line" ] || { printf 'Unknown model: %s\n' "$alias" >&2; return 1; }
  IFS='|' read -r _ company _ _ file cpu gpu <<<"$line"
  path="$MODEL_DIR/$file"
  [ -f "$path" ] || { printf 'Model %s is not downloaded. Run: download-models %s\n' "$alias" "$alias" >&2; return 1; }
  printf '\033[1;36mserving %s (%s, CPU)\033[0m\n' "$alias" "$company"
  printf 'OpenAI API : http://localhost:8080/v1  (model name: %s)\n' "$alias"
  printf 'Web UI     : http://localhost:8080\n\n'
  exec "$LLAMA_SERVER" -m "$path" $cpu -a "$alias" --host 0.0.0.0 --port 8080 "$@"
}

cmd_download() { exec /opt/llm/download-models.sh "$@"; }

cmd_bench() {
  local want=("$@") path args=() a line file
  if [ "${#want[@]}" -eq 0 ]; then
    while IFS='|' read -r a _ _ _ _ _ _; do
      case "$a" in ''|'#'*) continue ;; esac
      want+=("$a")
    done < "$CONF"
  fi
  for a in "${want[@]}"; do
    line="$(lookup "$a")" || continue
    [ -n "$line" ] || continue
    IFS='|' read -r _ _ _ _ file _ _ <<<"$line"
    [ -f "$MODEL_DIR/$file" ] || continue
    args+=(-m "$MODEL_DIR/$file")
  done
  [ "${#args[@]}" -gt 0 ] || die "no downloaded models to benchmark"
  exec "$LLAMA_BENCH" "${args[@]}"
}

cmd_doctor() {
  echo "=== llama.cpp ==="
  "$LLAMA_CLI" --version 2>&1 | head -n 3
  echo
  echo "=== CPU ==="
  grep -m1 'model name' /proc/cpuinfo
  echo "vcpus: $(nproc)"
  grep -m1 '^flags' /proc/cpuinfo | grep -oE 'avx512[a-z_]*|avx2|avx|fma|f16c' | sort -u | tr '\n' ' '
  echo
  echo "=== memory ==="
  free -h
  echo
  echo "=== models (${MODEL_DIR}) ==="
  du -sh "$MODEL_DIR" 2>/dev/null || true
}

cmd_help() {
  awk 'NR>1 && /^#/ {sub(/^# ?/,""); print; next} NR>1 {exit}' "$0"
}

case "${1:-list}" in
  list|"")        list_models ;;
  run)            shift; cmd_run "$@" ;;
  serve)          shift; cmd_serve "$@" ;;
  download|pull)  shift; cmd_download "$@" ;;
  bench)          shift; cmd_bench "$@" ;;
  doctor)         shift; cmd_doctor ;;
  help|-h|--help) cmd_help ;;
  *)              cmd_run "$@" ;;
esac
