#!/usr/bin/env bash
#
# track-a.sh - Track A: make a 32B-class model usable on this 6 GB laptop.
#
# Benchmarks local models honestly (tokens/s), frees the VM's RAM, pulls a
# sparse-MoE model, and saves a comparison report. No model weights are
# modified; everything runs through the local ollama server (CUDA).
#
# Usage:
#   ./track-a.sh                 # show this help
#   ./track-a.sh doctor          # host, GPU, RAM, disk, ollama + VM status
#   ./track-a.sh free            # stop the VM and show reclaimed RAM
#   ./track-a.sh models          # list installed models
#   ./track-a.sh pull moe        # pull the recommended sparse-MoE model
#   ./track-a.sh bench <name>    # timed generation, prints tokens/s
#   ./track-a.sh dense           # bench all installed dense models
#   ./track-a.sh moe             # bench the installed MoE model(s)
#   ./track-a.sh compare         # dense vs MoE summary from saved results
#   ./track-a.sh report          # show the saved results table
#   ./track-a.sh spec            # run llama.cpp speculative decoding (opt-in)
#   ./track-a.sh everything      # free VM -> doctor -> bench all -> report
#   ./track-a.sh reset-results   # delete the saved results file
#
# Env:
#   OLLAMA_HOST   (default http://127.0.0.1:11434)
#   NUM_PREDICT   (default 128)   tokens to generate per benchmark
#   CTX           (default 4096)  context size for the run
#   API_TIMEOUT   (default 1800)  seconds per benchmark request
#   MOE_MODEL     (default qwen3:30b-a3b)
#   WARMUP        (default 1)     1 = load+warm the model before timing
#
# Track A rationale: a dense 32B must read ~18.5 GB of weights per token, so it
# is capped well under 1 tok/s here. A sparse MoE (~3B active params/token)
# reads ~2 GB/token and is 10-40x faster at the same nominal size. This script
# lets you measure that difference on this exact machine.
#
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OLLAMA_HOST="${OLLAMA_HOST:-http://127.0.0.1:11434}"
NUM_PREDICT="${NUM_PREDICT:-128}"
CTX="${CTX:-4096}"
API_TIMEOUT="${API_TIMEOUT:-1800}"
MOE_MODEL="${MOE_MODEL:-qwen3:30b-a3b}"
RESULTS="${RESULTS:-$SCRIPT_DIR/track-a-results.tsv}"
PROMPT="${PROMPT:-Explain in detail how a transformer neural network processes text, covering tokens, embeddings and attention.}"

c_cy="\033[1;36m"; c_rd="\033[1;31m"; c_yl="\033[1;33m"; c_gr="\033[1;32m"; c_rs="\033[0m"
log()  { printf "${c_cy}[A]${c_rs} %s\n" "$*"; }
ok()   { printf "${c_gr}[A]${c_rs} %s\n" "$*"; }
warn() { printf "${c_yl}[A]${c_rs} %s\n" "$*" >&2; }
die()  { printf "${c_rd}[A]${c_rs} %s\n" "$*" >&2; exit 1; }

ollama_up() { curl -sf --max-time 5 "$OLLAMA_HOST/api/version" >/dev/null 2>&1; }

need_ollama() {
  command -v ollama >/dev/null 2>&1 || die "ollama is not installed"
  ollama_up || die "ollama API not reachable at $OLLAMA_HOST (is the service running?)"
}

installed_models() { ollama list 2>/dev/null | awk 'NR>1 {print $1}'; }

have_model() { installed_models | grep -qx "$1"; }

is_moe() { printf '%s' "$1" | grep -qiE 'a3b|a[0-9]+b|moe|mixtral'; }

# --------------------------------------------------------------- benchmark core
# Calls /api/generate (non-streaming) and reports steady-state generation rate.
gen_body() {
  jq -nc --arg m "$1" --arg p "$PROMPT" --argjson np "$2" --argjson ctx "$CTX" \
    '{model:$m,prompt:$p,stream:false,
      options:{temperature:0,num_predict:$np,num_ctx:$ctx}}'
}

bench_one() {
  local model="$1"
  local json err
  if [ "${WARMUP:-1}" = 1 ]; then
    log "warming up $model ..."
    curl -s --max-time "$API_TIMEOUT" -H 'Content-Type: application/json' \
      -d "$(gen_body "$model" 1)" "$OLLAMA_HOST/api/generate" >/dev/null || true
  fi
  json=$(curl -s --max-time "$API_TIMEOUT" -H 'Content-Type: application/json' \
    -d "$(gen_body "$model" "$NUM_PREDICT")" \
    "$OLLAMA_HOST/api/generate") || { warn "request failed: $model"; return 1; }

  err=$(jq -r '.error // empty' <<<"$json" 2>/dev/null)
  [ -n "$err" ] && { warn "$model: $err"; return 1; }

  local ec ed pc pd ld td rate prate lds tds gpu_pct vram size
  ec=$(jq   -r '.eval_count        // 0' <<<"$json")
  ed=$(jq   -r '.eval_duration     // 0' <<<"$json")
  pc=$(jq   -r '.prompt_eval_count // 0' <<<"$json")
  pd=$(jq   -r '.prompt_eval_duration // 0' <<<"$json")
  ld=$(jq   -r '.load_duration     // 0' <<<"$json")
  td=$(jq   -r '.total_duration    // 0' <<<"$json")

  rate=$(awk  -v c="$ec" -v d="$ed" 'BEGIN{printf (d>0 && c>0) ? "%.2f" : "0.00", c/(d/1e9)}')
  prate=$(awk -v c="$pc" -v d="$pd" 'BEGIN{printf (d>0 && c>0) ? "%.2f" : "0.00", c/(d/1e9)}')
  lds=$(awk  -v d="$ld" 'BEGIN{printf "%.1f", d/1e9}')
  tds=$(awk  -v d="$td" 'BEGIN{printf "%.1f", d/1e9}')

  # How much of the model sits in VRAM (from /api/ps).
  vram=$(curl -s --max-time 5 "$OLLAMA_HOST/api/ps" \
    | jq -r --arg m "$model" '.models[]? | select(.name==$m or (.name|startswith($m))) | .size_vram // 0' | head -n1)
  size=$(curl -s --max-time 5 "$OLLAMA_HOST/api/ps" \
    | jq -r --arg m "$model" '.models[]? | select(.name==$m or (.name|startswith($m))) | .size // 0' | head -n1)
  vram=${vram:-0}; size=${size:-0}
  if [ "$size" -gt 0 ] 2>/dev/null; then
    gpu_pct=$(awk -v v="$vram" -v s="$size" 'BEGIN{printf "%d%% VRAM", (100*v)/s}')
  else
    gpu_pct="n/a"
  fi

  printf '%-34s %8s tok/s  (%s gen, %s prompt tok/s, %s load, %s total, %s)\n' \
    "$model" "$rate" "$ec" "$prate" "${lds}s" "${tds}s" "$gpu_pct"

  [ -f "$RESULTS" ] || printf 'timestamp\tmodel\tgen_toks\tgen_tok_s\tprompt_toks\tprompt_tok_s\tload_s\ttotal_s\tvram\n' > "$RESULTS"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$(date +%Y-%m-%dT%H:%M:%S)" "$model" "$ec" "$rate" "$pc" "$prate" "$lds" "$tds" "$gpu_pct" >> "$RESULTS"
}

bench_selection() {
  local filter="$1" any=0 m
  for m in $(installed_models); do
    case "$m" in *embed*) continue ;; esac
    if [ "$filter" = "dense" ] && is_moe "$m"; then continue; fi
    if [ "$filter" = "moe" ]   && ! is_moe "$m"; then continue; fi
    any=1
    bench_one "$m" || true
  done
  [ "$any" = 1 ] || warn "no models matched '$filter' (try: ./track-a.sh models)"
}

# --------------------------------------------------------------------- commands
cmd_doctor() {
  echo "=== host ==="
  uname -srm
  echo "=== GPU ==="
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version \
      --format=csv,noheader
    nvidia-smi -q 2>/dev/null | grep -A2 'PCIe Generation' | grep -E 'Generation|Max' | head -2
  else
    warn "no nvidia-smi"
  fi
  echo "=== RAM ==="
  free -h
  echo "=== disk (root) ==="
  df -h / | tail -n1
  echo "=== ollama ==="
  if command -v ollama >/dev/null 2>&1; then
    ollama --version 2>/dev/null
    if ollama_up; then
      echo "server : $OLLAMA_HOST (up)"
      echo "loaded :"
      ollama ps 2>/dev/null | sed 's/^/         /'
    else
      warn "server not reachable at $OLLAMA_HOST"
    fi
  else
    warn "ollama not installed"
  fi
  echo "=== VM ==="
  if [ -x "$SCRIPT_DIR/run.sh" ]; then
    ( cd "$SCRIPT_DIR" && ./run.sh status 2>/dev/null ) || echo "vm: unknown"
  fi
  echo "=== tips ==="
  echo "  speed up ollama: OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0"
  echo "  (set these in the ollama systemd unit, then restart the service)"
}

cmd_free() {
  if [ -x "$SCRIPT_DIR/run.sh" ]; then
    log "stopping LLM VM to reclaim host RAM ..."
    ( cd "$SCRIPT_DIR" && ./run.sh stop ) || warn "run.sh stop returned non-zero"
  else
    warn "no ./run.sh found; skipping VM stop"
  fi
  echo
  free -h
}

cmd_models() {
  need_ollama
  ollama list
  echo
  echo "recommended MoE for this machine: $MOE_MODEL"
  have_model "$MOE_MODEL" && ok "MoE already installed" || warn "MoE not installed - run: ./track-a.sh pull moe"
}

cmd_pull() {
  need_ollama
  local want="${1:-moe}"
  [ "$want" = "moe" ] && want="$MOE_MODEL"
  log "pulling $want (this can be tens of GB) ..."
  ollama pull "$want"
}

cmd_bench() {
  need_ollama
  local model="${1:-}"
  [ -n "$model" ] || die "usage: ./track-a.sh bench <model-name>"
  have_model "$model" || die "model not installed: $model (see: ./track-a.sh models)"
  log "benchmarking $model (num_predict=$NUM_PREDICT ctx=$CTX)"
  bench_one "$model"
}

cmd_report() {
  if [ ! -f "$RESULTS" ]; then
    warn "no results yet at $RESULTS - run a benchmark first"
    return 0
  fi
  if command -v column >/dev/null 2>&1; then
    column -t -s $'\t' "$RESULTS"
  else
    cat "$RESULTS"
  fi
}

cmd_compare() {
  if [ ! -f "$RESULTS" ]; then warn "no results yet"; return 0; fi
  echo "=== latest run per model, ranked by generation speed ==="
  awk -F'\t' 'NR>1 {r[$2]=$4; g[$2]=$9; t[$2]=$1}
       END { for (m in r) printf "%s\t%s\t%s\n", r[m], m, g[m] }
  ' "$RESULTS" | sort -rn | awk -F'\t' \
    'BEGIN{printf "%-10s %-34s %s\n","tok/s","model","placement"} {printf "%-10s %-34s %s\n",$1,$2,$3}'
  echo
  if grep -qiE 'a3b|moe' "$RESULTS"; then
    ok "MoE result present: compare its tok/s against the dense 32B row above."
  else
    warn "no MoE result yet - run: ./track-a.sh pull moe && ./track-a.sh moe"
  fi
}

cmd_spec() {
  local img="${IMAGE:-ghcr.io/ggml-org/llama.cpp:full-cuda}"
  local target="${TARGET_GGUF:-$SCRIPT_DIR/models/deepseek-14b-q4.gguf}"
  local draft="${DRAFT_GGUF:-$SCRIPT_DIR/models/draft.gguf}"
  echo "Speculative decoding reads the big model once and verifies several draft"
  echo "tokens per pass - a real 2-4x for a bandwidth-bound model. It needs"
  echo "llama.cpp, not ollama. Requirements:"
  echo "  - docker + nvidia-container-toolkit  (set up: ./gpu-run.sh setup)"
  echo "  - a target GGUF  -> \$TARGET_GGUF (now: $target)"
  echo "  - a small draft  -> \$DRAFT_GGUF  (now: $draft)"
  echo
  if ! command -v docker >/dev/null 2>&1; then
    warn "docker is not installed; skipping."
    return 0
  fi
  if ! command -v nvidia-ctk >/dev/null 2>&1; then
    warn "nvidia-container-toolkit not installed. Run: ./gpu-run.sh setup"
    return 0
  fi
  if [ ! -f "$target" ] || [ ! -f "$draft" ]; then
    warn "target/draft GGUF missing; download them (see guest/models.conf) and retry."
    return 0
  fi
  log "running llama-server with speculative decoding ..."
  exec docker run --rm -it --gpus all --ulimit memlock=-1:-1 \
    -v "$(dirname "$target"):/models:ro" --entrypoint llama-server "$img" \
    -m "/models/$(basename "$target")" -md "/models/$(basename "$draft")" \
    -ngl 99 -c "$CTX" -fa on -ctk q8_0 -ctv q8_0 --host 0.0.0.0 --port 8080
}

cmd_everything() {
  cmd_free
  echo
  cmd_doctor
  echo
  need_ollama
  log "benchmarking every installed model ..."
  bench_selection all
  echo
  cmd_compare
  echo
  echo "next: install the MoE and re-run -> ./track-a.sh pull moe && ./track-a.sh moe && ./track-a.sh compare"
}

cmd_help() { awk 'NR>1 && /^#/ {sub(/^# ?/,""); print; next} NR>1 {exit}' "$0"; }

cmd="${1:-help}"
[ "$#" -gt 0 ] && shift || true
case "$cmd" in
  doctor)        cmd_doctor ;;
  free)          cmd_free ;;
  models|list)   cmd_models ;;
  pull)          cmd_pull "$@" ;;
  bench)         cmd_bench "$@" ;;
  dense)         need_ollama; log "benchmarking installed dense models ..."; bench_selection dense ;;
  moe)           need_ollama
                 if have_model "$MOE_MODEL"; then log "benchmarking MoE ..."; bench_selection moe
                 else warn "MoE '$MOE_MODEL' not installed - run: ./track-a.sh pull moe"; fi ;;
  compare)       cmd_compare ;;
  report)        cmd_report ;;
  spec)          cmd_spec ;;
  everything|all) cmd_everything ;;
  reset-results) rm -f "$RESULTS"; ok "removed $RESULTS" ;;
  help|-h|--help|"") cmd_help ;;
  *) die "unknown command: $cmd (try: ./track-a.sh help)" ;;
esac
