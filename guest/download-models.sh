#!/usr/bin/env bash
#
# Download the GGUF models listed in /opt/llm/models.conf.
# With no arguments, downloads every model (or the first-boot selection in
# /opt/llm/selection). Pass aliases to fetch only those.
#
set -uo pipefail

CONF="${LLM_CONF:-/opt/llm/models.conf}"
MODEL_DIR="${LLM_MODEL_DIR:-/opt/models}"
PY=/opt/llm/venv/bin/python
DL=/opt/llm/hf_download.py
SELECTION_FILE=/opt/llm/selection

mkdir -p "$MODEL_DIR"

sel=("$@")
if [ "${#sel[@]}" -eq 0 ] && [ -f "$SELECTION_FILE" ]; then
  mapfile -t sel < "$SELECTION_FILE"
fi

in_selection() {
  local want="$1" s
  [ "${#sel[@]}" -eq 0 ] && return 0          # no filter -> everything
  for s in "${sel[@]}"; do
    [ "$s" = "all" ] && return 0
    [ "$s" = "$want" ] && return 0
  done
  return 1
}

fail=0
printf '%-14s %-10s %-6s %s\n' ALIAS COMPANY SIZE FILE
while IFS='|' read -r alias company size repo file cpu gpu; do
  case "$alias" in ''|'#'*) continue ;; esac
  in_selection "$alias" || continue
  if [ -f "$MODEL_DIR/$file" ]; then
    printf '%-14s %-10s %-6s [have] %s\n' "$alias" "$company" "$size" "$file"
    continue
  fi
  printf '%-14s %-10s %-6s [get ] %s\n' "$alias" "$company" "$size" "$file"
  printf '        <- %s\n' "$repo"
  if "$PY" "$DL" "$repo" "$file" "$MODEL_DIR"; then
    printf '        [ ok ] %s\n' "$alias"
  else
    printf '        [FAIL] %s (will retry on next run)\n' "$alias"
    fail=1
  fi
done < "$CONF"

echo
echo "models in $MODEL_DIR:"
ls -lh "$MODEL_DIR"/*.gguf 2>/dev/null || echo "  (none downloaded)"
exit "$fail"
