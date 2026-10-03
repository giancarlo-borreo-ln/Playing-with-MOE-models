#!/usr/bin/env bash
#
# run.sh - boot a Debian 12 VM (KVM/QEMU) tuned for local LLM inference and SSH into it.
#
# Usage:
#   ./run.sh                 # first run builds+downloads, then opens an SSH shell
#   ./run.sh ssh             # open an SSH shell into a running VM
#   ./run.sh boot            # start the VM (and run first-boot setup) without opening a shell
#   ./run.sh stop            # gracefully shut the VM down
#   ./run.sh status          # show VM state
#   ./run.sh reset           # stop and wipe the VM disk (keeps base image + key)
#   ./run.sh purge           # reset and also delete base image, key and seed
#   ./run.sh download [m...] # (re)download models inside the VM
#   ./run.sh models          # list models and whether they are downloaded
#   ./run.sh logs            # stream the guest first-boot/setup log
#   ./run.sh serial          # tail the QEMU serial console log
#
# Everything is configurable through env vars, e.g.:
#   VM_RAM=8G VM_CPUS=4 SSH_PORT=2222 ./run.sh
#
# The VM runs models on CPU. To use the host RTX 3060 instead, see ./gpu-run.sh.
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ------------------------------------------------------------------ config
VM_NAME="${VM_NAME:-llm-vm}"
VM_DIR="${VM_DIR:-$SCRIPT_DIR/.vm}"
VM_RAM="${VM_RAM:-10G}"
VM_CPUS="${VM_CPUS:-6}"
DISK_SIZE="${DISK_SIZE:-512G}"
SSH_PORT="${SSH_PORT:-2222}"
SSH_USER="${SSH_USER:-debian}"
BASE_URL="${BASE_URL:-https://cloud.debian.org/images/cloud/bookworm/latest/debian-12-genericcloud-amd64.qcow2}"

BASE_IMG="$VM_DIR/debian-12-genericcloud-amd64.qcow2"
DISK="$VM_DIR/$VM_NAME.qcow2"
KEY="$VM_DIR/id_ed25519"
SEED="$VM_DIR/seed.iso"
PIDFILE="$VM_DIR/qemu.pid"
SERIAL="$VM_DIR/serial.log"
STATE="$VM_DIR/state"
INIT_MARK="$STATE/initialized"
MODELS_SEL="${VM_MODELS:-all}"

SSH_OPTS=(
  -o StrictHostKeyChecking=no
  -o UserKnownHostsFile=/dev/null
  -o LogLevel=ERROR
  -o ConnectTimeout=8
  -o ServerAliveInterval=30
  -o IdentitiesOnly=yes
  -p "$SSH_PORT"
  -i "$KEY"
)

# ------------------------------------------------------------------ helpers
c_cy="\033[1;36m"; c_rd="\033[1;31m"; c_yl="\033[1;33m"; c_rs="\033[0m"
log() { printf "${c_cy}[run]${c_rs} %s\n" "$*"; }
warn(){ printf "${c_yl}[run]${c_rs} %s\n" "$*" >&2; }
die() { printf "${c_rd}[run]${c_rs} %s\n" "$*" >&2; exit 1; }

vm_running() { [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE" 2>/dev/null)" 2>/dev/null; }
ssh_ready()  { ssh "${SSH_OPTS[@]}" "$SSH_USER@127.0.0.1" true 2>/dev/null; }
guest()      { ssh "${SSH_OPTS[@]}" "$SSH_USER@127.0.0.1" "$@"; }

# ------------------------------------------------------------------ stages
ensure_deps() {
  local t
  for t in qemu-system-x86_64 qemu-img curl ssh ssh-keygen xorriso python3; do
    command -v "$t" >/dev/null 2>&1 || die "missing dependency: $t"
  done
  if [ ! -w /dev/kvm ]; then
    warn "/dev/kvm is not writable - the VM will be emulated and MASSIVELY slower."
    warn "Fix with: sudo usermod -aG kvm \"$USER\"  (then re-login)"
  fi
  local host_mem_gb
  host_mem_gb=$(awk '/MemTotal/{printf "%d",$2/1024/1024}' /proc/meminfo)
  local want_gb="${VM_RAM%[Gg]}"
  if [ "$want_gb" -ge "$host_mem_gb" ] 2>/dev/null; then
    warn "VM_RAM=$VM_RAM vs host ${host_mem_gb}G: close the apps you don't need or set VM_RAM=8G."
  fi
}

ensure_base() {
  [ -f "$BASE_IMG" ] && return 0
  mkdir -p "$VM_DIR"
  log "downloading Debian 12 genericcloud image (~350 MB) ..."
  curl -L --fail --progress-bar -o "$BASE_IMG.part" "$BASE_URL" || die "download failed"
  mv "$BASE_IMG.part" "$BASE_IMG"
}

ensure_disk() {
  [ -f "$DISK" ] && return 0
  log "creating ${DISK_SIZE} sparse qcow2 overlay ..."
  qemu-img create -f qcow2 -F qcow2 -b "$BASE_IMG" -o lazy_refcounts=on "$DISK" "$DISK_SIZE" >/dev/null
}

ensure_key() {
  [ -f "$KEY" ] && return 0
  log "generating SSH key $KEY ..."
  ssh-keygen -t ed25519 -N '' -C "$VM_NAME" -f "$KEY" >/dev/null
}

ensure_seed() {
  log "building cloud-init seed ..."
  python3 "$SCRIPT_DIR/scripts/make_seed.py" \
    --guest-dir "$SCRIPT_DIR/guest" \
    --key "$KEY.pub" \
    --models "$MODELS_SEL" \
    --out "$SEED"
}

start_vm() {
  mkdir -p "$VM_DIR" "$STATE"
  vm_running && return 0
  log "starting VM '$VM_NAME' (ram=$VM_RAM cpus=$VM_CPUS disk=$DISK_SIZE, kvm) ..."
  : > "$SERIAL"
  qemu-system-x86_64 \
    -name "$VM_NAME" \
    -machine q35,accel=kvm \
    -cpu host \
    -smp "$VM_CPUS" \
    -m "$VM_RAM" \
    -drive file="$DISK",if=virtio,format=qcow2,cache=writeback,discard=unmap,detect-zeroes=unmap \
    -drive file="$SEED",if=virtio,format=raw,readonly=on \
    -netdev user,id=n0,hostfwd=tcp:127.0.0.1:"${SSH_PORT}"-:22 \
    -device virtio-net-pci,netdev=n0 \
    -device virtio-rng-pci \
    -display none \
    -monitor none \
    -serial "file:$SERIAL" \
    -daemonize -pidfile "$PIDFILE"
  vm_running || die "VM failed to start; see $SERIAL"
}

wait_ssh() {
  local i
  log "waiting for SSH on 127.0.0.1:$SSH_PORT ..."
  for i in $(seq 1 150); do
    ssh_ready && { log "SSH is up."; return 0; }
    if ! vm_running; then
      err_serial
      die "VM process died during boot."
    fi
    sleep 2
  done
  err_serial
  die "timed out waiting for SSH."
}

err_serial() {
  warn "last serial output:"
  [ -f "$SERIAL" ] && tail -n 30 "$SERIAL" >&2 || true
}

wait_setup() {
  log "first boot: compiling llama.cpp + downloading models."
  log "this can take a while (tens of GB) - streaming /var/log/vm-setup.log ..."
  ssh "${SSH_OPTS[@]}" -t "$SSH_USER@127.0.0.1" \
    "sudo touch /var/log/vm-setup.log; \
     sudo tail -n +1 -F /var/log/vm-setup.log & TP=\$!; \
     until sudo test -f /var/lib/vm-setup/ready; do sleep 3; done; \
     sleep 1; kill \$TP 2>/dev/null; \
     echo; echo '[run] guest setup complete.'" || true
}

require_running() {
  vm_running || die "VM is not running (start it with: $0)"
  ssh_ready || wait_ssh
}

# ------------------------------------------------------------------ commands
cmd_up() {
  local no_wait=0
  [ "${1:-}" = "--no-wait" ] && { no_wait=1; shift || true; }

  ensure_deps
  ensure_base
  local fresh=0
  [ -f "$DISK" ] || fresh=1
  ensure_disk
  ensure_key
  ensure_seed

  vm_running || start_vm
  wait_ssh

  if [ "$fresh" = 1 ] || [ ! -f "$INIT_MARK" ]; then
    if guest '[ -f /var/lib/vm-setup/ready ]' 2>/dev/null; then
      mkdir -p "$STATE"; : > "$INIT_MARK"
    elif [ "$no_wait" = 1 ]; then
      warn "guest setup still running; open a shell with '$0 ssh' or watch '$0 logs'."
      mkdir -p "$STATE"
    else
      wait_setup
      mkdir -p "$STATE"; : > "$INIT_MARK"
    fi
  fi

  log "opening SSH session into $VM_NAME ..."
  log "tip: run 'models' to list models, 'run' to chat, 'serve <alias>' for an API."
  ssh "${SSH_OPTS[@]}" -t "$SSH_USER@127.0.0.1" 'cat /etc/motd; exec bash -l'
}

cmd_ssh() {
  vm_running || cmd_up "$@"
  wait_ssh
  ssh "${SSH_OPTS[@]}" -t "$SSH_USER@127.0.0.1" 'cat /etc/motd; exec bash -l'
}

cmd_boot() {
  ensure_deps
  ensure_base
  ensure_disk
  ensure_key
  ensure_seed
  vm_running || start_vm
  wait_ssh
  log "VM is up. SSH: ssh -p $SSH_PORT -i $KEY $SSH_USER@127.0.0.1"
  log "guest setup status: $(guest '[ -f /var/lib/vm-setup/ready ] && echo ready || echo running')"
}

cmd_stop() {
  if ! vm_running; then log "VM not running."; return 0; fi
  log "shutting down ..."
  if ssh_ready; then guest 'sudo poweroff' || true; fi
  local i
  for i in $(seq 1 30); do vm_running || break; sleep 1; done
  if vm_running; then
    warn "graceful shutdown timed out, killing PID $(cat "$PIDFILE")"
    kill "$(cat "$PIDFILE")" 2>/dev/null || true
    sleep 2
  fi
  rm -f "$PIDFILE"
  log "VM stopped."
}

cmd_status() {
  if vm_running; then
    printf 'state:   %srunning%s (pid %s)\n' "$c_cy" "$c_rs" "$(cat "$PIDFILE")"
    printf 'ssh:     %s\n' "$(ssh_ready && echo reachable || echo 'not reachable yet')"
    printf 'config:  ram=%s cpus=%s disk=%s port=%s\n' "$VM_RAM" "$VM_CPUS" "$DISK_SIZE" "$SSH_PORT"
    printf 'disk:    %s\n' "$(du -h "$DISK" 2>/dev/null | cut -f1) used on host"
  else
    printf 'state:   %sstopped%s\n' "$c_rd" "$c_rs"
  fi
}

cmd_reset() {
  cmd_stop
  log "removing VM disk and state ..."
  rm -f "$DISK" "$SEED" "$INIT_MARK" "$PIDFILE" "$SERIAL"
  log "done. Next '$0' will boot a fresh Debian."
}

cmd_purge() {
  cmd_reset
  log "removing base image, key and state dir ..."
  rm -f "$BASE_IMG" "$KEY" "$KEY.pub"
  rmdir "$STATE" 2>/dev/null || true
  log "done."
}

cmd_download() {
  require_running
  if [ "$#" -gt 0 ]; then
    local quoted
    quoted=$(printf '%q ' "$@")
    guest "sudo /opt/llm/download-models.sh $quoted"
  else
    guest "sudo /opt/llm/download-models.sh"
  fi
}

cmd_models() {
  require_running
  guest "/usr/local/bin/models"
}

cmd_logs() {
  require_running
  ssh "${SSH_OPTS[@]}" -t "$SSH_USER@127.0.0.1" 'sudo tail -n +1 -F /var/log/vm-setup.log'
}

cmd_serial() {
  [ -f "$SERIAL" ] || die "no serial log yet"
  tail -n 100 -F "$SERIAL"
}

cmd_help() {
  awk 'NR>1 && /^#/ {sub(/^# ?/,""); print; next} NR>1 {exit}' "$0"
}

# ------------------------------------------------------------------ dispatch
cmd="${1:-up}"
[ "$#" -gt 0 ] && shift || true
case "$cmd" in
  up|"")      cmd_up "$@" ;;
  ssh|shell)  cmd_ssh "$@" ;;
  boot)       cmd_boot "$@" ;;
  stop|down)  cmd_stop "$@" ;;
  status)     cmd_status "$@" ;;
  reset)      cmd_reset "$@" ;;
  purge)      cmd_purge "$@" ;;
  download)   cmd_download "$@" ;;
  models)     cmd_models "$@" ;;
  logs)       cmd_logs "$@" ;;
  serial)     cmd_serial "$@" ;;
  help|-h|--help) cmd_help ;;
  *) die "unknown command: $cmd (try: $0 help)" ;;
esac
