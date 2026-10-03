#!/usr/bin/env python3
"""Build a cloud-init NoCloud seed ISO from the files in guest/.

The Debian genericcloud image ships cloud-init with the NoCloud datasource,
which reads user-data/meta-data from a filesystem labelled CIDATA.
"""
import argparse
import hashlib
import os
import subprocess
import sys
import tempfile

# (source file in guest dir, destination in VM, mode)
FILES = [
    ("guest-setup.sh",    "/opt/llm/guest-setup.sh",    "0755"),
    ("download-models.sh", "/opt/llm/download-models.sh", "0755"),
    ("hf_download.py",    "/opt/llm/hf_download.py",    "0755"),
    ("run_model.sh",      "/opt/llm/run_model.sh",      "0755"),
    ("models.conf",       "/opt/llm/models.conf",       "0644"),
    ("run",               "/usr/local/bin/run",         "0755"),
    ("serve",             "/usr/local/bin/serve",       "0755"),
    ("models",            "/usr/local/bin/models",      "0755"),
    ("download-models",   "/usr/local/bin/download-models", "0755"),
    ("motd",              "/etc/motd",                  "0644"),
    ("vm-setup.service",  "/etc/systemd/system/vm-setup.service", "0644"),
]


def indent(text: str, n: int) -> str:
    pad = " " * n
    return "".join(pad + line if line.strip("\n") else pad + "\n"
                   for line in text.splitlines(keepends=True))


def build_user_data(guest_dir: str, key: str, models: str) -> str:
    lines = []
    lines.append("#cloud-config")
    lines.append("hostname: llm-vm")
    lines.append("manage_etc_hosts: true")
    lines.append("ssh_pwauth: false")
    lines.append("ssh_authorized_keys:")
    lines.append("  - " + key.strip())
    lines.append("package_update: false")
    lines.append("package_upgrade: false")
    lines.append("write_files:")

    for src, dest, mode in FILES:
        path = os.path.join(guest_dir, src)
        with open(path, "r", encoding="utf-8") as fh:
            content = fh.read()
        if not content.endswith("\n"):
            content += "\n"
        lines.append(f"  - path: {dest}")
        lines.append("    owner: root:root")
        lines.append(f"    permissions: '{mode}'")
        lines.append("    content: |")
        lines.append(indent(content, 6))

    # selection controls which models download-models.sh fetches on first boot
    sel = [m for m in models.replace(",", " ").split() if m]
    if sel and sel != ["all"]:
        lines.append("  - path: /opt/llm/selection")
        lines.append("    owner: root:root")
        lines.append("    permissions: '0644'")
        lines.append("    content: |")
        lines.append(indent("".join(m + "\n" for m in sel), 6))

    lines.append("runcmd:")
    lines.append("  - [ systemctl, daemon-reload ]")
    lines.append("  - [ systemctl, enable, --now, vm-setup.service ]")
    lines.append('final_message: "llm-vm cloud-init finished"')
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--guest-dir", required=True)
    ap.add_argument("--key", required=True, help="public key file")
    ap.add_argument("--models", default="all")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    with open(args.key, "r", encoding="utf-8") as fh:
        key = fh.read().strip()
    if not key:
        print("error: empty public key", file=sys.stderr)
        return 1

    user_data = build_user_data(args.guest_dir, key, args.models)
    instance_id = "llm-vm-" + hashlib.sha256(user_data.encode()).hexdigest()[:12]
    meta_data = (
        f"instance-id: {instance_id}\n"
        "local-hostname: llm-vm\n"
    )

    with tempfile.TemporaryDirectory() as td:
        with open(os.path.join(td, "user-data"), "w", encoding="utf-8") as fh:
            fh.write(user_data)
        with open(os.path.join(td, "meta-data"), "w", encoding="utf-8") as fh:
            fh.write(meta_data)

        if os.path.exists(args.out):
            os.remove(args.out)
        subprocess.run(
            ["xorriso", "-as", "mkisofs",
             "-output", args.out,
             "-volid", "CIDATA",
             "-joliet", "-rock", "-quiet",
             td],
            check=True,
        )

    print(f"[seed] wrote {args.out} (instance-id={instance_id})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
