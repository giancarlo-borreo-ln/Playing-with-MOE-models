#!/usr/bin/env python3
"""Robust single-file Hugging Face downloader used by download-models.sh."""
import os
import sys
import time

os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "0")

from huggingface_hub import hf_hub_download  # noqa: E402


def main() -> int:
    if len(sys.argv) != 4:
        print("usage: hf_download.py <repo_id> <filename> <dest_dir>", file=sys.stderr)
        return 2
    repo, filename, dest = sys.argv[1], sys.argv[2], sys.argv[3]
    os.makedirs(dest, exist_ok=True)
    last = None
    for attempt in range(1, 6):
        try:
            path = hf_hub_download(repo_id=repo, filename=filename, local_dir=dest)
            print(path)
            return 0
        except Exception as exc:  # noqa: BLE001
            last = exc
            print(f"attempt {attempt}/5 failed: {exc}", file=sys.stderr)
            if attempt < 5:
                time.sleep(5 * attempt)
    print(f"giving up on {repo}/{filename}: {last}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
