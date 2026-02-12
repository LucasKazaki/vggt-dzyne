#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import hf_hub_download


def main():
    parser = argparse.ArgumentParser(description="Download a VGGT checkpoint from Hugging Face")
    parser.add_argument("--repo-id", required=True, help="HF repo id that contains a VGGT checkpoint")
    parser.add_argument("--filename", required=True, help="Checkpoint file name inside the repo")
    parser.add_argument("--target-dir", default="models/vggt")
    parser.add_argument("--revision", default="main")
    args = parser.parse_args()

    target_dir = Path(args.target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    local_path = hf_hub_download(
        repo_id=args.repo_id,
        filename=args.filename,
        revision=args.revision,
        local_dir=target_dir,
        local_dir_use_symlinks=False,
    )
    print(f"Downloaded checkpoint to: {local_path}")


if __name__ == "__main__":
    main()
