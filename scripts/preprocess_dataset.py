#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def preprocess_image(path: Path, out_dir: Path, size: int):
    img = Image.open(path).convert("RGB").resize((size, size))
    arr = np.array(img)
    arr = cv2.fastNlMeansDenoisingColored(arr, None, 7, 7, 7, 21)
    out_path = out_dir / f"{path.stem}_proc.png"
    Image.fromarray(arr).save(out_path)
    return out_path


def preprocess_video(path: Path, out_dir: Path, size: int, max_frames: int):
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    stride = max(total // max_frames, 1)
    idx, kept = 0, 0
    saved = []
    while cap.isOpened() and kept < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % stride == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (size, size))
            out_path = out_dir / f"{path.stem}_f{kept:04d}.png"
            Image.fromarray(frame).save(out_path)
            saved.append(out_path)
            kept += 1
        idx += 1
    cap.release()
    return saved


def main():
    parser = argparse.ArgumentParser(description="Preprocess multimodal data for DINOV3/VGGT training")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--max-video-frames", type=int, default=32)
    parser.add_argument("--metadata-json", default="")
    args = parser.parse_args()

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    processed = {"images": [], "videos": [], "metadata": None}

    for p in in_dir.rglob("*"):
        if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}:
            processed["images"].append(str(preprocess_image(p, out_dir, args.size)))
        elif p.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv"}:
            frames = preprocess_video(p, out_dir, args.size, args.max_video_frames)
            processed["videos"].extend([str(f) for f in frames])

    if args.metadata_json:
        metadata_path = Path(args.metadata_json)
        if metadata_path.exists():
            processed["metadata"] = json.loads(metadata_path.read_text())

    manifest = out_dir / "manifest.json"
    manifest.write_text(json.dumps(processed, indent=2))
    print(f"Saved preprocessed dataset and manifest: {manifest}")


if __name__ == "__main__":
    main()
