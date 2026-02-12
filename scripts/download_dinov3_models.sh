#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET_DIR="$ROOT_DIR/models/dinov3"
URLS_FILE="$TARGET_DIR/urls.txt"

mkdir -p "$TARGET_DIR"

if [[ ! -f "$URLS_FILE" ]]; then
  echo "Missing URLs file: $URLS_FILE" >&2
  exit 1
fi

echo "Downloading DINOv3 models into: $TARGET_DIR"

wget -c --content-disposition --directory-prefix "$TARGET_DIR" -i "$URLS_FILE"

echo "Done."
