#!/usr/bin/env python3
from __future__ import annotations

import argparse

from ml_app.training import train_dinov3


def main():
    parser = argparse.ArgumentParser(description="Train DINOV3 model on custom image folder")
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    print(train_dinov3(args.dataset_dir, args.epochs, args.batch_size, args.lr, args.device))


if __name__ == "__main__":
    main()
