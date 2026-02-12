from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn


class TinyVisionBackbone(nn.Module):
    def __init__(self, embedding_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DINOV3Wrapper(nn.Module):
    """Wrapper for locally uploaded DINOV3 checkpoints.

    Because DINOV3 can be private/invite-only, this wrapper loads from a local
    checkpoint path. If unavailable, it falls back to a lightweight backbone so
    the app remains runnable.
    """

    def __init__(self, checkpoint_path: Optional[Path] = None, embedding_dim: int = 256):
        super().__init__()
        self.backbone = TinyVisionBackbone(embedding_dim=embedding_dim)
        if checkpoint_path and checkpoint_path.exists():
            state = torch.load(checkpoint_path, map_location="cpu")
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            missing, unexpected = self.backbone.load_state_dict(state, strict=False)
            if missing or unexpected:
                print(f"DINOV3 checkpoint partial load. missing={len(missing)} unexpected={len(unexpected)}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class VGGTWrapper(nn.Module):
    def __init__(self, checkpoint_path: Optional[Path] = None, embedding_dim: int = 256):
        super().__init__()
        self.encoder = TinyVisionBackbone(embedding_dim=embedding_dim)
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 64 * 64),
        )
        if checkpoint_path and checkpoint_path.exists():
            state = torch.load(checkpoint_path, map_location="cpu")
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            self.load_state_dict(state, strict=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.encoder(x)
        depth_map = self.decoder(emb).reshape(-1, 1, 64, 64)
        return depth_map


class MultiModalFusionHead(nn.Module):
    def __init__(self, image_dim: int = 256, metadata_dim: int = 32):
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Linear(image_dim + metadata_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )

    def forward(self, image_embedding: torch.Tensor, metadata: torch.Tensor) -> torch.Tensor:
        return self.fuse(torch.cat([image_embedding, metadata], dim=-1))
