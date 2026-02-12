from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from .config import CONFIG
from .models import DINOV3Wrapper, VGGTWrapper


class ImageFolderDataset(Dataset):
    def __init__(self, folder: Path):
        self.paths: List[Path] = []
        for ext in ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff", "*.bmp"):
            self.paths.extend(folder.rglob(ext))
        self.transform = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
            ]
        )

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        img = Image.open(self.paths[idx]).convert("RGB")
        x = self.transform(img)
        y = torch.flip(x, dims=[2])
        return x, y


@dataclass
class TrainConfig:
    dataset_dir: Path
    epochs: int = 5
    batch_size: int = 8
    lr: float = 1e-4
    device: str = "cpu"


def _run_training(model: nn.Module, cfg: TrainConfig, save_path: Path) -> str:
    dataset = ImageFolderDataset(cfg.dataset_dir)
    if len(dataset) == 0:
        return f"No images found in {cfg.dataset_dir}."

    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)
    model = model.to(cfg.device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    criterion = nn.L1Loss()

    losses = []
    for epoch in range(cfg.epochs):
        epoch_loss = 0.0
        for x, y in loader:
            x = x.to(cfg.device)
            y = y.to(cfg.device)

            optimizer.zero_grad()
            pred = model(x)

            if pred.ndim == 4 and pred.shape[1] == 1:
                pred = pred.repeat(1, 3, 4, 4)
                pred = pred[:, :, : y.shape[2], : y.shape[3]]
            elif pred.ndim == 2:
                pred = pred[:, :3].unsqueeze(-1).unsqueeze(-1).repeat(1, 1, y.shape[2], y.shape[3])

            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())

        avg = epoch_loss / max(len(loader), 1)
        losses.append(avg)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "losses": np.array(losses)}, save_path)
    return f"Training complete. Saved checkpoint: {save_path}. Final loss: {losses[-1]:.4f}"


def train_dinov3(dataset_dir: str, epochs: int, batch_size: int, lr: float, device: str = "cpu") -> str:
    cfg = TrainConfig(Path(dataset_dir), epochs=epochs, batch_size=batch_size, lr=lr, device=device)
    model = DINOV3Wrapper(next(CONFIG.dinov3_dir.glob("*.pt"), None))
    save_path = CONFIG.checkpoint_dir / "dinov3_custom.pt"
    return _run_training(model, cfg, save_path)


def train_vggt(dataset_dir: str, epochs: int, batch_size: int, lr: float, device: str = "cpu") -> str:
    cfg = TrainConfig(Path(dataset_dir), epochs=epochs, batch_size=batch_size, lr=lr, device=device)
    model = VGGTWrapper(next(CONFIG.vggt_dir.glob("*.pt"), None))
    save_path = CONFIG.checkpoint_dir / "vggt_custom.pt"
    return _run_training(model, cfg, save_path)
