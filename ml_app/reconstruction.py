from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import cv2
import numpy as np
import open3d as o3d
import torch
from PIL import Image
from torchvision import transforms

from .config import CONFIG
from .models import DINOV3Wrapper, VGGTWrapper


def _image_to_tensor(image: Image.Image) -> torch.Tensor:
    tfm = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ]
    )
    return tfm(image.convert("RGB")).unsqueeze(0)


def _extract_video_frames(video_path: Path, max_frames: int = 20) -> List[Image.Image]:
    cap = cv2.VideoCapture(str(video_path))
    frames: List[Image.Image] = []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    stride = max(total // max_frames, 1)
    idx = 0
    while cap.isOpened() and len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % stride == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame))
        idx += 1
    cap.release()
    return frames


def _depth_to_point_cloud(depth: np.ndarray, rgb: np.ndarray, fx: float = 120.0, fy: float = 120.0) -> o3d.geometry.PointCloud:
    h, w = depth.shape
    i, j = np.meshgrid(np.arange(w), np.arange(h))

    z = depth.flatten()
    x = (i.flatten() - w / 2) * z / fx
    y = (j.flatten() - h / 2) * z / fy

    points = np.stack((x, y, z), axis=1)
    colors = rgb.reshape(-1, 3) / 255.0

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def reconstruct_scene(image_paths: Iterable[Path], video_path: Path | None = None) -> Path:
    dinov3_ckpt = next(CONFIG.dinov3_dir.glob("*.pt"), None)
    vggt_ckpt = next(CONFIG.vggt_dir.glob("*.pt"), None)

    dino = DINOV3Wrapper(dinov3_ckpt).eval()
    vggt = VGGTWrapper(vggt_ckpt).eval()

    images: List[Image.Image] = [Image.open(p) for p in image_paths]
    if video_path:
        images.extend(_extract_video_frames(video_path))

    if not images:
        raise ValueError("No input images or frames provided.")

    merged_cloud = o3d.geometry.PointCloud()

    with torch.no_grad():
        for image in images:
            tensor = _image_to_tensor(image)
            _ = dino(tensor)
            depth = vggt(tensor).squeeze().cpu().numpy()

            depth = cv2.resize(depth, image.size)
            depth = depth - depth.min()
            depth = depth / (depth.max() + 1e-8)
            depth = 0.2 + depth * 3.0

            rgb = np.array(image.convert("RGB"))
            cloud = _depth_to_point_cloud(depth, rgb)
            merged_cloud += cloud

    merged_cloud = merged_cloud.voxel_down_sample(voxel_size=0.03)
    out_path = CONFIG.outputs_dir / "scene_reconstruction.ply"
    o3d.io.write_point_cloud(str(out_path), merged_cloud)
    return out_path
