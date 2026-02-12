# DINOV3 + VGGT 3D Scene Builder (Gradio)

This repository contains a Python machine learning app that combines:

- **DINOV3** (private/invite-only checkpoint expected from user upload)
- **VGGT** (downloaded checkpoint support + custom training)
- **Gradio UI** for checkpoint management, training, and 3D reconstruction

The app can reconstruct a rough 3D scene from:

- a **single image**
- **multiple images**
- a **video** (sampled frames)

It also includes scripts to preprocess multimodal datasets (visible light, IR, satellite imagery, videos, metadata) and train DINOV3/VGGT independently.

---

## Repository Layout

```text
.
├── app.py
├── ml_app/
│   ├── config.py
│   ├── models.py
│   ├── reconstruction.py
│   └── training.py
├── scripts/
│   ├── download_vggt.py
│   ├── preprocess_dataset.py
│   ├── train_dinov3.py
│   └── train_vggt.py
├── models/
│   ├── dinov3_uploaded/   # place DINOV3 checkpoints here (private model)
│   └── vggt/              # downloaded/uploaded VGGT checkpoints
├── data/
│   ├── raw/
│   └── processed/
├── checkpoints/
└── outputs/
```

---

## 1) Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## 2) DINOV3 Model Upload (Private)

Because DINOV3 may be invite-only/private, this app does **not** attempt to auto-download it.

Use either:

- Gradio "Model Setup" tab to upload `.pt` checkpoint, or
- manually copy files into:

```text
models/dinov3_uploaded/
```

The training and reconstruction code automatically looks for the first `.pt` file in that folder.

---

## 3) Download VGGT Checkpoint

You can download any VGGT-compatible checkpoint from Hugging Face:

```bash
python scripts/download_vggt.py \
  --repo-id <your-vggt-repo> \
  --filename <checkpoint.pt> \
  --target-dir models/vggt
```

Then train/fine-tune with local data.

---

## 4) Dataset Preprocessing (Images, Video, Metadata)

Preprocess mixed datasets (visible, IR, satellite, video frames + metadata manifest):

```bash
python scripts/preprocess_dataset.py \
  --input-dir data/raw \
  --output-dir data/processed \
  --size 256 \
  --max-video-frames 32 \
  --metadata-json data/raw/metadata.json
```

This creates processed images + `manifest.json` in `data/processed`.

---

## 5) Train Models Separately

### Train DINOV3

```bash
python scripts/train_dinov3.py \
  --dataset-dir data/processed \
  --epochs 5 \
  --batch-size 8 \
  --lr 1e-4 \
  --device cpu
```

Checkpoint output:

- `checkpoints/dinov3_custom.pt`

### Train VGGT

```bash
python scripts/train_vggt.py \
  --dataset-dir data/processed \
  --epochs 5 \
  --batch-size 8 \
  --lr 1e-4 \
  --device cpu
```

Checkpoint output:

- `checkpoints/vggt_custom.pt`

---

## 6) Launch Gradio App

```bash
python app.py
```

Open the local URL printed by Gradio.

Tabs include:

1. **Model Setup** (upload DINOV3/VGGT checkpoints)
2. **Train DINOV3**
3. **Train VGGT**
4. **3D Reconstruction** (single image, multi-image, or video)

Generated point cloud is saved to:

- `outputs/scene_reconstruction.ply`

---

## Notes

- DINOV3 integration is designed for **local private checkpoint usage**.
- VGGT downloader is generic to support whichever checkpoint repo/file you have access to.
- The included default model wrappers are lightweight scaffolds intended to be replaced/adapted with your exact DINOV3/VGGT architectures and losses.
- For large-scale training, run on GPU and tune transforms/objectives for each modality (IR, satellite, RGB, depth, LiDAR/3D scenes, etc.).
