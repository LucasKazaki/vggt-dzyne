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

## How DINOV3 and VGGT Work (High-Level)

### DINOV3 (self-supervised visual representation learning)

DINOV3 belongs to the DINO family of models that learn strong visual features from large, mostly unlabeled image corpora using **self-distillation**.

At a high level, the training loop is:

1. Create multiple augmented views/crops of the same image.
2. Pass a global view through a **teacher** network.
3. Pass global + local views through a **student** network.
4. Train the student to match the teacher's output distribution across views.
5. Update the teacher from the student using an exponential moving average (EMA), rather than direct gradient updates.

This yields embeddings that transfer well to downstream tasks (classification, retrieval, segmentation, geometry pipelines) with limited labels.

Recommended references:

- DINO (foundational method): <https://arxiv.org/abs/2104.14294>
- DINOv2 (widely used successor): <https://arxiv.org/abs/2304.07193>
- DINOv3 papers/releases (latest): <https://arxiv.org/search/?query=dinov3&searchtype=all>

### VGGT (geometry-aware vision transformer for 3D reconstruction)

In this project, VGGT is used as the geometry-focused model in the 3D pipeline. Conceptually, a geometry transformer works by:

1. Encoding one or more images into patch/token features.
2. Exchanging context across views (or frames) with attention.
3. Regressing/decoding geometric signals such as depth, camera pose, correspondences, or point-level structure.
4. Aggregating those predictions into a coarse 3D representation (e.g., point cloud).

Our app combines DINOV3-style semantic features + VGGT-style geometric prediction to build robust scene reconstructions from a single image, multiple images, or sampled video frames.

Recommended references:

- VGGT papers/releases (latest): <https://arxiv.org/search/?query=vggt&searchtype=all>

> Note: this repository ships lightweight wrappers/scaffolds for local experimentation. For production-level fidelity, adapt the exact architecture, losses, data sampling, and camera/depth supervision strategy used by your chosen DINOV3/VGGT checkpoints.

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
