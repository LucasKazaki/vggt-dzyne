from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import gradio as gr

from ml_app.config import CONFIG
from ml_app.reconstruction import reconstruct_scene
from ml_app.training import train_dinov3, train_vggt


def save_dinov3_checkpoint(file_obj) -> str:
    if file_obj is None:
        return "No checkpoint uploaded."

    src = Path(file_obj.name)
    dst = CONFIG.dinov3_dir / src.name
    dst.write_bytes(src.read_bytes())
    return f"Saved DINOV3 checkpoint to {dst}"


def save_vggt_checkpoint(file_obj) -> str:
    if file_obj is None:
        return "No checkpoint uploaded."

    src = Path(file_obj.name)
    dst = CONFIG.vggt_dir / src.name
    dst.write_bytes(src.read_bytes())
    return f"Saved VGGT checkpoint to {dst}"


def run_scene_reconstruction(images: Optional[List], video):
    image_paths = []
    if images:
        for img in images:
            image_paths.append(Path(img.name))

    video_path = Path(video.name) if video else None
    output = reconstruct_scene(image_paths=image_paths, video_path=video_path)
    return str(output)


with gr.Blocks(title="DINOV3 + VGGT 3D Scene Builder") as demo:
    gr.Markdown(
        "# DINOV3 + VGGT 3D Scene Builder\n"
        "Upload private DINOV3 checkpoints, train DINOV3/VGGT separately, and reconstruct 3D scenes from videos or images."
    )

    with gr.Tab("Model Setup"):
        gr.Markdown("### Upload model checkpoints")
        dinov3_upload = gr.File(label="DINOV3 checkpoint (.pt)")
        dinov3_btn = gr.Button("Save DINOV3")
        dinov3_status = gr.Textbox(label="DINOV3 status")
        dinov3_btn.click(save_dinov3_checkpoint, inputs=dinov3_upload, outputs=dinov3_status)

        vggt_upload = gr.File(label="VGGT checkpoint (.pt)")
        vggt_btn = gr.Button("Save VGGT")
        vggt_status = gr.Textbox(label="VGGT status")
        vggt_btn.click(save_vggt_checkpoint, inputs=vggt_upload, outputs=vggt_status)

    with gr.Tab("Train DINOV3"):
        d_dataset = gr.Textbox(label="Dataset folder path", value=str(CONFIG.data_processed_dir))
        d_epochs = gr.Slider(1, 100, value=5, step=1, label="Epochs")
        d_batch = gr.Slider(1, 64, value=8, step=1, label="Batch size")
        d_lr = gr.Number(value=1e-4, label="Learning rate")
        d_device = gr.Dropdown(["cpu", "cuda"], value="cpu", label="Device")
        d_run = gr.Button("Train DINOV3")
        d_log = gr.Textbox(label="Training output")
        d_run.click(train_dinov3, inputs=[d_dataset, d_epochs, d_batch, d_lr, d_device], outputs=d_log)

    with gr.Tab("Train VGGT"):
        v_dataset = gr.Textbox(label="Dataset folder path", value=str(CONFIG.data_processed_dir))
        v_epochs = gr.Slider(1, 100, value=5, step=1, label="Epochs")
        v_batch = gr.Slider(1, 64, value=8, step=1, label="Batch size")
        v_lr = gr.Number(value=1e-4, label="Learning rate")
        v_device = gr.Dropdown(["cpu", "cuda"], value="cpu", label="Device")
        v_run = gr.Button("Train VGGT")
        v_log = gr.Textbox(label="Training output")
        v_run.click(train_vggt, inputs=[v_dataset, v_epochs, v_batch, v_lr, v_device], outputs=v_log)

    with gr.Tab("3D Reconstruction"):
        images = gr.Files(label="Input images (single or multi-view)")
        video = gr.File(label="Optional video", file_types=["video"])
        recon_btn = gr.Button("Build 3D Scene")
        recon_out = gr.Textbox(label="Output point cloud path")
        recon_btn.click(run_scene_reconstruction, inputs=[images, video], outputs=recon_out)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
