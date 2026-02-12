from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AppConfig:
    repo_root: Path = Path(__file__).resolve().parents[1]
    dinov3_dir: Path = repo_root / "models" / "dinov3_uploaded"
    vggt_dir: Path = repo_root / "models" / "vggt"
    checkpoint_dir: Path = repo_root / "checkpoints"
    data_raw_dir: Path = repo_root / "data" / "raw"
    data_processed_dir: Path = repo_root / "data" / "processed"
    outputs_dir: Path = repo_root / "outputs"


CONFIG = AppConfig()

for path in [
    CONFIG.dinov3_dir,
    CONFIG.vggt_dir,
    CONFIG.checkpoint_dir,
    CONFIG.data_raw_dir,
    CONFIG.data_processed_dir,
    CONFIG.outputs_dir,
]:
    path.mkdir(parents=True, exist_ok=True)
