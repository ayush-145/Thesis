"""
config.py — Configuration for Model A: EfficientNetV2-S
========================================================
Single-model config for Kaggle / Google Colab / local execution.
All hyperparameters, paths, and constants in one place.
Designed for T4/P100 (16GB VRAM) execution.

Training: ImageNet weights → fine-tune directly on APTOS (5-fold CV).
"""

import os
import sys
from dataclasses import dataclass, field
from typing import List, Tuple


# ──────────────────────────────────────────────
# Model Identity
# ──────────────────────────────────────────────
MODEL_NAME = "ModelA_EfficientNetV2S"
IMG_SIZE = 512  # EfficientNetV2-S input resolution


# ──────────────────────────────────────────────
# Universal Environment Detection
# ──────────────────────────────────────────────
IS_KAGGLE = "KAGGLE_KERNEL_RUN_TYPE" in os.environ
IS_COLAB = not IS_KAGGLE and (
    "COLAB_RELEASE_TAG" in os.environ
    or "google.colab" in sys.modules
    or os.path.exists("/content/drive")
)

# Google Drive base path for Colab (mounted at /content/drive/MyDrive)
COLAB_DRIVE_BASE = "/content/drive/MyDrive/DR_Thesis"


def _find_kaggle_dataset(name: str, fallback: str = "") -> str:
    """
    Auto-discover a Kaggle input dataset by name.

    Kaggle mounts datasets at varying paths depending on type:
      - Competitions:  /kaggle/input/competitions/<name>/
      - User datasets: /kaggle/input/datasets/<user>/<name>/
      - Legacy:        /kaggle/input/<name>/

    This function searches all possibilities via glob and returns
    the first match, so hardcoded paths never break.
    """
    import glob
    candidates = glob.glob(f"/kaggle/input/**/{name}", recursive=True)
    if candidates:
        # Prefer shortest path (most direct mount)
        result = min(candidates, key=len)
        return result
    return fallback


def _resolve_path(kaggle_finder, kaggle_fallback, colab_path, local_path):
    """Resolve path based on runtime environment."""
    if IS_KAGGLE:
        return kaggle_finder if kaggle_finder else kaggle_fallback
    elif IS_COLAB:
        return colab_path
    else:
        return local_path


@dataclass
class PathConfig:
    """Dataset and output paths — auto-switches between Kaggle, Colab, and local."""

    # ── APTOS Dataset ──
    aptos_root: str = _resolve_path(
        _find_kaggle_dataset("aptos2019-blindness-detection") if IS_KAGGLE else "",
        "/kaggle/input/aptos2019-blindness-detection",
        "/content/aptos2019-blindness-detection",
        "d:/College/Thesis/Thesis/data/aptos",
    )
    aptos_train_dir: str = ""
    aptos_labels_csv: str = ""

    # ── Output ──
    output_dir: str = (
        "/kaggle/working/outputs" if IS_KAGGLE
        else f"{COLAB_DRIVE_BASE}/outputs/model_A" if IS_COLAB
        else "d:/College/Thesis/Thesis/outputs"
    )
    checkpoint_dir: str = ""
    plots_dir: str = ""
    logs_dir: str = ""
    exports_dir: str = ""  # For .npy ensemble export files

    def __post_init__(self):
        # ── APTOS sub-paths (auto-discover) ──
        self.aptos_labels_csv = self._find_file(
            self.aptos_root, ["train.csv"], "APTOS labels CSV",
        )
        self.aptos_train_dir = self._find_dir(
            self.aptos_root, ["train_images", "train"], "APTOS train images",
        )

        # Output sub-directories
        self.checkpoint_dir = os.path.join(self.output_dir, "checkpoints")
        self.plots_dir = os.path.join(self.output_dir, "plots")
        self.logs_dir = os.path.join(self.output_dir, "logs")
        self.exports_dir = os.path.join(self.output_dir, "exports")

        # Create output directories
        for d in [self.checkpoint_dir, self.plots_dir, self.logs_dir, self.exports_dir]:
            os.makedirs(d, exist_ok=True)

        # Print resolved paths for debugging
        env = "Kaggle" if IS_KAGGLE else "Colab" if IS_COLAB else "Local"
        print(f"  [{env}] Resolved paths:")
        print(f"    APTOS root:  {self.aptos_root}")
        print(f"    APTOS csv:   {self.aptos_labels_csv}")
        print(f"    APTOS imgs:  {self.aptos_train_dir}")
        print(f"    Output:      {self.output_dir}")

    @staticmethod
    def _find_file(root: str, candidates: list, label: str) -> str:
        """Search for a file by name inside root (recursively)."""
        import glob as _glob
        for name in candidates:
            # Direct path
            direct = os.path.join(root, name)
            if os.path.isfile(direct):
                return direct
            # Recursive search
            matches = _glob.glob(os.path.join(root, "**", name), recursive=True)
            if matches:
                return matches[0]
        # Not found — print what IS there for debugging
        print(f"  WARNING: {label} not found in {root}")
        print(f"    Searched for: {candidates}")
        if os.path.isdir(root):
            for item in sorted(os.listdir(root))[:20]:
                full = os.path.join(root, item)
                kind = "DIR" if os.path.isdir(full) else "FILE"
                print(f"      [{kind}] {item}")
        else:
            print(f"      (directory does not exist)")
        return os.path.join(root, candidates[0])  # fallback

    @staticmethod
    def _find_dir(root: str, candidates: list, label: str) -> str:
        """Search for a directory by name inside root (recursively)."""
        import glob as _glob
        for name in candidates:
            direct = os.path.join(root, name)
            if os.path.isdir(direct):
                return direct
            matches = _glob.glob(os.path.join(root, "**", name), recursive=True)
            matches = [m for m in matches if os.path.isdir(m)]
            if matches:
                return matches[0]
        # Not found
        print(f"  WARNING: {label} directory not found in {root}")
        print(f"    Searched for: {candidates}")
        if os.path.isdir(root):
            for item in sorted(os.listdir(root))[:20]:
                full = os.path.join(root, item)
                if os.path.isdir(full):
                    print(f"      [DIR] {item}")
        return os.path.join(root, candidates[0])  # fallback


@dataclass
class DataConfig:
    """Data pipeline configuration."""

    # ── Image Resolution ──
    img_size: int = 512            # EfficientNetV2-S input resolution

    # ── DR Grades ──
    num_classes: int = 5           # 0=No DR, 1=Mild, 2=Moderate, 3=Severe, 4=Proliferative
    num_ranks: int = 4             # K-1 binary tasks for CORN ordinal regression
    class_names: Tuple[str, ...] = (
        "No DR", "Mild", "Moderate", "Severe", "Proliferative"
    )

    # ── Cross-Validation ──
    num_folds: int = 5
    random_seed: int = 42

    # ── DataLoader (Colab has ~12.7 GB RAM vs Kaggle's ~30 GB) ──
    batch_size: int = 2 if IS_COLAB else 4
    grad_accum_steps: int = 8 if IS_COLAB else 4  # Effective batch = 16
    num_workers: int = 2
    pin_memory: bool = True

    # ── Preprocessing (Ben Graham) ──
    gaussian_sigma_factor: float = 1 / 30  # σ = image_width × factor
    clahe_clip_limit: float = 2.0
    clahe_tile_grid: Tuple[int, int] = (8, 8)

    # ── Normalization (ImageNet) ──
    img_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    img_std: Tuple[float, float, float] = (0.229, 0.224, 0.225)


@dataclass
class ModelConfig:
    """Architecture configuration for EfficientNetV2-S."""

    # ── EfficientNetV2-S (Model A) ──
    effnet_feature_dim: int = 1280  # Penultimate feature dimension


@dataclass
class TrainConfig:
    """Training regimen configuration — APTOS-only fine-tuning from ImageNet."""

    # ── Fine-tuning on APTOS ──
    epochs: int = 30
    lr: float = 1e-5
    weight_decay: float = 0.03     # Increased for APTOS-only (combat overfitting)

    # ── Optimizer ──
    optimizer: str = "AdamW"
    betas: Tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8

    # ── Scheduler: Cosine Annealing with Linear Warmup ──
    warmup_epochs: int = 3
    min_lr: float = 1e-7

    # ── Loss ──
    corn_alpha: float = 0.7        # Weight for CORN loss
    focal_alpha_weight: float = 0.3 # Weight for Focal loss
    focal_gamma: float = 2.0       # Focal loss focusing parameter

    # ── Regularization ──
    gradient_clip_max_norm: float = 1.0
    label_smoothing: float = 0.0   # Disabled (incompatible with CORN ordinal targets)
    dropout: float = 0.15          # Classifier head dropout (increased for small dataset)

    # ── Early Stopping ──
    early_stop_patience: int = 7   # Stop if val QWK doesn't improve for N epochs
    early_stop_min_delta: float = 0.001

    # ── Mixed Precision ──
    use_amp: bool = True           # Automatic Mixed Precision (FP16)

    # ── Gradient Checkpointing ──
    use_grad_checkpoint: bool = True


@dataclass
class EvalConfig:
    """Evaluation and visualization configuration."""

    # ── Primary Metric ──
    primary_metric: str = "qwk"    # Quadratic Weighted Kappa

    # ── Visualization ──
    umap_n_neighbors: int = 15
    umap_min_dist: float = 0.1
    umap_n_components: int = 2
    tsne_perplexity: float = 30.0
    tsne_n_iter: int = 1000

    # ── Plot Settings ──
    fig_dpi: int = 300
    fig_format: str = "pdf"        # Publication-quality
    color_palette: str = "viridis"


# ──────────────────────────────────────────────
# Global Config Instance
# ──────────────────────────────────────────────
@dataclass
class Config:
    """Master configuration — aggregates all sub-configs."""
    paths: PathConfig = field(default_factory=PathConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    def __repr__(self):
        lines = [f"{'='*60}", f"DR Ablation Study — {MODEL_NAME} Configuration", f"{'='*60}"]
        for section_name in ["paths", "data", "model", "train", "eval"]:
            section = getattr(self, section_name)
            lines.append(f"\n[{section_name.upper()}]")
            for k, v in section.__dict__.items():
                lines.append(f"  {k}: {v}")
        return "\n".join(lines)


# Singleton
CFG = Config()
