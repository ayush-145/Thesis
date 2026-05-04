"""
config.py — Configuration for Model D: RETFound ViT-L/16 + LoRA
=================================================================
Single-model config for Kaggle / Google Colab / local execution.
RETFound was pretrained on 1.6M retinal images via MAE.
Training: RETFound weights + LoRA → fine-tune on APTOS (5-fold CV).
"""

import os
import sys
from dataclasses import dataclass, field
from typing import List, Tuple

MODEL_NAME = "ModelD_RETFound"
IMG_SIZE = 224  # RETFound native resolution

# ── Universal Environment Detection ──
IS_KAGGLE = "KAGGLE_KERNEL_RUN_TYPE" in os.environ
IS_COLAB = not IS_KAGGLE and (
    "COLAB_RELEASE_TAG" in os.environ
    or "google.colab" in sys.modules
    or os.path.exists("/content/drive")
)
COLAB_DRIVE_BASE = "/content/drive/MyDrive/DR_Thesis"

def _find_kaggle_dataset(name: str, fallback: str = "") -> str:
    import glob
    candidates = glob.glob(f"/kaggle/input/**/{name}", recursive=True)
    if candidates:
        return min(candidates, key=len)
    return fallback

def _resolve_path(kaggle_finder, kaggle_fallback, colab_path, local_path):
    if IS_KAGGLE:
        return kaggle_finder if kaggle_finder else kaggle_fallback
    elif IS_COLAB:
        return colab_path
    else:
        return local_path

@dataclass
class PathConfig:
    aptos_root: str = _resolve_path(
        _find_kaggle_dataset("aptos2019-blindness-detection") if IS_KAGGLE else "",
        "/kaggle/input/aptos2019-blindness-detection",
        "/content/aptos2019-blindness-detection",
        "d:/College/Thesis/Thesis/data/aptos",
    )
    aptos_train_dir: str = ""
    aptos_labels_csv: str = ""

    # RETFound pretrained weights
    retfound_hf_repo: str = "YukunZhou/RETFound_mae_natureCFP"
    retfound_hf_filename: str = "RETFound_mae_natureCFP.pth"
    retfound_weights: str = (
        (_find_kaggle_dataset("retfound-weights", "/kaggle/input/retfound-weights")
         + "/RETFound_cfp_weights.pth")
        if IS_KAGGLE
        else f"{COLAB_DRIVE_BASE}/weights/RETFound_cfp_weights.pth"
        if IS_COLAB
        else "d:/College/Thesis/Thesis/weights/RETFound_cfp_weights.pth"
    )

    output_dir: str = (
        "/kaggle/working/outputs" if IS_KAGGLE
        else f"{COLAB_DRIVE_BASE}/outputs/model_D" if IS_COLAB
        else "d:/College/Thesis/Thesis/outputs"
    )
    checkpoint_dir: str = ""
    plots_dir: str = ""
    logs_dir: str = ""
    exports_dir: str = ""

    def __post_init__(self):
        self.aptos_labels_csv = self._find_file(self.aptos_root, ["train.csv"], "APTOS labels CSV")
        self.aptos_train_dir = self._find_dir(self.aptos_root, ["train_images", "train"], "APTOS train images")
        self.checkpoint_dir = os.path.join(self.output_dir, "checkpoints")
        self.plots_dir = os.path.join(self.output_dir, "plots")
        self.logs_dir = os.path.join(self.output_dir, "logs")
        self.exports_dir = os.path.join(self.output_dir, "exports")
        for d in [self.checkpoint_dir, self.plots_dir, self.logs_dir, self.exports_dir]:
            os.makedirs(d, exist_ok=True)
        env = "Kaggle" if IS_KAGGLE else "Colab" if IS_COLAB else "Local"
        print(f"  [{env}] APTOS={self.aptos_root}")
        print(f"  [{env}] RETFound Weights={self.retfound_weights}")
        print(f"  [{env}] Output={self.output_dir}")

    @staticmethod
    def _find_file(root, candidates, label):
        import glob as _g
        for name in candidates:
            direct = os.path.join(root, name)
            if os.path.isfile(direct): return direct
            m = _g.glob(os.path.join(root, "**", name), recursive=True)
            if m: return m[0]
        print(f"  WARNING: {label} not found in {root}")
        return os.path.join(root, candidates[0])

    @staticmethod
    def _find_dir(root, candidates, label):
        import glob as _g
        for name in candidates:
            direct = os.path.join(root, name)
            if os.path.isdir(direct): return direct
            m = [x for x in _g.glob(os.path.join(root, "**", name), recursive=True) if os.path.isdir(x)]
            if m: return m[0]
        print(f"  WARNING: {label} directory not found in {root}")
        return os.path.join(root, candidates[0])

@dataclass
class DataConfig:
    img_size: int = 224  # RETFound native resolution
    num_classes: int = 5
    num_ranks: int = 4
    class_names: Tuple[str, ...] = ("No DR", "Mild", "Moderate", "Severe", "Proliferative")
    num_folds: int = 5
    random_seed: int = 42
    batch_size: int = 2 if IS_COLAB else 4
    grad_accum_steps: int = 8 if IS_COLAB else 4
    num_workers: int = 2
    pin_memory: bool = True
    gaussian_sigma_factor: float = 1 / 30
    clahe_clip_limit: float = 2.0
    clahe_tile_grid: Tuple[int, int] = (8, 8)
    img_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    img_std: Tuple[float, float, float] = (0.229, 0.224, 0.225)

@dataclass
class ModelConfig:
    retfound_embed_dim: int = 1024
    retfound_num_heads: int = 16
    retfound_depth: int = 24
    retfound_patch_size: int = 16
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: Tuple[str, ...] = ("qkv", "proj")

@dataclass
class TrainConfig:
    """APTOS fine-tuning with LoRA — RETFound backbone frozen."""
    epochs: int = 30
    lr: float = 2e-4
    weight_decay: float = 0.03     # Moderate (LoRA adapters are small)
    optimizer: str = "AdamW"
    betas: Tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    warmup_epochs: int = 5
    min_lr: float = 1e-6
    corn_alpha: float = 0.7
    focal_alpha_weight: float = 0.3
    focal_gamma: float = 2.0
    gradient_clip_max_norm: float = 1.0
    label_smoothing: float = 0.0
    dropout: float = 0.1
    early_stop_patience: int = 7
    early_stop_min_delta: float = 0.001
    use_amp: bool = True
    use_grad_checkpoint: bool = True

@dataclass
class EvalConfig:
    primary_metric: str = "qwk"
    umap_n_neighbors: int = 15
    umap_min_dist: float = 0.1
    umap_n_components: int = 2
    tsne_perplexity: float = 30.0
    tsne_n_iter: int = 1000
    fig_dpi: int = 300
    fig_format: str = "pdf"
    color_palette: str = "viridis"

@dataclass
class Config:
    paths: PathConfig = field(default_factory=PathConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

CFG = Config()
