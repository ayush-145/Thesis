"""
dataset.py — Dataset & DataLoader for DR Ablation Study
========================================================
Handles APTOS dataset for fine-tuning.
Features:
  - On-the-fly Ben Graham preprocessing
  - CORN ordinal label encoding
  - Strong spatial augmentations (albumentations)
  - Stratified 5-Fold cross-validation
  - Environment-adaptive DataLoader settings
"""

import os
import cv2
import numpy as np
import pandas as pd
import torch
from typing import Optional, Tuple
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Dict, List, Optional, Tuple

from config import CFG
from preprocessing import preprocess_fundus


# ──────────────────────────────────────────────
# CORN Ordinal Label Encoding
# ──────────────────────────────────────────────

def encode_corn_label(grade: int, num_ranks: int = 4) -> np.ndarray:
    """
    Encode a DR grade (0-4) into CORN binary ordinal targets.

    CORN decomposes K-class ordinal classification into K-1 binary tasks:
      "Is the severity strictly greater than rank k?"

    Examples:
        Grade 0 → [0, 0, 0, 0]  (not > any rank)
        Grade 1 → [1, 0, 0, 0]  (> rank 0 only)
        Grade 2 → [1, 1, 0, 0]  (> rank 0 and 1)
        Grade 3 → [1, 1, 1, 0]  (> rank 0, 1, and 2)
        Grade 4 → [1, 1, 1, 1]  (> all ranks)

    Args:
        grade: Integer DR grade in [0, num_ranks].
        num_ranks: Number of binary tasks (K-1).

    Returns:
        Binary array of shape (num_ranks,).
    """
    label = np.zeros(num_ranks, dtype=np.float32)
    label[:grade] = 1.0
    return label


def decode_corn_prediction(logits: torch.Tensor) -> torch.Tensor:
    """
    Convert CORN sigmoid logits to predicted class labels.

    Strategy: Apply sigmoid → threshold at 0.5 → count consecutive 1s.
    The predicted grade = number of ranks where P(Y > k) > 0.5.

    Args:
        logits: Raw logits of shape (B, num_ranks).

    Returns:
        Predicted class labels of shape (B,), integers in [0, K].
    """
    probas = torch.sigmoid(logits)
    predicted = (probas > 0.5).long()
    # Sum consecutive 1s from the left
    return predicted.sum(dim=1)


# ──────────────────────────────────────────────
# Augmentation Pipelines
# ──────────────────────────────────────────────

def _build_augmentation_list(img_size: int) -> list:
    """
    Build augmentation list compatible with both albumentations v1.x and v2.x.
    Detects the installed version and uses the correct API.
    """
    import albumentations as _A
    v2 = int(_A.__version__.split('.')[0]) >= 2  # v2.x check

    crop_kwargs = {"size": (img_size, img_size)} if v2 else {"height": img_size, "width": img_size}

    augments = [
        A.RandomResizedCrop(
            **crop_kwargs,
            scale=(0.8, 1.0),
            ratio=(0.9, 1.1),
            interpolation=cv2.INTER_CUBIC,
            p=1.0,
        ),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
    ]

    # Geometric: Affine (v2) or ShiftScaleRotate (v1)
    if v2 and hasattr(A, 'Affine'):
        augments.append(A.Affine(
            translate_percent=(-0.05, 0.05),
            scale=(0.9, 1.1),
            rotate=(-30, 30),
            mode=cv2.BORDER_CONSTANT,
            p=0.5,
        ))
    else:
        augments.append(A.ShiftScaleRotate(
            shift_limit=0.05, scale_limit=0.1, rotate_limit=30,
            border_mode=cv2.BORDER_CONSTANT, p=0.5,
        ))

    # Distortions
    distortions = [
        A.ElasticTransform(alpha=1.0, sigma=50, p=1.0),
        A.GridDistortion(num_steps=5, distort_limit=0.3, p=1.0),
    ]
    if v2:
        distortions.append(A.OpticalDistortion(distort_limit=0.3, p=1.0))
    else:
        distortions.append(A.OpticalDistortion(distort_limit=0.3, shift_limit=0.1, p=1.0))
    augments.append(A.OneOf(distortions, p=0.3))

    # Photometric
    augments.append(A.OneOf([
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
        A.HueSaturationValue(
            hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=15, p=1.0
        ),
    ], p=0.4))

    # Noise (v2 uses std_range, v1 uses var_limit)
    if v2:
        augments.append(A.GaussNoise(std_range=(0.02, 0.1), p=0.2))
    else:
        augments.append(A.GaussNoise(var_limit=(5.0, 25.0), p=0.2))

    augments.append(A.GaussianBlur(blur_limit=(3, 5), p=0.2))

    # CoarseDropout (v2 changed param names)
    if v2:
        augments.append(A.CoarseDropout(
            num_holes_range=(1, 8),
            hole_height_range=(img_size // 32, img_size // 16),
            hole_width_range=(img_size // 32, img_size // 16),
            p=0.3,
        ))
    else:
        augments.append(A.CoarseDropout(
            max_holes=8, max_height=img_size // 16,
            max_width=img_size // 16, fill_value=0, p=0.3,
        ))

    # Normalize + ToTensor
    augments.extend([
        A.Normalize(mean=CFG.data.img_mean, std=CFG.data.img_std, max_pixel_value=255.0),
        ToTensorV2(),
    ])

    return augments


def get_train_transforms(img_size: int) -> A.Compose:
    """
    Strong spatial augmentations for training.
    Auto-detects albumentations version for API compatibility.
    """
    return A.Compose(_build_augmentation_list(img_size))


def get_val_transforms(img_size: int) -> A.Compose:
    """
    Minimal transforms for validation/test: just resize + normalize.
    """
    return A.Compose([
        A.Resize(img_size, img_size, interpolation=cv2.INTER_CUBIC),
        A.Normalize(
            mean=CFG.data.img_mean,
            std=CFG.data.img_std,
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ])


# ──────────────────────────────────────────────
# Dataset Class
# ──────────────────────────────────────────────

class DRDataset(Dataset):
    """
    Dataset for APTOS fundus images.

    Handles:
    - On-the-fly Ben Graham preprocessing (crop → blur sub → CLAHE)
    - CORN ordinal label encoding
    - Albumentations augmentation
    """

    def __init__(
        self,
        image_paths: List[str],
        labels: List[int],
        img_size: int = 512,
        transform: Optional[A.Compose] = None,
        preprocess: bool = True,
    ):
        """
        Args:
            image_paths: List of absolute paths to fundus images.
            labels: List of integer DR grades (0-4).
            img_size: Target image size (512 for CNN/Swin, 224 for RETFound).
            transform: Albumentations transform pipeline.
            preprocess: Whether to apply Ben Graham preprocessing.
        """
        assert len(image_paths) == len(labels)
        self.image_paths = list(image_paths)
        self.labels = list(labels)
        self.img_size = img_size
        self.transform = transform
        self.preprocess = preprocess
        self.num_ranks = CFG.data.num_ranks

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Load image
        image = cv2.imread(self.image_paths[idx])
        if image is None:
            # Fallback: return a black image (shouldn't happen after filtering)
            image = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)

        # Ben Graham preprocessing (crop, resize, blur sub, CLAHE)
        if self.preprocess:
            image = preprocess_fundus(
                image,
                target_size=self.img_size,
                sigma_factor=CFG.data.gaussian_sigma_factor,
                clip_limit=CFG.data.clahe_clip_limit,
                tile_grid_size=CFG.data.clahe_tile_grid,
            )
        else:
            image = cv2.resize(image, (self.img_size, self.img_size))

        # Convert BGR → RGB for albumentations / PyTorch
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply augmentations
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]  # Tensor (C, H, W)
        else:
            # Manual fallback: normalize and convert
            image = image.astype(np.float32) / 255.0
            image = torch.from_numpy(image).permute(2, 0, 1)

        # Labels
        grade = self.labels[idx]
        corn_label = encode_corn_label(grade, self.num_ranks)

        return {
            "image": image,                                     # (C, H, W) float32
            "grade": torch.tensor(grade, dtype=torch.long),     # scalar int
            "corn_label": torch.from_numpy(corn_label),          # (num_ranks,) float32
        }


# ──────────────────────────────────────────────
# DataLoader Factories
# ──────────────────────────────────────────────

def load_aptos_dataframe() -> pd.DataFrame:
    """Load and validate the APTOS CSV."""
    csv_path = CFG.paths.aptos_labels_csv
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"APTOS CSV not found at {csv_path}")

    df = pd.read_csv(csv_path)
    df.columns = ["id_code", "diagnosis"]

    # ── Robust Path Resolution ──
    import glob
    print(f"  Scanning APTOS directory for images...")
    all_files = glob.glob(os.path.join(CFG.paths.aptos_train_dir, "**", "*.*"), recursive=True)
    
    img_map = {}
    for p in all_files:
        base = os.path.splitext(os.path.basename(p))[0]
        img_map[base] = p

    df["image_path"] = df["id_code"].astype(str).map(img_map)

    exists_mask = df["image_path"].notnull()
    if not exists_mask.all():
        missing = (~exists_mask).sum()
        total = len(df)
        print(f"  WARNING: {missing}/{total} APTOS images not found in directory tree.")
        df = df[exists_mask].reset_index(drop=True)

    print(f"  APTOS: {len(df)} images loaded")
    print(f"  Class distribution:\n{df['diagnosis'].value_counts().sort_index()}\n")
    return df


def get_stratified_kfold_splits(
    df: pd.DataFrame,
    label_col: str = "diagnosis",
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Generate stratified K-fold splits.

    Returns:
        List of (train_indices, val_indices) tuples, one per fold.
    """
    skf = StratifiedKFold(
        n_splits=CFG.data.num_folds,
        shuffle=True,
        random_state=CFG.data.random_seed,
    )
    splits = list(skf.split(df.index, df[label_col]))
    for i, (train_idx, val_idx) in enumerate(splits):
        train_dist = df.iloc[train_idx][label_col].value_counts().sort_index()
        val_dist = df.iloc[val_idx][label_col].value_counts().sort_index()
        print(f"  Fold {i+1}: train={len(train_idx)}, val={len(val_idx)}")
    return splits


def build_dataloaders(
    df: pd.DataFrame,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    img_size: int,
    label_col: str = "diagnosis",
) -> Tuple[DataLoader, DataLoader]:
    """
    Build train and validation DataLoaders for a single fold.

    Args:
        df: DataFrame with 'image_path' and label_col columns.
        train_idx, val_idx: Indices for this fold.
        img_size: Target image resolution.
        label_col: Column name for DR grade labels.

    Returns:
        (train_loader, val_loader)
    """
    train_df = df.iloc[train_idx]
    val_df = df.iloc[val_idx]

    train_dataset = DRDataset(
        image_paths=train_df["image_path"].tolist(),
        labels=train_df[label_col].tolist(),
        img_size=img_size,
        transform=get_train_transforms(img_size),
        preprocess=True,
    )

    val_dataset = DRDataset(
        image_paths=val_df["image_path"].tolist(),
        labels=val_df[label_col].tolist(),
        img_size=img_size,
        transform=get_val_transforms(img_size),
        preprocess=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=CFG.data.batch_size,
        shuffle=True,
        num_workers=CFG.data.num_workers,
        pin_memory=CFG.data.pin_memory,
        drop_last=True,
        persistent_workers=True if CFG.data.num_workers > 0 else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=CFG.data.batch_size * 2,  # Larger batch for inference
        shuffle=False,
        num_workers=CFG.data.num_workers,
        pin_memory=CFG.data.pin_memory,
        drop_last=False,
        persistent_workers=True if CFG.data.num_workers > 0 else False,
    )

    return train_loader, val_loader
