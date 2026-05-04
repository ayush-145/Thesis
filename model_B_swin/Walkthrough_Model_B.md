# Model B: Swin-V2-Tiny Pipeline Walkthrough

## 1. Architectural Overview
Model B serves as the **Pure Vision Transformer (ViT) baseline**. 
- **Backbone**: Swin Transformer V2 Tiny (Swin-V2-T, pretrained on ImageNet)
- **Input Resolution**: 512 × 512
- **Feature Extraction**: Shifted-Window Self-Attention. Unlike the CNN, Swin utilizes hierarchical self-attention to model long-range global dependencies across the fundus (e.g., correlating a hemorrhage in the top left with exudates in the bottom right).
- **Classification Head**: Swin Patch Merging → LayerNorm → AvgPool → Flatten → CORN Head.
- **VRAM Hardening**: Checkpoints the Transformer encoder stages to heavily reduce VRAM footprint during the backward pass.

## 2. Preprocessing & Data Augmentation
Shares the identical pipeline as Model A to ensure a fair architectural ablation:
1. **Ben Graham Spatial Preprocessing** (`preprocessing.py`):
   - Auto-cropping of black borders.
   - Lighting normalization via Gaussian blur subtraction.
   - Contrast enhancement via Green Channel CLAHE.
2. **Albumentations Augmentation** (`dataset.py`):
   - Geometric: `RandomResizedCrop`, `HorizontalFlip`, `VerticalFlip`.
   - Photometric: `RandomBrightnessContrast`, `HueSaturationValue`, `GaussianBlur`.
   - Normalization: ImageNet standard mean/std.

## 3. Handling Class Imbalance
1. **Hybrid Ordinal-Focal Loss** (`losses.py`):
   - Uses the same `alpha * CORN + (1-alpha) * OrdinalFocalLoss` formulation. The `gamma=2.0` parameter forces the Swin Transformer to pay exponentially more attention to the sparse minority classes (Severe/Proliferative DR) that it is getting wrong, rather than over-optimizing on the majority "No DR" class.
2. **Stratified K-Fold** (`train.py`): 
   - Maintains the exact same 5-fold splits as Model A (`random_state=42`), guaranteeing identical class distribution during validation.

## 4. Training Flow
- **Initialization**: Loads `Swin_V2_T_Weights.DEFAULT`.
- **Optimization**: AdamW + Cosine Annealing.
- **Transformer-Specific Regularization**: ViTs are notoriously prone to overfitting small datasets. Therefore, Model B utilizes a much stricter regularization profile than Model A: higher `weight_decay` (`0.05`) and higher `dropout` (`0.20`).
- **Exports**: Outputs `ModelB_SwinV2T_fold{k}_logits.npy` to participate in the late-fusion ensemble.
