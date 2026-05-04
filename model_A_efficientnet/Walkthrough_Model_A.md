# Model A: EfficientNetV2-S Pipeline Walkthrough

## 1. Architectural Overview
Model A serves as the **Pure CNN baseline** for the Diabetic Retinopathy (DR) pipeline. 
- **Backbone**: EfficientNetV2-S (pretrained on ImageNet)
- **Input Resolution**: 512 × 512
- **Feature Extraction**: Localized spatial feature extraction utilizing convolutions.
- **Classification Head**: Global Average Pooling followed by a CORN (Conditional Ordinal Regression for Neural Networks) classification head.
- **VRAM Hardening**: Implements gradient checkpointing across the `features` sequential block to allow training on Kaggle/Colab T4 GPUs (16GB VRAM) without OOM errors.

## 2. Preprocessing & Data Augmentation
All images go through a rigorous, clinically-inspired pipeline before reaching the model:
1. **Ben Graham Spatial Preprocessing** (`preprocessing.py`):
   - **Auto-cropping**: Detects the circular fundus contour and strips away uninformative black borders.
   - **Gaussian Blur Subtraction**: Subtracts a blurred version of the image from the original (`img - blur + 128`) to normalize uneven lighting/exposure across different clinic cameras.
   - **Green Channel CLAHE**: Applies Contrast Limited Adaptive Histogram Equalization to the green color channel, which holds the highest contrast for vascular structures and red lesions (microaneurysms).
2. **Albumentations Augmentation** (`dataset.py`):
   - `RandomResizedCrop` (Scale/Zoom invariance)
   - `HorizontalFlip` & `VerticalFlip` (Anatomical symmetry)
   - `RandomBrightnessContrast` & `HueSaturationValue` (Camera variation robustness)
   - `ImageNet Normalization`

## 3. Handling Class Imbalance
The APTOS 2019 dataset is heavily imbalanced (e.g., 1805 'No DR' vs. 193 'Severe DR'). This pipeline handles it using two major techniques:
1. **Hybrid Ordinal-Focal Loss** (`losses.py`):
   - Rather than treating severity grades as independent classes, the model predicts *ordinal ranks* (e.g. "Is it > Grade 1?", "Is it > Grade 2?"). 
   - This CORN loss is hybridized with an **Ordinal Focal Loss** (`gamma=2.0`, `alpha=0.25`). The Focal Loss dynamically scales the loss based on prediction confidence, heavily down-weighting the easily classified majority classes ("No DR") and forcing the network to focus its gradients on the hard-to-classify minority cases ("Severe", "Proliferative").
2. **Stratified K-Fold** (`train.py`): 
   - Ensures that the severe and proliferative cases are evenly distributed across all 5 validation folds.

## 4. Training Flow
- **Initialization**: Loads standard `EfficientNet_V2_S_Weights.DEFAULT`.
- **Optimization**: AdamW optimizer with Cosine Annealing + Warmup (3 epochs warmup, 30 epochs total). 
- **Regularization**: High weight decay (`0.03`) and dropout (`0.15`) are utilized to prevent overfitting on the small 3.6K image dataset.
- **Exports**: Saves best weights, state checkpoints, and exports `.npy` logits for the ensemble.
