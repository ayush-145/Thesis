# Model D: RETFound Foundation Pipeline Walkthrough

## 1. Architectural Overview
Model D leverages a domain-specific **Medical Vision Foundation Model**.
- **Backbone**: ViT-L/16 initialized with **RETFound** weights (pretrained via Masked Autoencoding on 1.6 million unlabeled retinal images).
- **Input Resolution**: 224 × 224 (Fixed by RETFound's pretrained positional embeddings).
- **Parameter-Efficient Fine-Tuning (PEFT)**: Because ViT-L is massive (~304M parameters), the backbone is strictly frozen. Trainable **LoRA (Low-Rank Adaptation)** matrices (`rank=8`, `alpha=16`) are injected into the Query/Value attention projections.
- **Classification Head**: Global average pooling over the `[CLS]` token and sequence tokens → CORN ordinal head.

## 2. Preprocessing & Data Augmentation
1. **Raw Spatial Preprocessing** (`preprocessing.py`):
   - RETFound was pretrained on raw images. Applying the Ben Graham pipeline (Blur Subtraction, Green CLAHE) used in Models A and B would cause a massive domain shift, rendering the frozen MAE features useless. Therefore, images are *only* auto-cropped and resized to preserve the original color distribution.
2. **Albumentations Augmentation** (`dataset.py`):
   - Uses identical augmentations to Models A-C, but enforces the stricter `224x224` resolution required by the ViT-L architecture.

## 3. Handling Class Imbalance
1. **Hybrid Ordinal-Focal Loss** (`losses.py`):
   - Similar to the other models, the focal loss dynamically scales gradients to focus on the hardest minority examples.
2. **Stratified K-Fold** (`train.py`): 
   - Uses the exact same 5-fold stratification.

## 4. Training Flow
- **Initialization**: Loads `RETFound_cfp_weights.pth` directly from disk. LoRA matrices are initialized randomly.
- **Optimization**: AdamW + Cosine Annealing. The learning rate is set to `2e-4`, which is higher than typical full fine-tuning, to ensure the randomly initialized LoRA adapters and classification head can converge properly.
- **Freezing Strategy**: `requires_grad = False` for all layers except LoRA layers and the final CORN classification head. This reduces the trainable parameters to ~1 Million, completely eliminating VRAM bottlenecks.
- **Exports**: Generates `ModelD_RETFound_fold{k}_logits.npy` for the ensemble. Because RETFound understands retinal topology inherently, it usually contributes the most orthogonal information to the final ensemble.
