# Model A: EfficientNetV2-S — DR Classification

> **Architecture:** Pure CNN baseline — EfficientNetV2-S backbone → Global Avg Pool → CORN ordinal head  
> **Input Resolution:** 512×512  
> **Parameters:** ~21M (all trainable)  
> **Spatial Paradigm:** Local feature extraction

---

## 📋 File Structure

```
model_A_efficientnet/
├── config.py                      ← Universal tri-env config (Kaggle/Colab/Local)
├── preprocessing.py               ← Ben Graham pipeline (shared)
├── dataset.py                     ← Dataset, CORN encoding, augmentations (APTOS only)
├── losses.py                      ← CORN + Focal hybrid loss (shared)
├── model.py                       ← EfficientNetV2-S architecture (with grad checkpointing)
├── metrics.py                     ← Evaluation metrics & plot functions (shared)
├── train.py                       ← Single-stage training loop (APTOS fine-tuning)
├── evaluate.py                    ← Evaluation + .npy export for ensemble
├── master_notebook.ipynb          ← 🚀 RUN THIS ON KAGGLE OR COLAB
└── README.md                      ← You are here
```

---

## ⚡ Execution Instructions (Universal)

This codebase automatically detects if it's running on **Kaggle** or **Google Colab** and routes dataset/output paths accordingly.

### Option 1: Google Colab (Recommended)
1. Upload this entire directory (`model_A_efficientnet/`) to your Google Drive at `MyDrive/DR_Thesis/model_A_efficientnet/`
2. Download the APTOS 2019 dataset and extract it to `MyDrive/DR_Thesis/datasets/aptos2019-blindness-detection/`
3. Open `master_notebook.ipynb` in Colab.
4. **Runtime** → **Change runtime type** → **T4 GPU** (or A100).
5. Run all cells. Checkpoints will automatically be saved back to your Google Drive.

### Option 2: Kaggle
1. Go to **kaggle.com → New Dataset**
2. Name it `dr-model-a-code`
3. Upload ALL `.py` files from this directory.
4. **Create a new Kaggle Notebook**.
5. Add Input Datasets:
   - APTOS 2019: `aptos2019-blindness-detection`
   - Your code: `dr-model-a-code`
6. Upload or copy the contents of `master_notebook.ipynb` into the notebook.
7. Set accelerator to **GPU T4 x2** and **Internet ON**.
8. Run all cells. 

### Resume After Session Timeout
This pipeline checkpoints after **every epoch**.
- **Colab:** Simply re-run the notebook. It will detect the `_state.pth` in your Google Drive and resume automatically.
- **Kaggle:** Save the outputs as a new dataset (`dr-model-a-checkpoints`), add it as an input to your next session, and run.

---

## 📊 Training Pipeline

```
ImageNet Pretrained Weights
         ↓
Fine-tune on APTOS 2019 (3,662 images, 5-fold CV)
  └─ 30 epochs, LR=1e-5, Early stopping (patience=7)
  └─ Anti-overfit: Weight decay 0.03, Dropout 0.15, Grad Checkpointing Enabled
```

Each fold saves:
- `ModelA_EfficientNetV2S_fold{k}_stage2_best.pth`
- `ModelA_EfficientNetV2S_fold{k}_stage2_state.pth`
- `ModelA_EfficientNetV2S_fold{k}_stage2_history.json`

---

## 📦 Output Files

### Required for Ensemble (Priority 1)
```
exports/
├── ModelA_EfficientNetV2S_fold{0-4}_logits.npy
├── ModelA_EfficientNetV2S_fold{0-4}_predictions.npy
├── ModelA_EfficientNetV2S_fold{0-4}_ground_truth.npy
└── ModelA_EfficientNetV2S_fold{0-4}_features.npy
```

### Visual Proofs (Priority 2)
```
plots/
├── ModelA_EfficientNetV2S_fold{k}_cm.pdf       ← Confusion matrices
├── ModelA_EfficientNetV2S_fold{k}_roc.pdf      ← ROC-AUC curves
├── ModelA_EfficientNetV2S_fold{k}_pr.pdf       ← Precision-Recall curves
├── ModelA_EfficientNetV2S_fold{k}_umap.pdf     ← UMAP latent space
├── ModelA_EfficientNetV2S_fold{k}_tsne.pdf     ← t-SNE latent space
└── ModelA_EfficientNetV2S_fold{k}_curves.pdf   ← Training curves
```

---

## ⏱️ Expected Runtime

| Stage | Estimated Time (T4) |
|---|---|
| Training (APTOS 5-fold) | ~1.5 - 2 hours |
| Evaluation | ~15 minutes |
| **Total** | **~2 hours** |
