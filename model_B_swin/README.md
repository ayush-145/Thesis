# Model B: Swin-V2-Tiny — DR Classification

> **Architecture:** Global reasoning baseline — Shifted Window ViT (Swin-V2-T) → CORN ordinal head  
> **Input Resolution:** 512×512  
> **Parameters:** ~28M (all trainable)  
> **Spatial Paradigm:** Global self-attention

---

## 📋 File Structure

```
model_B_swin/
├── config.py                      ← Universal tri-env config (Kaggle/Colab/Local)
├── preprocessing.py               ← Ben Graham pipeline (shared)
├── dataset.py                     ← Dataset, CORN encoding, augmentations (APTOS only)
├── losses.py                      ← CORN + Focal hybrid loss (shared)
├── model.py                       ← Swin-V2-T architecture (with grad checkpointing)
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
1. Upload this entire directory (`model_B_swin/`) to your Google Drive at `MyDrive/DR_Thesis/model_B_swin/`
2. Open `master_notebook.ipynb` in Colab.
3. **Runtime** → **Change runtime type** → **T4 GPU** (or A100).
4. Run all cells. 
   - *Note:* The notebook will automatically download and extract the APTOS 2019 dataset to your Google Drive if it isn't already there. It will prompt you to paste your Kaggle API Token (starts with `KGAT_`) directly in the notebook cell.
5. Checkpoints will automatically be saved back to your Google Drive.

### Option 2: Kaggle
1. Go to **kaggle.com → New Dataset**
2. Name it `dr-model-b-code`
3. Upload ALL `.py` files from this directory.
4. **Create a new Kaggle Notebook**.
5. Add Input Datasets:
   - APTOS 2019: `aptos2019-blindness-detection`
   - Your code: `dr-model-b-code`
6. Upload or copy the contents of `master_notebook.ipynb` into the notebook.
7. Set accelerator to **GPU T4 x2** and **Internet ON**.
8. Run all cells. 

### Resume After Session Timeout
This pipeline checkpoints after **every epoch**.
- **Colab:** Simply re-run the notebook. It will detect the `_state.pth` in your Google Drive and resume automatically.
- **Kaggle:** Save the outputs as a new dataset (`dr-model-b-checkpoints`), add it as an input to your next session, and run.

---

## 📊 Training Pipeline

```
ImageNet Pretrained Weights
         ↓
Fine-tune on APTOS 2019 (3,662 images, 5-fold CV)
  └─ 30 epochs, LR=5e-6, Early stopping (patience=7)
  └─ Anti-overfit: Weight decay 0.05, Dropout 0.2, Grad Checkpointing Enabled
```

Each fold saves:
- `ModelB_SwinV2T_fold{k}_stage2_best.pth`
- `ModelB_SwinV2T_fold{k}_stage2_state.pth`
- `ModelB_SwinV2T_fold{k}_stage2_history.json`

---

## 📦 Output Files

### Required for Ensemble (Priority 1)
```
exports/
├── ModelB_SwinV2T_fold{0-4}_logits.npy
├── ModelB_SwinV2T_fold{0-4}_predictions.npy
├── ModelB_SwinV2T_fold{0-4}_ground_truth.npy
└── ModelB_SwinV2T_fold{0-4}_features.npy
```

### Visual Proofs (Priority 2)
```
plots/
├── ModelB_SwinV2T_fold{k}_cm.pdf       ← Confusion matrices
├── ModelB_SwinV2T_fold{k}_roc.pdf      ← ROC-AUC curves
├── ModelB_SwinV2T_fold{k}_pr.pdf       ← Precision-Recall curves
├── ModelB_SwinV2T_fold{k}_umap.pdf     ← UMAP latent space
├── ModelB_SwinV2T_fold{k}_tsne.pdf     ← t-SNE latent space
└── ModelB_SwinV2T_fold{k}_curves.pdf   ← Training curves
```

---

## ⏱️ Expected Runtime

| Stage | Estimated Time (T4) |
|---|---|
| Training (APTOS 5-fold) | ~2 - 2.5 hours |
| Evaluation | ~20 minutes |
| **Total** | **~2.5 - 3 hours** |
