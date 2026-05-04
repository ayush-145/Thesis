# Model D: RETFound — DR Classification

> **Architecture:** Pretrained ViT-L/16 (RETFound) + LoRA Adapters → CORN ordinal head  
> **Input Resolution:** 224×224  
> **Parameters:** ~304M (Only ~1M trainable via LoRA)  
> **Spatial Paradigm:** Domain-specific (Retinal MAE) Foundation Model

---

## 📋 File Structure

```
model_D_retfound/
├── config.py                      ← Universal tri-env config (Kaggle/Colab/Local)
├── preprocessing.py               ← Raw spatial preprocessing (disabled Ben Graham)
├── dataset.py                     ← Dataset, CORN encoding, augmentations (APTOS only)
├── losses.py                      ← CORN + Focal hybrid loss (shared)
├── model.py                       ← ViT-L with LoRA wrappers
├── metrics.py                     ← Evaluation metrics & plot functions (shared)
├── train.py                       ← Single-stage training loop (LoRA fine-tuning)
├── evaluate.py                    ← Evaluation + .npy export for ensemble
├── master_notebook.ipynb          ← 🚀 RUN THIS ON KAGGLE OR COLAB
└── README.md                      ← You are here
```

---

## ⚡ Execution Instructions (Universal)

This codebase automatically detects if it's running on **Kaggle** or **Google Colab** and routes dataset/output paths accordingly.

### Option 1: Google Colab (Recommended)
1. Upload this entire directory (`model_D_retfound/`) to your Google Drive at `MyDrive/DR_Thesis/model_D_retfound/`
2. Download the APTOS 2019 dataset and extract it to `MyDrive/DR_Thesis/datasets/aptos2019-blindness-detection/`
3. Download the RETFound weights `RETFound_cfp_weights.pth` and place it at `MyDrive/DR_Thesis/weights/RETFound_cfp_weights.pth`, OR ensure you have added your Hugging Face access token to Colab Secrets with the name `HF_TOKEN` to download it automatically.
4. Open `master_notebook.ipynb` in Colab.
5. **Runtime** → **Change runtime type** → **T4 GPU** (or A100).
6. Run all cells. Checkpoints will automatically be saved back to your Google Drive.

### Option 2: Kaggle
1. Go to **kaggle.com → New Dataset**
2. Name it `dr-model-d-code`
3. Upload ALL `.py` files from this directory.
4. **Create a new Kaggle Notebook**.
5. Add Input Datasets:
   - APTOS 2019: `aptos2019-blindness-detection`
   - RETFound weights: `retfound-weights` (ensure `RETFound_cfp_weights.pth` is inside), OR ensure you have added your Hugging Face access token to Kaggle Secrets with the name `HF_TOKEN` to download it automatically.
   - Your code: `dr-model-d-code`
6. Upload or copy the contents of `master_notebook.ipynb` into the notebook.
7. Set accelerator to **GPU T4 x2** and **Internet ON**. *(Note: The codebase intentionally utilizes only a single GPU to maintain stability with Gradient Checkpointing and LoRA. The second GPU will remain idle, which is expected).*
8. Run all cells. 

### Resume After Session Timeout
This pipeline checkpoints after **every epoch**.
- **Colab:** Simply re-run the notebook. It will detect the `_state.pth` in your Google Drive and resume automatically.
- **Kaggle:** Save the outputs as a new dataset (`dr-model-d-checkpoints`), add it as an input to your next session, and run.

---

## 📊 Training Pipeline

```
RETFound Pretrained Weights (ViT-L/16 frozen)
         ↓
Attach LoRA Adapters (Rank=8)
         ↓
Fine-tune LoRA on APTOS 2019 (3,662 images, 5-fold CV)
  └─ 30 epochs, LR=2e-4, Early stopping (patience=7)
  └─ Anti-overfit: Weight decay 0.03, Dropout 0.1, Grad Checkpointing Enabled
```

Each fold saves:
- `ModelD_RETFound_fold{k}_stage2_best.pth`
- `ModelD_RETFound_fold{k}_stage2_state.pth`
- `ModelD_RETFound_fold{k}_stage2_history.json`

---

## 📦 Output Files

### Required for Ensemble (Priority 1)
```
exports/
├── ModelD_RETFound_fold{0-4}_logits.npy
├── ModelD_RETFound_fold{0-4}_predictions.npy
├── ModelD_RETFound_fold{0-4}_ground_truth.npy
└── ModelD_RETFound_fold{0-4}_features.npy
```

### Visual Proofs (Priority 2)
```
plots/
├── ModelD_RETFound_fold{k}_cm.pdf       ← Confusion matrices
├── ModelD_RETFound_fold{k}_roc.pdf      ← ROC-AUC curves
├── ModelD_RETFound_fold{k}_pr.pdf       ← Precision-Recall curves
├── ModelD_RETFound_fold{k}_umap.pdf     ← UMAP latent space
├── ModelD_RETFound_fold{k}_tsne.pdf     ← t-SNE latent space
└── ModelD_RETFound_fold{k}_curves.pdf   ← Training curves
```

---

## ⏱️ Expected Runtime

| Stage | Estimated Time (T4) |
|---|---|
| Training (APTOS 5-fold) | ~3 hours |
| Evaluation | ~20 minutes |
| **Total** | **~3.5 hours** |
