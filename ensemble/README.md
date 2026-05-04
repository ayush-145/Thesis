# Ensemble — Heterogeneous Late Fusion

> Run this **AFTER** all 3 model directories have completed on their respective Kaggle or Colab accounts.

---

## 📋 What This Does

1. Loads `.npy` exports from all 3 models (logits, predictions, ground_truth)
2. Converts CORN ordinal logits → K-class probabilities
3. Performs weighted average fusion with grid-search weight optimization
4. Reports per-fold and mean ensemble QWK
5. Generates confusion matrices for each fold
6. Computes pairwise ensemble analysis

---

## 📦 Prerequisites

### Step 1: Download Exports from All 3 Runs

From each model's Kaggle or Colab output, collect the `exports/` directory. You will have files like:

```
exports/
├── ModelA_EfficientNetV2S_fold{0-4}_{logits,predictions,ground_truth,features}.npy
├── ModelB_SwinV2T_fold{0-4}_{logits,predictions,ground_truth,features}.npy
└── ModelD_RETFound_fold{0-4}_{logits,predictions,ground_truth,features}.npy
```

### Step 2: Organize Files

Place ALL `.npy` files into a single `exports/` directory:

```
ensemble/
├── exports/                        ← Put ALL .npy files here (60-80 files total)
├── ensemble.py
├── ensemble_notebook.ipynb         ← 🚀 RUN THIS
└── README.md
```

---

## ⚡ Running (Universal)

This codebase automatically detects your environment.

### Option A: Google Colab (Recommended)
1. Upload the `ensemble/` directory to `MyDrive/DR_Thesis/ensemble/`
2. Ensure your `exports/` folder has all the `.npy` files.
3. Open `ensemble_notebook.ipynb` in Colab.
4. Run all cells (No GPU needed).

### Option B: Kaggle

Kaggle is an excellent platform for running this ensemble because the script is designed to automatically detect your Kaggle environment and find the necessary datasets.

**Step-by-step Kaggle Execution:**

1. **Create the Exports Dataset**: 
   - Download the `exports/` directories from Folds 1-5 for Model A, Model B, and Model D.
   - You should have 60 `.npy` files total (3 models × 5 folds × 4 export files).
   - Go to Kaggle -> Datasets -> New Dataset. 
   - You can either drag-and-drop the folder directly (Kaggle will handle the zipping/unzipping) OR you can zip the `.npy` files locally and upload the `.zip`. 
   - **Crucial**: Name this dataset exactly `dr-all-exports` (or ensure the folder inside is named that). The `ensemble.py` script automatically searches `/kaggle/input/**/dr-all-exports` for the files!
   - **Expected Structure** (once attached to a notebook):
     ```
     /kaggle/input/dr-all-exports/
     ├── ModelA_EfficientNetV2S_fold0_logits.npy
     ├── ModelA_EfficientNetV2S_fold0_predictions.npy
     ├── ... (58 more files) ...
     └── ModelD_RETFound_fold4_features.npy
     ```

2. **Create the Code Dataset**:
   - Go to Kaggle -> Datasets -> New Dataset.
   - Drag-and-drop the local `ensemble/` directory (containing `ensemble.py`) or upload it as a `.zip`. Name the dataset `dr-ensemble-code`.
   - **Expected Structure** (once attached to a notebook):
     ```
     /kaggle/input/dr-ensemble-code/
     ├── ensemble.py
     ├── ensemble_notebook.ipynb
     └── README.md
     ```

3. **Set Up the Notebook**:
   - Create a new Kaggle Notebook.
   - Click **Add Input** on the right sidebar and attach BOTH of the datasets you just created (`dr-all-exports` and `dr-ensemble-code`).
   - Open `ensemble_notebook.ipynb` on your local machine, copy its contents, and paste it into your Kaggle Notebook.
   - Run the notebook. You do **not** need a GPU enabled for this step; a standard CPU kernel will execute this in under a minute!

4. **Retrieve Results**:
   - When the notebook finishes, it will generate an `ensemble_outputs/` folder in `/kaggle/working/`. 
   - You can download the `.pdf` charts and the `.json` results directly from the right-hand output pane.

### Option C: Local Machine
```bash
cd ensemble/
python ensemble.py --exports ./exports --output ./ensemble_outputs
```

---

## 📊 Output

```
ensemble_outputs/
├── ensemble_fold{0-4}_cm.pdf       ← Confusion matrices
├── comparative_qwk_bar.pdf         ← Bar chart comparing models vs ensemble
├── comparative_qwk_box.pdf         ← Box plot showing stability across folds
├── model_correlation_heatmap.pdf   ← Spearman correlation of model predictions
└── ensemble_results.json           ← Full results with weights and QWKs
```

---

## 🔗 Fold Alignment

All 3 models use `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` on the APTOS dataset. This guarantees:
- Fold 0 across all 3 models contains the **exact same** validation samples
- Ground truth labels are perfectly aligned for fusion
- The ensemble script verifies this with an assertion check

---

## ⏱️ Runtime

< 1 minute (no GPU needed — just numpy operations on pre-computed logits)
