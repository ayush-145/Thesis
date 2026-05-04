"""
evaluate.py — Evaluation & Ensemble Export for ModelA_EfficientNetV2S
=====================================================================
- Loads best checkpoints for each fold
- Generates all visual proofs (confusion matrix, ROC, PR, UMAP, t-SNE)
- Exports standardized .npy files for the 5th Ensemble notebook
"""

import os
import json
import torch
import numpy as np
from torch.amp import autocast
from typing import Dict

from config import CFG, MODEL_NAME
from model import build_model, get_img_size_for_model
from dataset import (
    decode_corn_prediction, load_aptos_dataframe,
    get_stratified_kfold_splits, build_dataloaders,
)
from metrics import (
    compute_all_metrics, plot_confusion_matrix, plot_roc_curves,
    plot_pr_curves, plot_latent_space, plot_training_curves,
)


# ──────────────────────────────────────────────
# Feature Extraction
# ──────────────────────────────────────────────

@torch.no_grad()
def extract_features_and_predictions(model, loader, device, use_amp=True):
    """Extract penultimate features, CORN logits, predictions, and labels."""
    model.eval()
    all_features, all_logits, all_preds, all_labels = [], [], [], []

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["grade"]
        with autocast(device_type="cuda", enabled=use_amp):
            logits = model(images)
            features = model._features_cache
            if features is None:
                features = model.get_features(images)
        preds = decode_corn_prediction(logits)
        all_features.append(features.cpu().numpy())
        all_logits.append(logits.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

    return {
        "features": np.concatenate(all_features, axis=0),
        "logits": np.concatenate(all_logits, axis=0),
        "preds": np.array(all_preds),
        "labels": np.array(all_labels),
    }


# ──────────────────────────────────────────────
# Export .npy for Ensemble
# ──────────────────────────────────────────────

def export_ensemble_artifacts(data: Dict, fold: int):
    """Save standardized .npy files for the Ensemble notebook."""
    exports_dir = CFG.paths.exports_dir
    model_name = MODEL_NAME

    files = {
        f"{model_name}_fold{fold}_logits.npy": data["logits"],
        f"{model_name}_fold{fold}_predictions.npy": data["preds"],
        f"{model_name}_fold{fold}_ground_truth.npy": data["labels"],
        f"{model_name}_fold{fold}_features.npy": data["features"],
    }

    for filename, array in files.items():
        path = os.path.join(exports_dir, filename)
        np.save(path, array)
        print(f"  Exported: {path} — shape={array.shape}")


# ──────────────────────────────────────────────
# Single-Fold Evaluation
# ──────────────────────────────────────────────

def evaluate_fold(fold: int, val_loader, device: torch.device):
    """Full evaluation for one fold: metrics, plots, .npy export."""
    model_name = MODEL_NAME
    print(f"\n  Evaluating {model_name} — Fold {fold+1}...")

    ckpt_path = os.path.join(
        CFG.paths.checkpoint_dir, f"{model_name}_fold{fold}_stage2_best.pth")
    if not os.path.exists(ckpt_path):
        import glob
        cands = glob.glob(f"/kaggle/input/**/{os.path.basename(ckpt_path)}", recursive=True)
        if cands:
            ckpt_path = cands[0]
        else:
            print(f"  ✖ Checkpoint not found: {ckpt_path}")
            return None

    model = build_model(pretrained=False)
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=False))
    model = model.to(device)

    data = extract_features_and_predictions(model, val_loader, device)
    metrics = compute_all_metrics(data["labels"], data["preds"])
    print(f"  QWK={metrics['qwk']:.4f} | Acc={metrics['accuracy']:.4f}")

    # ── Generate plots ──
    prefix = f"{model_name}_fold{fold+1}"
    plot_dir = CFG.paths.plots_dir
    fmt = CFG.eval.fig_format

    plot_confusion_matrix(data["labels"], data["preds"],
        title=f"{model_name} — Fold {fold+1} Confusion Matrix",
        save_path=os.path.join(plot_dir, f"{prefix}_cm.{fmt}"))
    plot_roc_curves(data["labels"], data["logits"],
        title=f"{model_name} — Fold {fold+1} ROC Curves",
        save_path=os.path.join(plot_dir, f"{prefix}_roc.{fmt}"))
    plot_pr_curves(data["labels"], data["logits"],
        title=f"{model_name} — Fold {fold+1} PR Curves",
        save_path=os.path.join(plot_dir, f"{prefix}_pr.{fmt}"))
    plot_latent_space(data["features"], data["labels"],
        title=f"{model_name} — Fold {fold+1} UMAP",
        save_path=os.path.join(plot_dir, f"{prefix}_umap.{fmt}"), method="umap")
    plot_latent_space(data["features"], data["labels"],
        title=f"{model_name} — Fold {fold+1} t-SNE",
        save_path=os.path.join(plot_dir, f"{prefix}_tsne.{fmt}"), method="tsne")

    # Training curves
    history_path = os.path.join(
        CFG.paths.checkpoint_dir, f"{model_name}_fold{fold}_stage2_history.json")
    if os.path.exists(history_path):
        with open(history_path) as f:
            history = json.load(f)
        plot_training_curves(history,
            title=f"{model_name} — Fold {fold+1} Training Curves",
            save_path=os.path.join(plot_dir, f"{prefix}_curves.{fmt}"))

    # ── Export .npy for ensemble ──
    export_ensemble_artifacts(data, fold)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {"metrics": metrics, "data": data}


# ──────────────────────────────────────────────
# Full Evaluation Pipeline
# ──────────────────────────────────────────────

def run_single_model_evaluation(device: torch.device):
    """Evaluate ModelA across all 5 folds and export ensemble artifacts."""
    model_name = MODEL_NAME
    img_size = get_img_size_for_model()

    print("\n" + "#" * 60)
    print(f"  EVALUATION: {model_name}")
    print("#" * 60)

    aptos_df = load_aptos_dataframe()
    folds = get_stratified_kfold_splits(aptos_df)
    all_metrics = []

    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        _, val_loader = build_dataloaders(
            aptos_df, train_idx, val_idx, img_size, label_col="diagnosis")
        result = evaluate_fold(fold_idx, val_loader, device)
        if result:
            all_metrics.append(result["metrics"])

    if all_metrics:
        qwks = [m["qwk"] for m in all_metrics]
        print(f"\n  {model_name} — CV QWK: {np.mean(qwks):.4f} ± {np.std(qwks):.4f}")

    # Save evaluation summary
    eval_summary = {
        "model": model_name,
        "folds": [{k: v for k, v in m.items() if k != "report"} for m in all_metrics],
        "mean_qwk": float(np.mean([m["qwk"] for m in all_metrics])) if all_metrics else 0,
        "std_qwk": float(np.std([m["qwk"] for m in all_metrics])) if all_metrics else 0,
    }
    eval_path = os.path.join(CFG.paths.logs_dir, "evaluation_summary.json")
    with open(eval_path, "w") as f:
        json.dump(eval_summary, f, indent=2)
    print(f"  Evaluation summary saved to: {eval_path}")

    return all_metrics


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_single_model_evaluation(device)
