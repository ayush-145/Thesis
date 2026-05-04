"""
ensemble.py — Heterogeneous Late Fusion Ensemble
==================================================
Loads .npy exports from all 4 models, converts CORN logits to probabilities,
and performs weighted fusion with grid-search optimization on validation QWK.

Supports running on Kaggle, Google Colab, or local machine.
Run this AFTER all 4 models have completed training and evaluation.
"""

import os
import sys
import numpy as np
from sklearn.metrics import cohen_kappa_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional


# ──────────────────────────────────────────────
# Environment Detection & Configuration
# ──────────────────────────────────────────────

IS_KAGGLE = "KAGGLE_KERNEL_RUN_TYPE" in os.environ
IS_COLAB = not IS_KAGGLE and (
    "COLAB_RELEASE_TAG" in os.environ
    or "google.colab" in sys.modules
    or os.path.exists("/content/drive")
)
COLAB_DRIVE_BASE = "/content/drive/MyDrive/DR_Thesis"

MODEL_NAMES = [
    "ModelA_EfficientNetV2S",
    "ModelB_SwinV2T",
    "ModelD_RETFound",
]

NUM_FOLDS = 5
NUM_CLASSES = 5
CLASS_NAMES = ("No DR", "Mild", "Moderate", "Severe", "Proliferative")


def _find_kaggle_dataset(name: str, fallback: str = "") -> str:
    """Auto-discover a Kaggle input dataset by name."""
    import glob
    candidates = glob.glob(f"/kaggle/input/**/{name}", recursive=True)
    if candidates:
        return min(candidates, key=len)
    return fallback


def _resolve_exports_dir() -> str:
    """Resolve exports directory based on runtime environment."""
    if IS_KAGGLE:
        # Try to find a dataset with all exports
        found = _find_kaggle_dataset("dr-all-exports")
        if found:
            return found
        # Fallback: look in /kaggle/input for any directory with .npy files
        import glob
        npy_files = glob.glob("/kaggle/input/**/*.npy", recursive=True)
        if npy_files:
            return os.path.dirname(npy_files[0])
        return "/kaggle/input/dr-all-exports"
    elif IS_COLAB:
        return f"{COLAB_DRIVE_BASE}/exports"
    else:
        return "./exports"


def _resolve_output_dir() -> str:
    """Resolve output directory based on runtime environment."""
    if IS_KAGGLE:
        return "/kaggle/working/ensemble_outputs"
    elif IS_COLAB:
        return f"{COLAB_DRIVE_BASE}/outputs/ensemble"
    else:
        return "./ensemble_outputs"


EXPORTS_DIR = _resolve_exports_dir()
OUTPUT_DIR = _resolve_output_dir()


# ──────────────────────────────────────────────
# CORN → Probability Conversion
# ──────────────────────────────────────────────

def corn_logits_to_class_probs(logits: np.ndarray) -> np.ndarray:
    """Convert CORN ordinal logits (K-1) to K-class probabilities."""
    sigm = 1.0 / (1.0 + np.exp(-logits))
    B, K_minus_1 = sigm.shape
    K = K_minus_1 + 1
    probs = np.zeros((B, K), dtype=np.float32)
    probs[:, 0] = 1.0 - sigm[:, 0]
    for k in range(1, K_minus_1):
        probs[:, k] = sigm[:, k - 1] - sigm[:, k]
    probs[:, K - 1] = sigm[:, K_minus_1 - 1]
    probs = np.clip(probs, 0, 1)
    row_sums = probs.sum(axis=1, keepdims=True)
    return probs / np.where(row_sums > 0, row_sums, 1.0)


# ──────────────────────────────────────────────
# Load Exports
# ──────────────────────────────────────────────

def load_model_fold_data(model_name: str, fold: int, exports_dir: str) -> Optional[Dict]:
    """Load .npy files for a specific model and fold."""
    files = {
        "logits": f"{model_name}_fold{fold}_logits.npy",
        "predictions": f"{model_name}_fold{fold}_predictions.npy",
        "ground_truth": f"{model_name}_fold{fold}_ground_truth.npy",
    }
    data = {}
    for key, filename in files.items():
        path = os.path.join(exports_dir, filename)
        if not os.path.exists(path):
            print(f"  ✖ Missing: {path}")
            return None
        data[key] = np.load(path)

    # Optional: features
    feat_path = os.path.join(exports_dir, f"{model_name}_fold{fold}_features.npy")
    if os.path.exists(feat_path):
        data["features"] = np.load(feat_path)

    return data


def load_all_exports(exports_dir: str) -> Dict:
    """Load all model exports organized by fold."""
    all_data = {}
    for model_name in MODEL_NAMES:
        all_data[model_name] = {}
        for fold in range(NUM_FOLDS):
            data = load_model_fold_data(model_name, fold, exports_dir)
            if data:
                all_data[model_name][fold] = data
                print(f"  ✓ {model_name} fold {fold}: "
                      f"logits={data['logits'].shape}, labels={data['ground_truth'].shape}")
    return all_data


# ──────────────────────────────────────────────
# Fusion Strategies
# ──────────────────────────────────────────────

def weighted_average_fusion(probs_list: List[np.ndarray],
                            weights: Optional[List[float]] = None) -> np.ndarray:
    """Weighted average of multiple probability distributions."""
    if weights is None:
        weights = [1.0 / len(probs_list)] * len(probs_list)
    fused = sum(w * p for w, p in zip(weights, probs_list))
    return fused / fused.sum(axis=1, keepdims=True)


def optimize_fusion_weights(probs_list: List[np.ndarray], labels: np.ndarray,
                             n_models: int, steps: int = 11) -> Tuple[List[float], float]:
    """Grid search over fusion weights to maximize QWK (for 2-4 models)."""
    if n_models == 2:
        best_w, best_qwk = [0.5, 0.5], -1.0
        for w1 in np.linspace(0, 1, steps):
            w = [w1, 1 - w1]
            fused = weighted_average_fusion(probs_list, w)
            preds = np.argmax(fused, axis=1)
            qwk = cohen_kappa_score(labels, preds, weights="quadratic")
            if qwk > best_qwk:
                best_qwk = qwk
                best_w = w
        return best_w, best_qwk

    # For 3+ models: use coarser grid + iterative refinement
    best_w = [1.0 / n_models] * n_models
    best_qwk = -1.0
    # Simple: try equal weights first, then optimize pairs
    fused = weighted_average_fusion(probs_list, best_w)
    preds = np.argmax(fused, axis=1)
    best_qwk = cohen_kappa_score(labels, preds, weights="quadratic")

    # Pair-wise optimization
    for i in range(n_models):
        for j in range(i + 1, n_models):
            for w_i in np.linspace(0, 1, steps):
                weights = best_w.copy()
                remaining = 1.0 - w_i
                weights[i] = w_i
                denom = sum(best_w[k] for k in range(n_models) if k != i)
                for k in range(n_models):
                    if k != i:
                        weights[k] = remaining * (best_w[k] / max(denom, 1e-8))
                fused = weighted_average_fusion(probs_list, weights)
                preds = np.argmax(fused, axis=1)
                qwk = cohen_kappa_score(labels, preds, weights="quadratic")
                if qwk > best_qwk:
                    best_qwk = qwk
                    best_w = weights.copy()

    return best_w, best_qwk


# ──────────────────────────────────────────────
# Visualization
# ──────────────────────────────────────────────

def plot_ensemble_confusion_matrix(labels, preds, title, save_path):
    cm = confusion_matrix(labels, preds, labels=list(range(NUM_CLASSES)))
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    cm_norm = np.nan_to_num(cm_norm)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                ax=ax, vmin=0, vmax=1, linewidths=0.5)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title)
    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"  Saved: {save_path}")
    plt.close(fig)

def generate_comparative_charts(fold_results, available_models, output_dir):
    """Generate comparative charts for the thesis."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np

    models_plus_ensemble = available_models + ["Ensemble"]
    
    # 1. Bar Chart with Error Bars (Mean + Std QWK)
    mean_qwks = []
    std_qwks = []
    for m in models_plus_ensemble:
        if m == "Ensemble":
            scores = [r["ensemble_qwk"] for r in fold_results]
        else:
            scores = [r["individual_qwks"][m] for r in fold_results]
        mean_qwks.append(np.mean(scores))
        std_qwks.append(np.std(scores))
        
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(models_plus_ensemble, mean_qwks, yerr=std_qwks, capsize=5, 
                  color=['#4C72B0', '#55A868', '#C44E52', '#8172B3'][:len(models_plus_ensemble)], alpha=0.8)
    
    ax.set_ylabel('Mean QWK across 5 Folds')
    ax.set_title('Model Performance Comparison (Quadratic Weighted Kappa)')
    ax.set_ylim(min(mean_qwks) - 0.05, min(1.0, max(mean_qwks) + 0.05))
    
    # Add values on top of bars
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + 0.005, f'{yval:.4f}', ha='center', va='bottom', fontweight='bold')
        
    plt.tight_layout()
    bar_path = os.path.join(output_dir, "comparative_qwk_bar.pdf")
    fig.savefig(bar_path, dpi=300, bbox_inches="tight")
    print(f"  Saved: {bar_path}")
    plt.close(fig)

    # 2. Box Plot (QWK Distribution across folds)
    data = []
    for m in models_plus_ensemble:
        if m == "Ensemble":
            scores = [r["ensemble_qwk"] for r in fold_results]
        else:
            scores = [r["individual_qwks"][m] for r in fold_results]
        data.append(scores)
        
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=data, ax=ax, palette=['#4C72B0', '#55A868', '#C44E52', '#8172B3'][:len(models_plus_ensemble)])
    ax.set_xticklabels(models_plus_ensemble)
    ax.set_ylabel('Quadratic Weighted Kappa (QWK)')
    ax.set_title('Stability Analysis: QWK Distribution Across 5 Folds')
    
    plt.tight_layout()
    box_path = os.path.join(output_dir, "comparative_qwk_box.pdf")
    fig.savefig(box_path, dpi=300, bbox_inches="tight")
    print(f"  Saved: {box_path}")
    plt.close(fig)

def plot_correlation_heatmap(all_data, available_models, output_dir):
    """Plot correlation between models based on their predictions for Fold 0."""
    import pandas as pd
    fold = 0
    if not all(fold in all_data.get(m, {}) for m in available_models):
        return

    preds = {m: all_data[m][fold]["predictions"] for m in available_models}
    df = pd.DataFrame(preds)
    corr = df.corr(method="spearman") # spearman since ordinal

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".3f", cmap="coolwarm", vmin=0.5, vmax=1.0, ax=ax)
    ax.set_title("Model Prediction Correlation (Spearman) - Fold 0")
    
    plt.tight_layout()
    corr_path = os.path.join(output_dir, "model_correlation_heatmap.pdf")
    fig.savefig(corr_path, dpi=300, bbox_inches="tight")
    print(f"  Saved: {corr_path}")
    plt.close(fig)


# ──────────────────────────────────────────────
# Main Ensemble Pipeline
# ──────────────────────────────────────────────

def run_ensemble(exports_dir: str = EXPORTS_DIR, output_dir: str = OUTPUT_DIR):
    """Full ensemble pipeline: load exports → fuse → optimize → report."""
    os.makedirs(output_dir, exist_ok=True)

    env = "Kaggle" if IS_KAGGLE else "Colab" if IS_COLAB else "Local"
    print("=" * 60)
    print(f"  HETEROGENEOUS LATE FUSION ENSEMBLE [{env}]")
    print("  Models:", ", ".join(MODEL_NAMES))
    print(f"  Exports: {exports_dir}")
    print("=" * 60)

    # Load all exports
    print("\n  Loading exports...")
    all_data = load_all_exports(exports_dir)

    # Check which models have complete data
    available_models = [m for m in MODEL_NAMES if len(all_data.get(m, {})) == NUM_FOLDS]
    print(f"\n  Available models with all {NUM_FOLDS} folds: {available_models}")

    if len(available_models) < 2:
        print("  ✖ Need at least 2 models for ensemble. Exiting.")
        return

    # Per-fold ensemble
    fold_results = []
    for fold in range(NUM_FOLDS):
        print(f"\n  ── Fold {fold + 1}/{NUM_FOLDS} ──")

        # Verify label alignment
        labels = all_data[available_models[0]][fold]["ground_truth"]
        for m in available_models[1:]:
            assert np.array_equal(labels, all_data[m][fold]["ground_truth"]), \
                f"Label mismatch: {available_models[0]} vs {m} at fold {fold}"

        # Convert logits → probabilities
        probs_list = []
        for m in available_models:
            logits = all_data[m][fold]["logits"]
            probs = corn_logits_to_class_probs(logits)
            probs_list.append(probs)

            # Individual model QWK
            preds = all_data[m][fold]["predictions"]
            qwk = cohen_kappa_score(labels, preds, weights="quadratic")
            print(f"    {m}: QWK={qwk:.4f}")

        # Optimize fusion weights
        best_weights, best_qwk = optimize_fusion_weights(
            probs_list, labels, len(available_models), steps=21)

        print(f"    Ensemble QWK: {best_qwk:.4f}")
        print(f"    Weights: {dict(zip(available_models, [f'{w:.3f}' for w in best_weights]))}")

        # Generate ensemble predictions
        fused_probs = weighted_average_fusion(probs_list, best_weights)
        fused_preds = np.argmax(fused_probs, axis=1)

        # Confusion matrix
        plot_ensemble_confusion_matrix(
            labels, fused_preds,
            title=f"Ensemble — Fold {fold + 1}",
            save_path=os.path.join(output_dir, f"ensemble_fold{fold}_cm.pdf"))

        fold_results.append({
            "fold": fold,
            "weights": dict(zip(available_models, best_weights)),
            "ensemble_qwk": best_qwk,
            "individual_qwks": {
                m: cohen_kappa_score(labels, all_data[m][fold]["predictions"], weights="quadratic")
                for m in available_models
            },
        })

    # Summary
    print(f"\n{'='*60}")
    print(f"  ENSEMBLE SUMMARY")
    print(f"{'='*60}")

    header = f"  {'Fold':<6}"
    for m in available_models:
        short = m.replace("Model", "").replace("_", "")[:12]
        header += f" {short:<14}"
    header += f" {'Ensemble':<12}"
    print(header)
    print(f"  {'-'*len(header)}")

    for r in fold_results:
        line = f"  {r['fold']+1:<6}"
        for m in available_models:
            line += f" {r['individual_qwks'][m]:<14.4f}"
        line += f" {r['ensemble_qwk']:<12.4f}"
        print(line)

    mean_ens = np.mean([r["ensemble_qwk"] for r in fold_results])
    std_ens = np.std([r["ensemble_qwk"] for r in fold_results])
    print(f"\n  Mean Ensemble QWK: {mean_ens:.4f} ± {std_ens:.4f}")

    for m in available_models:
        mean_m = np.mean([r["individual_qwks"][m] for r in fold_results])
        print(f"  Mean {m}: {mean_m:.4f}")

    best_individual = max(
        np.mean([r["individual_qwks"][m] for r in fold_results])
        for m in available_models
    )
    print(f"  Ensemble gain: +{mean_ens - best_individual:.4f}")

    # Save results
    import json
    results_path = os.path.join(output_dir, "ensemble_results.json")
    with open(results_path, "w") as f:
        json.dump({"fold_results": fold_results, "mean_qwk": mean_ens, "std_qwk": std_ens}, f, indent=2)
    print(f"\n  Results saved: {results_path}")

    print("\n  Generating Comparative Charts for Thesis...")
    generate_comparative_charts(fold_results, available_models, output_dir)
    plot_correlation_heatmap(all_data, available_models, output_dir)

    return fold_results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="DR Ensemble — Late Fusion")
    parser.add_argument("--exports", default=EXPORTS_DIR, help="Directory with all .npy exports")
    parser.add_argument("--output", default=OUTPUT_DIR, help="Output directory for ensemble results")
    args = parser.parse_args()
    run_ensemble(args.exports, args.output)
