"""
metrics.py — Evaluation Metrics & Visualization
=================================================
Shared metric computation and plot generation functions.
- QWK, accuracy, classification report
- Confusion matrix, ROC-AUC, Precision-Recall curves
- UMAP & t-SNE latent space visualizations
- Training curves (loss, QWK, LR)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    cohen_kappa_score, confusion_matrix, roc_curve, auc,
    precision_recall_curve, average_precision_score,
    classification_report,
)
from sklearn.preprocessing import label_binarize
from typing import Dict, Optional
from config import CFG

plt.rcParams.update({
    "font.family": "serif", "font.size": 11,
    "axes.labelsize": 12, "axes.titlesize": 13, "figure.dpi": 150,
})


def compute_all_metrics(labels: np.ndarray, preds: np.ndarray) -> Dict:
    qwk = cohen_kappa_score(labels, preds, weights="quadratic")
    accuracy = np.mean(labels == preds)
    report = classification_report(
        labels, preds, target_names=list(CFG.data.class_names), output_dict=True
    )
    return {"qwk": qwk, "accuracy": accuracy, "report": report}


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
    probs = probs / np.where(row_sums > 0, row_sums, 1.0)
    return probs


def plot_confusion_matrix(labels, preds, title="Confusion Matrix", save_path=None):
    cm = confusion_matrix(labels, preds, labels=list(range(CFG.data.num_classes)))
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    cm_norm = np.nan_to_num(cm_norm)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=CFG.data.class_names, yticklabels=CFG.data.class_names,
                ax=ax, vmin=0, vmax=1, linewidths=0.5, linecolor="white")
    ax.set_xlabel("Predicted Grade"); ax.set_ylabel("True Grade"); ax.set_title(title)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j + 0.5, i + 0.75, f"n={cm[i,j]}", ha="center",
                    va="center", fontsize=7, color="gray")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=CFG.eval.fig_dpi, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close(fig)


def plot_roc_curves(labels, logits, title="ROC Curves (OvR)", save_path=None):
    probs = corn_logits_to_class_probs(logits)
    y_bin = label_binarize(labels, classes=list(range(CFG.data.num_classes)))
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = plt.cm.Set2(np.linspace(0, 1, CFG.data.num_classes))
    for i, (name, color) in enumerate(zip(CFG.data.class_names, colors)):
        if y_bin[:, i].sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(y_bin[:, i], probs[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=2, label=f"{name} (AUC={roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title(title); ax.legend(loc="lower right", fontsize=9)
    ax.set_xlim([-0.02, 1.02]); ax.set_ylim([-0.02, 1.02])
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=CFG.eval.fig_dpi, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close(fig)


def plot_pr_curves(labels, logits, title="Precision-Recall Curves (OvR)", save_path=None):
    probs = corn_logits_to_class_probs(logits)
    y_bin = label_binarize(labels, classes=list(range(CFG.data.num_classes)))
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = plt.cm.Set2(np.linspace(0, 1, CFG.data.num_classes))
    for i, (name, color) in enumerate(zip(CFG.data.class_names, colors)):
        if y_bin[:, i].sum() == 0:
            continue
        precision, recall, _ = precision_recall_curve(y_bin[:, i], probs[:, i])
        ap = average_precision_score(y_bin[:, i], probs[:, i])
        ax.plot(recall, precision, color=color, lw=2, label=f"{name} (AP={ap:.3f})")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title(title); ax.legend(loc="lower left", fontsize=9)
    ax.set_xlim([-0.02, 1.02]); ax.set_ylim([-0.02, 1.02])
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=CFG.eval.fig_dpi, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close(fig)


def plot_latent_space(features, labels, title="Latent Space", save_path=None, method="umap"):
    if method == "umap":
        try:
            from umap import UMAP
            reducer = UMAP(n_neighbors=CFG.eval.umap_n_neighbors, min_dist=CFG.eval.umap_min_dist,
                           n_components=CFG.eval.umap_n_components, random_state=CFG.data.random_seed,
                           metric="cosine")
        except ImportError:
            print("  WARNING: umap-learn not installed. Falling back to t-SNE.")
            method = "tsne"
    if method == "tsne":
        from sklearn.manifold import TSNE
        reducer = TSNE(n_components=2, perplexity=CFG.eval.tsne_perplexity,
                       n_iter=CFG.eval.tsne_n_iter, random_state=CFG.data.random_seed)
    embedding = reducer.fit_transform(features)
    fig, ax = plt.subplots(figsize=(9, 7))
    scatter = ax.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap="Spectral",
                         s=12, alpha=0.7, edgecolors="none")
    cbar = plt.colorbar(scatter, ax=ax, ticks=range(CFG.data.num_classes))
    cbar.set_ticklabels(CFG.data.class_names); cbar.set_label("DR Grade")
    ax.set_xlabel(f"{method.upper()} 1"); ax.set_ylabel(f"{method.upper()} 2")
    ax.set_title(title); ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=CFG.eval.fig_dpi, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close(fig)


def plot_training_curves(history, title="Training Curves", save_path=None):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    epochs = range(1, len(history["train_loss"]) + 1)
    axes[0].plot(epochs, history["train_loss"], "b-", label="Train", lw=1.5)
    axes[0].plot(epochs, history["val_loss"], "r-", label="Val", lw=1.5)
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss Curves"); axes[0].legend(); axes[0].grid(True, alpha=0.3)
    axes[1].plot(epochs, history["train_qwk"], "b-", label="Train", lw=1.5)
    axes[1].plot(epochs, history["val_qwk"], "r-", label="Val", lw=1.5)
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("QWK")
    axes[1].set_title("Quadratic Weighted Kappa"); axes[1].legend(); axes[1].grid(True, alpha=0.3)
    axes[2].plot(epochs, history["lr"], "g-", lw=1.5)
    axes[2].set_xlabel("Epoch"); axes[2].set_ylabel("Learning Rate")
    axes[2].set_title("LR Schedule"); axes[2].set_yscale("log"); axes[2].grid(True, alpha=0.3)
    plt.suptitle(title, fontsize=14, y=1.02); plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=CFG.eval.fig_dpi, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close(fig)
