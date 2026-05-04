"""
train.py — APTOS Fine-Tuning for ModelA_EfficientNetV2S
========================================================
Single-stage: ImageNet pretrained weights → fine-tune on APTOS with 5-fold CV.

Features: AMP, gradient accumulation, cosine warmup, early stopping,
robust stateful checkpointing for Kaggle/Colab session resumption.
"""

import os
import json
import time
import torch
import torch.nn as nn
import numpy as np
from torch.amp import GradScaler, autocast
from sklearn.metrics import cohen_kappa_score
from tqdm import tqdm
from typing import Dict, Optional, Tuple

from config import CFG, MODEL_NAME
from model import build_model, get_img_size_for_model
from losses import HybridOrdinalLoss
from dataset import (
    decode_corn_prediction,
    load_aptos_dataframe,
    get_stratified_kfold_splits,
    build_dataloaders,
)


# ──────────────────────────────────────────────
# Scheduler: Cosine Annealing with Linear Warmup
# ──────────────────────────────────────────────

class CosineWarmupScheduler:
    """Linear warmup then cosine decay. Updates per-epoch."""
    def __init__(self, optimizer, warmup_epochs, total_epochs, base_lr, min_lr):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.current_epoch = 0

    def step(self):
        self.current_epoch += 1
        if self.current_epoch <= self.warmup_epochs:
            lr = self.base_lr * (self.current_epoch / self.warmup_epochs)
        else:
            progress = (self.current_epoch - self.warmup_epochs) / \
                       max(1, self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * \
                 (1 + np.cos(np.pi * progress))
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        return lr

    def get_lr(self):
        return self.optimizer.param_groups[0]["lr"]


# ──────────────────────────────────────────────
# Stateful Checkpoint Manager
# ──────────────────────────────────────────────

class CheckpointManager:
    """Saves/loads full training state for session resumption."""
    def __init__(self, checkpoint_dir: str, model_name: str, fold: int):
        self.checkpoint_dir = checkpoint_dir
        self.prefix = f"{model_name}_fold{fold}_stage2"
        os.makedirs(checkpoint_dir, exist_ok=True)

    @property
    def checkpoint_path(self):
        return os.path.join(self.checkpoint_dir, f"{self.prefix}_state.pth")

    @property
    def best_model_path(self):
        return os.path.join(self.checkpoint_dir, f"{self.prefix}_best.pth")

    @property
    def history_path(self):
        return os.path.join(self.checkpoint_dir, f"{self.prefix}_history.json")

    def save_state(self, model, optimizer, scaler, scheduler,
                   epoch, best_qwk, patience_counter, history):
        state = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict() if scaler else None,
            "scheduler_epoch": scheduler.current_epoch,
            "epoch": epoch,
            "best_qwk": best_qwk,
            "patience_counter": patience_counter,
        }
        torch.save(state, self.checkpoint_path)
        with open(self.history_path, "w") as f:
            json.dump(history, f, indent=2)

    def save_best_model(self, model):
        torch.save(model.state_dict(), self.best_model_path)

    def load_state(self, model, optimizer, scaler, scheduler):
        path_to_load = self.checkpoint_path
        history_to_load = self.history_path
        
        if not os.path.exists(path_to_load):
            import glob
            cands = glob.glob(f"/kaggle/input/**/{os.path.basename(self.checkpoint_path)}", recursive=True)
            if cands:
                path_to_load = cands[0]
                hist_cands = glob.glob(f"/kaggle/input/**/{os.path.basename(self.history_path)}", recursive=True)
                if hist_cands:
                    history_to_load = hist_cands[0]
            else:
                return None

        print(f"  ⟳ Resuming from checkpoint: {path_to_load}")
        state = torch.load(path_to_load, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        if scaler and state["scaler_state_dict"]:
            scaler.load_state_dict(state["scaler_state_dict"])
        scheduler.current_epoch = state["scheduler_epoch"]
        history = {}
        if os.path.exists(history_to_load):
            with open(history_to_load) as f:
                history = json.load(f)
        return (state["epoch"] + 1, state["best_qwk"], state["patience_counter"], history)


# ──────────────────────────────────────────────
# Training & Validation Loops
# ──────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, scaler, device,
                    grad_accum_steps=4, max_grad_norm=1.0, use_amp=True, epoch_info=""):
    model.train()
    total_loss, total_corn, total_focal = 0.0, 0.0, 0.0
    all_preds, all_labels = [], []
    num_batches = 0
    optimizer.zero_grad()

    pbar = tqdm(enumerate(loader), total=len(loader), desc=f"  Train {epoch_info}",
                bar_format="{l_bar}{bar:20}{r_bar}", leave=False)

    for step, batch in pbar:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["grade"].to(device, non_blocking=True)
        corn_labels = batch["corn_label"].to(device, non_blocking=True)

        with autocast(device_type="cuda", enabled=use_amp):
            logits = model(images)
            loss_dict = criterion(logits, labels, corn_labels)
            loss = loss_dict["total"] / grad_accum_steps

        scaler.scale(loss).backward()

        if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(loader):
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += loss_dict["total"].item()
        total_corn += loss_dict["corn"].item()
        total_focal += loss_dict["focal"].item()
        num_batches += 1

        preds = decode_corn_prediction(logits.detach())
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        if num_batches % 10 == 0 or (step + 1) == len(loader):
            pbar.set_postfix({
                "loss": f"{total_loss / num_batches:.4f}",
                "VRAM": f"{torch.cuda.memory_allocated() / 1e9:.1f}G" if torch.cuda.is_available() else "N/A",
            })
    pbar.close()
    qwk = cohen_kappa_score(all_labels, all_preds, weights="quadratic")
    return {"loss": total_loss / num_batches, "corn_loss": total_corn / num_batches,
            "focal_loss": total_focal / num_batches, "qwk": qwk}


@torch.no_grad()
def validate(model, loader, criterion, device, use_amp=True):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []
    num_batches = 0

    pbar = tqdm(loader, total=len(loader), desc="  Val  ",
                bar_format="{l_bar}{bar:20}{r_bar}", leave=False)
    for batch in pbar:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["grade"].to(device, non_blocking=True)
        corn_labels = batch["corn_label"].to(device, non_blocking=True)
        with autocast(device_type="cuda", enabled=use_amp):
            logits = model(images)
            loss_dict = criterion(logits, labels, corn_labels)
        total_loss += loss_dict["total"].item()
        num_batches += 1
        preds = decode_corn_prediction(logits)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    pbar.close()
    qwk = cohen_kappa_score(all_labels, all_preds, weights="quadratic")
    return {"loss": total_loss / num_batches, "qwk": qwk,
            "preds": np.array(all_preds), "labels": np.array(all_labels)}


# ──────────────────────────────────────────────
# Single-Fold Trainer
# ──────────────────────────────────────────────

def train_model_fold(fold, train_loader, val_loader, device):
    """Train a single fold: ImageNet → APTOS fine-tuning."""
    model_name = MODEL_NAME
    epochs = CFG.train.epochs
    lr = CFG.train.lr

    print(f"\n{'='*60}")
    print(f"  {model_name} | Fold {fold+1}/{CFG.data.num_folds} | "
          f"LR={lr:.1e} | Epochs={epochs}")
    print(f"{'='*60}")

    model = build_model(pretrained=True)
    model = model.to(device)

    # Enable gradient checkpointing if configured
    if CFG.train.use_grad_checkpoint and hasattr(model, 'enable_gradient_checkpointing'):
        model.enable_gradient_checkpointing()

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params, lr=lr,
        weight_decay=CFG.train.weight_decay,
        betas=CFG.train.betas, eps=CFG.train.eps)

    scheduler = CosineWarmupScheduler(optimizer, CFG.train.warmup_epochs, epochs, lr, CFG.train.min_lr)
    criterion = HybridOrdinalLoss(corn_alpha=CFG.train.corn_alpha, focal_gamma=CFG.train.focal_gamma,
                                  num_classes=CFG.data.num_classes)
    scaler = GradScaler("cuda", enabled=CFG.train.use_amp)
    ckpt = CheckpointManager(CFG.paths.checkpoint_dir, model_name, fold)

    best_qwk, patience_counter, start_epoch = -1.0, 0, 0
    history = {"train_loss": [], "val_loss": [], "train_qwk": [], "val_qwk": [], "lr": []}

    resume = ckpt.load_state(model, optimizer, scaler, scheduler)
    if resume:
        start_epoch, best_qwk, patience_counter, history = resume
        print(f"  Resumed at epoch {start_epoch}, best QWK={best_qwk:.4f}")

        if patience_counter >= CFG.train.early_stop_patience:
            print(f"  Fold already completed (early stopped). Skipping.")
            return {"best_qwk": best_qwk, "history": history, "best_model_path": ckpt.best_model_path}

        if start_epoch >= epochs:
            print(f"  Fold already completed (max epochs). Skipping.")
            return {"best_qwk": best_qwk, "history": history, "best_model_path": ckpt.best_model_path}

    for epoch in range(start_epoch, epochs):
        t0 = time.time()
        current_lr = scheduler.step()
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device,
            grad_accum_steps=CFG.data.grad_accum_steps,
            max_grad_norm=CFG.train.gradient_clip_max_norm,
            use_amp=CFG.train.use_amp, epoch_info=f"E{epoch+1}/{epochs}")
        val_metrics = validate(model, val_loader, criterion, device, CFG.train.use_amp)
        elapsed = time.time() - t0

        print(f"  Fold {fold+1} | Epoch {epoch+1:3d}/{epochs} | "
              f"T_loss={train_metrics['loss']:.4f} T_QWK={train_metrics['qwk']:.4f} | "
              f"V_loss={val_metrics['loss']:.4f} V_QWK={val_metrics['qwk']:.4f} | "
              f"LR={current_lr:.2e} | {elapsed:.0f}s")

        history["train_loss"].append(train_metrics["loss"])
        history["val_loss"].append(val_metrics["loss"])
        history["train_qwk"].append(train_metrics["qwk"])
        history["val_qwk"].append(val_metrics["qwk"])
        history["lr"].append(current_lr)

        if val_metrics["qwk"] > best_qwk + CFG.train.early_stop_min_delta:
            best_qwk = val_metrics["qwk"]
            patience_counter = 0
            ckpt.save_best_model(model)
            print(f"  ★ New best QWK: {best_qwk:.4f}")
        else:
            patience_counter += 1

        ckpt.save_state(model, optimizer, scaler, scheduler,
                        epoch, best_qwk, patience_counter, history)

        if patience_counter >= CFG.train.early_stop_patience:
            print(f"  ✖ Early stopping at epoch {epoch+1} (patience={CFG.train.early_stop_patience})")
            break

    print(f"\n  Final best QWK: {best_qwk:.4f}")
    return {"best_qwk": best_qwk, "history": history, "best_model_path": ckpt.best_model_path}


# ──────────────────────────────────────────────
# Single-Model Pipeline
# ──────────────────────────────────────────────

def run_single_model_pipeline(device: torch.device):
    """Train ModelA_EfficientNetV2S: ImageNet → APTOS 5-fold CV."""
    model_name = MODEL_NAME
    img_size = get_img_size_for_model()
    pipeline_start = time.time()

    print("\n" + "=" * 60)
    print(f"  {model_name} — Fine-tuning on APTOS (ImageNet → APTOS)")
    print("=" * 60)

    aptos_df = load_aptos_dataframe()
    folds = get_stratified_kfold_splits(aptos_df)
    results = {"folds": []}

    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        print(f"\n  ┌─ Fold {fold_idx+1}/{CFG.data.num_folds}")
        train_loader, val_loader = build_dataloaders(
            aptos_df, train_idx, val_idx, img_size, label_col="diagnosis")
        fold_result = train_model_fold(
            fold=fold_idx, train_loader=train_loader,
            val_loader=val_loader, device=device)
        results["folds"].append(fold_result)
        print(f"  └─ Fold {fold_idx+1} done — QWK={fold_result['best_qwk']:.4f}")

    fold_qwks = [f["best_qwk"] for f in results["folds"]]
    results["mean_qwk"] = np.mean(fold_qwks)
    results["std_qwk"] = np.std(fold_qwks)
    print(f"\n  ★ {model_name} — Mean QWK: {results['mean_qwk']:.4f} ± {results['std_qwk']:.4f}")

    total_time = time.time() - pipeline_start
    summary = {"model": model_name, "mean_qwk": results["mean_qwk"],
               "std_qwk": results["std_qwk"], "total_time_seconds": total_time}
    summary_path = os.path.join(CFG.paths.logs_dir, "training_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Training summary saved to: {summary_path}")
    print(f"  Total training time: {total_time/3600:.1f} hours")
    return results


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
    run_single_model_pipeline(device)
