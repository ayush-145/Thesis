"""
losses.py — Hybrid Ordinal Loss for DR Classification
======================================================
Combines CORN (ordinal regression) + Focal (class imbalance) loss.
L_total = α · L_CORN + (1-α) · L_Focal
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import CFG


class CORNLoss(nn.Module):
    """
    Conditional Ordinal Regression Loss.

    For K classes, CORN uses K-1 independent binary classifiers,
    each predicting P(Y > k | Y >= k). This enforces ordinal
    consistency and directly aligns with QWK's distance-aware penalty.

    Each binary task uses BCE loss conditioned on the relevant samples:
    - Task k: only includes samples where true_grade >= k.

    Reference: Shi et al., "CORN — Conditional Ordinal Regression for
    Neural Networks" (Pattern Recognition, 2021).
    """
    def __init__(self, num_classes: int = 5):
        super().__init__()
        self.num_classes = num_classes
        self.num_ranks = num_classes - 1

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Raw model output, shape (B, num_ranks).
            labels: Integer class labels, shape (B,), values in [0, num_classes-1].

        Returns:
            Scalar loss.
        """
        batch_size = logits.size(0)
        device = logits.device

        total_loss = torch.zeros(1, device=device)
        num_tasks_with_samples = 0

        for k in range(self.num_ranks):
            # Conditional set: samples where true grade >= k
            # (these are the samples relevant to the binary task "is grade > k?")
            condition_mask = labels >= k
            if condition_mask.sum() == 0:
                continue

            # Extract relevant logits and binary targets
            task_logits = logits[condition_mask, k]
            task_targets = (labels[condition_mask] > k).float()

            # Binary cross-entropy for this rank
            task_loss = F.binary_cross_entropy_with_logits(
                task_logits, task_targets, reduction="mean"
            )
            total_loss += task_loss
            num_tasks_with_samples += 1

        if num_tasks_with_samples > 0:
            total_loss /= num_tasks_with_samples

        return total_loss.squeeze()


class OrdinalFocalLoss(nn.Module):
    """
    Focal Loss adapted for ordinal binary tasks.

    Applied per-rank (matching CORN's K-1 binary structure).
    Down-weights easy examples, focuses on hard-to-classify boundaries.

    L_focal = -α_t · (1 - p_t)^γ · log(p_t)
    """
    def __init__(self, gamma: float = 2.0, alpha: float = 0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, corn_labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Shape (B, num_ranks), raw ordinal logits.
            corn_labels: Shape (B, num_ranks), binary ordinal targets
                         (e.g., Grade 2 → [1, 1, 0, 0]).

        Returns:
            Scalar focal loss averaged across all ranks.
        """
        probs = torch.sigmoid(logits)

        # Focal weighting
        p_t = corn_labels * probs + (1 - corn_labels) * (1 - probs)
        focal_weight = (1 - p_t) ** self.gamma

        # Alpha balancing
        alpha_t = corn_labels * self.alpha + (1 - corn_labels) * (1 - self.alpha)

        # BCE component
        bce = F.binary_cross_entropy_with_logits(
            logits, corn_labels, reduction="none"
        )

        loss = alpha_t * focal_weight * bce
        return loss.mean()


class HybridOrdinalLoss(nn.Module):
    """
    Combined loss: α · CORN + (1-α) · OrdinalFocal

    CORN handles ordinal consistency (aligned with QWK).
    OrdinalFocal handles class imbalance (focuses on hard examples).

    Args:
        corn_alpha: Weight for CORN loss (default 0.7).
        focal_gamma: Focal loss γ parameter (default 2.0).
    """
    def __init__(
        self,
        corn_alpha: float = 0.7,
        focal_gamma: float = 2.0,
        focal_alpha: float = 0.25,
        num_classes: int = 5,
    ):
        super().__init__()
        self.corn_weight = corn_alpha
        self.focal_weight = 1.0 - corn_alpha
        self.corn_loss = CORNLoss(num_classes=num_classes)
        self.focal_loss = OrdinalFocalLoss(gamma=focal_gamma, alpha=focal_alpha)

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        corn_labels: torch.Tensor,
    ) -> dict:
        """
        Args:
            logits: (B, num_ranks), raw model output.
            labels: (B,), integer class labels.
            corn_labels: (B, num_ranks), binary ordinal targets.

        Returns:
            Dict with 'total', 'corn', and 'focal' loss values.
        """
        l_corn = self.corn_loss(logits, labels)
        l_focal = self.focal_loss(logits, corn_labels)

        total = self.corn_weight * l_corn + self.focal_weight * l_focal

        return {
            "total": total,
            "corn": l_corn.detach(),
            "focal": l_focal.detach(),
        }
