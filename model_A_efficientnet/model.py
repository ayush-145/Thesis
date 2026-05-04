"""
model.py — Model A: EfficientNetV2-S
======================================
Local feature baseline — pure CNN.
EfficientNetV2-S backbone → Global Avg Pool → CORN ordinal head.
Supports gradient checkpointing for VRAM savings.
"""

import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
import torchvision.models as tvm
from config import CFG, MODEL_NAME


class CORNHead(nn.Module):
    """CORN ordinal regression head. Outputs K-1 logits."""
    def __init__(self, in_features: int, num_ranks: int = 4, dropout: float = 0.1):
        super().__init__()
        self.head = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Dropout(dropout),
            nn.Linear(in_features, num_ranks),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


class ModelA_EfficientNetV2S(nn.Module):
    """EfficientNetV2-S backbone → Global Avg Pool → CORN head."""
    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = tvm.EfficientNet_V2_S_Weights.DEFAULT if pretrained else None
        backbone = tvm.efficientnet_v2_s(weights=weights)
        self.features = backbone.features
        self.avgpool = backbone.avgpool
        self.feature_dim = CFG.model.effnet_feature_dim
        self.classifier = CORNHead(self.feature_dim, CFG.data.num_ranks, CFG.train.dropout)
        self._features_cache = None
        self._use_grad_checkpoint = False

    def enable_gradient_checkpointing(self):
        """Enable gradient checkpointing for VRAM savings."""
        self._use_grad_checkpoint = True
        print("  Gradient checkpointing enabled for EfficientNetV2-S")

    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Forward through backbone with optional gradient checkpointing."""
        if self._use_grad_checkpoint and self.training:
            # Checkpoint each sequential block in EfficientNet features
            for block in self.features:
                x = cp.checkpoint(block, x, use_reentrant=False)
        else:
            x = self.features(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self._forward_features(x)
        f = self.avgpool(f).flatten(1)
        self._features_cache = f.detach()
        return self.classifier(f)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            f = self.features(x)
            return self.avgpool(f).flatten(1)


def build_model(pretrained: bool = True) -> nn.Module:
    return ModelA_EfficientNetV2S(pretrained=pretrained)


def get_img_size_for_model() -> int:
    return CFG.data.img_size
