"""
model.py — Model B: Swin-V2-Tiny
==================================
Global reasoning baseline — Shifted Window ViT.
Swin-V2-Tiny backbone → LayerNorm → Avg Pool → CORN head.
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


class ModelB_SwinV2T(nn.Module):
    """Swin-V2-Tiny backbone → LayerNorm → Avg Pool → CORN head."""
    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = tvm.Swin_V2_T_Weights.DEFAULT if pretrained else None
        backbone = tvm.swin_v2_t(weights=weights)
        self.features = backbone.features
        self.norm = backbone.norm
        self.permute = backbone.permute
        self.avgpool = backbone.avgpool
        self.flatten = backbone.flatten
        self.feature_dim = CFG.model.swin_feature_dim
        self.classifier = CORNHead(self.feature_dim, CFG.data.num_ranks, CFG.train.dropout)
        self._features_cache = None
        self._use_grad_checkpoint = False

    def enable_gradient_checkpointing(self):
        """Enable gradient checkpointing for VRAM savings."""
        self._use_grad_checkpoint = True
        print("  Gradient checkpointing enabled for Swin-V2-T")

    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        if self._use_grad_checkpoint and self.training:
            for block in self.features:
                x = cp.checkpoint(block, x, use_reentrant=False)
        else:
            x = self.features(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self._forward_features(x)
        f = self.norm(f)
        f = self.permute(f)
        f = self.avgpool(f)
        f = self.flatten(f)
        self._features_cache = f.detach()
        return self.classifier(f)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            f = self.features(x)
            f = self.norm(f)
            f = self.permute(f)
            f = self.avgpool(f)
            return self.flatten(f)


def build_model(pretrained: bool = True) -> nn.Module:
    return ModelB_SwinV2T(pretrained=pretrained)

def get_img_size_for_model() -> int:
    return CFG.data.img_size
