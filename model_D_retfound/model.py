"""
model.py — Model D: RETFound ViT-L/16 + LoRA
===============================================
Foundation Model paradigm: RETFound (ViT-Large) pretrained via MAE on 1.6M retinal images.
All backbone weights frozen. LoRA injected into self-attention qkv and proj layers.
Trainable params: ~1.2M out of 307M (~0.4%).
Input: 224x224 (RETFound native resolution).
"""

import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from typing import Optional
from config import CFG, MODEL_NAME


class CORNHead(nn.Module):
    def __init__(self, in_features, num_ranks=4, dropout=0.1):
        super().__init__()
        self.head = nn.Sequential(nn.LayerNorm(in_features), nn.Dropout(dropout),
                                  nn.Linear(in_features, num_ranks))
    def forward(self, x): return self.head(x)


class LoRALinear(nn.Module):
    """Low-Rank Adaptation wrapper. Freezes W, adds trainable B@A decomposition."""
    def __init__(self, original_linear, rank=8, alpha=16, dropout=0.05):
        super().__init__()
        self.original = original_linear
        self.rank = rank
        self.scaling = alpha / rank
        d_in, d_out = original_linear.in_features, original_linear.out_features
        self.original.weight.requires_grad = False
        if self.original.bias is not None:
            self.original.bias.requires_grad = False
        self.lora_A = nn.Parameter(torch.zeros(rank, d_in))
        self.lora_B = nn.Parameter(torch.zeros(d_out, rank))
        self.lora_dropout = nn.Dropout(dropout)
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        result = self.original(x)
        lora_out = self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T
        return result + lora_out * self.scaling


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=1024):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)


class Attention(nn.Module):
    def __init__(self, dim, num_heads=16, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return self.proj_drop(x)


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, drop=0.):
        super().__init__()
        hidden_features = hidden_features or in_features * 4
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(drop)
    def forward(self, x):
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads, qkv_bias, attn_drop, drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), drop)
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformerMAE(nn.Module):
    """ViT-Large encoder matching RETFound's MAE architecture."""
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4., qkv_bias=True, drop_rate=0.):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias, drop_rate, drop_rate)
            for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        self.embed_dim = embed_dim
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x[:, 0]


class ModelD_RETFound(nn.Module):
    """RETFound (ViT-L/16) + LoRA. Backbone frozen, only LoRA adapters + head trained."""
    def __init__(self, weights_path: Optional[str] = None, pretrained: bool = True):
        super().__init__()
        self.backbone = VisionTransformerMAE(
            img_size=CFG.data.img_size, patch_size=CFG.model.retfound_patch_size,
            embed_dim=CFG.model.retfound_embed_dim, depth=CFG.model.retfound_depth,
            num_heads=CFG.model.retfound_num_heads)

        if pretrained:
            resolved_path = self._resolve_weights(weights_path)
            if resolved_path:
                self._load_retfound_weights(resolved_path)

        for param in self.backbone.parameters():
            param.requires_grad = False
        self._inject_lora()

        self.feature_dim = CFG.model.retfound_embed_dim
        self.classifier = CORNHead(self.feature_dim, CFG.data.num_ranks, CFG.train.dropout)
        self._features_cache = None

    @staticmethod
    def _resolve_weights(weights_path):
        if weights_path and os.path.exists(weights_path):
            print(f"  RETFound weights found at: {weights_path}")
            return weights_path

        # Attempt to load HF_TOKEN from environment or notebook secrets
        token = os.environ.get("HF_TOKEN", None)
        if not token:
            from config import IS_KAGGLE, IS_COLAB
            if IS_KAGGLE:
                try:
                    from kaggle_secrets import UserSecretsClient
                    token = UserSecretsClient().get_secret("HF_TOKEN")
                    print("  Loaded HF_TOKEN from Kaggle Secrets.")
                except Exception:
                    pass
            elif IS_COLAB:
                try:
                    from google.colab import userdata
                    token = userdata.get("HF_TOKEN")
                    print("  Loaded HF_TOKEN from Colab userdata.")
                except Exception:
                    pass

        try:
            from huggingface_hub import hf_hub_download
            print(f"  Downloading RETFound from HuggingFace: {CFG.paths.retfound_hf_repo}...")
            path = hf_hub_download(repo_id=CFG.paths.retfound_hf_repo,
                                   filename=CFG.paths.retfound_hf_filename, token=token)
            print(f"  Downloaded to: {path}")
            return path
        except ImportError:
            print("  huggingface_hub not installed.")
        except Exception as e:
            print(f"  HuggingFace download failed: {e}")
        print("  No RETFound weights available. Using random init.")
        return None

    def _load_retfound_weights(self, weights_path):
        checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
        state_dict = checkpoint.get("model", checkpoint)
        encoder_dict = {}
        for k, v in state_dict.items():
            if k.startswith("decoder") or k.startswith("mask_token"):
                continue
            clean_key = k.replace("encoder.", "") if k.startswith("encoder.") else k
            encoder_dict[clean_key] = v
        msg = self.backbone.load_state_dict(encoder_dict, strict=False)
        print(f"  RETFound loaded. Missing: {len(msg.missing_keys)}, Unexpected: {len(msg.unexpected_keys)}")

    def _inject_lora(self):
        lora_count = 0
        for block in self.backbone.blocks:
            if hasattr(block.attn, 'qkv'):
                block.attn.qkv = LoRALinear(block.attn.qkv, rank=CFG.model.lora_rank,
                    alpha=CFG.model.lora_alpha, dropout=CFG.model.lora_dropout)
                lora_count += 1
            if hasattr(block.attn, 'proj'):
                block.attn.proj = LoRALinear(block.attn.proj, rank=CFG.model.lora_rank,
                    alpha=CFG.model.lora_alpha, dropout=CFG.model.lora_dropout)
                lora_count += 1
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"  LoRA: {lora_count} layers. Total: {total:,} | Trainable: {trainable:,} ({100*trainable/total:.2f}%)")

    def forward(self, x):
        if CFG.train.use_grad_checkpoint and self.training:
            features = self._forward_with_checkpointing(x)
        else:
            features = self.backbone(x)
        self._features_cache = features.detach()
        return self.classifier(features)

    def _forward_with_checkpointing(self, x):
        B = x.shape[0]
        x = self.backbone.patch_embed(x)
        cls_tokens = self.backbone.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.backbone.pos_embed
        for blk in self.backbone.blocks:
            x = cp.checkpoint(blk, x, use_reentrant=False)
        x = self.backbone.norm(x)
        return x[:, 0]

    def get_features(self, x):
        with torch.no_grad():
            return self.backbone(x)


def build_model(pretrained: bool = True) -> nn.Module:
    return ModelD_RETFound(
        weights_path=CFG.paths.retfound_weights if pretrained else None,
        pretrained=pretrained)

def get_img_size_for_model() -> int:
    return CFG.data.img_size
