"""Study-level classifier: per-slice CNN encoder + attention pooling.

The supervision is one label set per study, but a study is dozens of slices and a
finding is visible in only a few of them. That is multiple-instance learning: the
model has to find which slices matter without ever being told.

Mean pooling would drown a small finding -- one torn meniscus across 36 slices
averages away to nothing. Max pooling latches onto a single slice and is unstable
under noise. Gated attention pooling learns a weight per slice, which is both a
better fit and inspectable: the weights show which slices drove a prediction.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class AttentionPool(nn.Module):
    """Gated attention (Ilse et al., 2018). Weights are per slice, per study."""

    def __init__(self, dim: int, hidden: int = 256):
        super().__init__()
        self.v = nn.Sequential(nn.Linear(dim, hidden), nn.Tanh())
        self.u = nn.Sequential(nn.Linear(dim, hidden), nn.Sigmoid())
        self.w = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (B, N, D)
        a = self.w(self.v(x) * self.u(x))          # (B, N, 1)
        a = torch.softmax(a, dim=1)
        return (a * x).sum(dim=1), a.squeeze(-1)   # (B, D), (B, N)


class KneeModel(nn.Module):
    def __init__(self, backbone: str = "resnet34", num_labels: int = 12,
                 pretrained: bool = True, dropout: float = 0.3):
        super().__init__()
        import timm

        self.encoder = timm.create_model(
            backbone, pretrained=pretrained, num_classes=0, in_chans=3
        )
        dim = self.encoder.num_features
        self.pool = AttentionPool(dim)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(dim, num_labels))

    def forward(self, x: torch.Tensor, return_attention: bool = False):
        # x: (B, N, 3, H, W) -- fold slices into the batch so the CNN sees 2D images
        b, n = x.shape[:2]
        feats = self.encoder(x.flatten(0, 1)).view(b, n, -1)
        pooled, attn = self.pool(feats)
        logits = self.head(pooled)
        return (logits, attn) if return_attention else logits


def build_loss(pos_weight: torch.Tensor | None = None) -> nn.Module:
    """BCE over 12 independent binary targets.

    Multi-label, not multi-class: a knee can have an ACL tear and an effusion and a
    Baker's cyst at once, so no softmax. pos_weight compensates for rare labels --
    the metric is MACRO AUC, which weights Fracture the same as Effusion regardless
    of how much rarer it is.
    """
    return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
