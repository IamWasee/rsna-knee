"""Study-level classifier: per-slice encoder, then per-diagnosis pooling.

Two design choices here matter more than the backbone, and both target the same
failure: small findings.

Pooling. A meniscal tear occupies a tiny fraction of the frame, so a global average
over the feature map dilutes it by orders of magnitude while an effusion survives
easily. That is exactly the measured pattern -- Medial OA 0.89 and Effusion 0.88
against Medial Meniscus 0.59. Keeping the top fraction of each channel's responses
alongside the mean preserves a strong local response instead of averaging it away.

Attention. Each finding is read on a particular sequence: cruciates sagittally,
collaterals and meniscal body coronally, patellar cartilage axially. A single
attention shared by all twelve labels has to find one weighting that suits every
finding at once. Giving each diagnosis its own query lets Baker's look where Baker's
lives.
"""

from __future__ import annotations

import torch
import torch.nn as nn

# Which slots each finding is most readable on, as a soft bias rather than a filter.
# Slot order matches preprocess.SLOTS: Sag-FS, Cor-FS, Ax-FS, Sag-T1.
SLOT_PRIOR = {
    "ACL": (0, 3), "MCL": (1,), "Medial Meniscus": (0, 1), "Lateral Meniscus": (0, 1),
    "Medial OA": (1,), "Lateral OA": (1,), "PF OA": (2,), "Effusion": (0, 2),
    "Synovitis": (0, 2), "Baker's": (0, 2), "Contusion": (0, 1), "Fracture": (0, 1),
}
PRIOR_STRENGTH = 0.55      # exp(0.55) ~ 1.73x preference; biases, never excludes


class FocalPool(nn.Module):
    """Mean plus the upper tail of each channel over the spatial grid.

    Per channel rather than by selecting whole positions: a finding is a strong
    response in a few channels at one location, and picking positions by total
    activation would follow whatever is brightest overall instead.
    """

    def __init__(self, frac: int = 8):
        super().__init__()
        self.frac = frac

    def forward(self, fmap: torch.Tensor) -> torch.Tensor:   # (N, C, H, W)
        n, c = fmap.shape[:2]
        flat = fmap.flatten(2)                                # (N, C, HW)
        k = max(1, flat.shape[-1] // self.frac)
        return torch.cat([flat.mean(-1), flat.topk(k, dim=-1).values.mean(-1)], dim=1)


class SlotHead(nn.Module):
    """One attention query per diagnosis, over the slot embeddings of a study."""

    def __init__(self, dim: int, n_slot: int, labels: list[str], hidden: int = 256,
                 p: float = 0.2, prior: bool = True):
        super().__init__()
        self.proj = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, hidden), nn.GELU())
        self.slot_emb = nn.Parameter(torch.randn(n_slot, hidden) * 0.02)
        self.query = nn.Parameter(torch.randn(len(labels), hidden) * 0.02)
        self.drop = nn.Dropout(p)
        self.out = nn.Linear(hidden, len(labels))
        self.hidden = hidden

        bias = torch.zeros(len(labels), n_slot)
        if prior:
            for i, t in enumerate(labels):
                for s in SLOT_PRIOR.get(t, ()):
                    if s < n_slot:
                        bias[i, s] = PRIOR_STRENGTH
        self.register_buffer("slot_prior", bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (B, n_slot, dim)
        h = self.proj(x) + self.slot_emb
        att = torch.einsum("bsh,oh->bos", h, self.query) / self.hidden ** 0.5
        att = (att + self.slot_prior.unsqueeze(0)).softmax(-1)
        ctx = self.drop(torch.einsum("bos,bsh->boh", att, h))
        logits = (ctx * self.out.weight.unsqueeze(0)).sum(-1) + self.out.bias
        return logits, att


class AttentionPool(nn.Module):
    """Gated attention shared across labels (Ilse et al.). The previous head, kept
    so the two can be compared rather than swapped on faith."""

    def __init__(self, dim: int, hidden: int = 256):
        super().__init__()
        self.v = nn.Sequential(nn.Linear(dim, hidden), nn.Tanh())
        self.u = nn.Sequential(nn.Linear(dim, hidden), nn.Sigmoid())
        self.w = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        a = torch.softmax(self.w(self.v(x) * self.u(x)), dim=1)
        return (a * x).sum(dim=1), a.squeeze(-1)


class KneeModel(nn.Module):
    def __init__(self, backbone: str = "resnet34", labels: list[str] | None = None,
                 pretrained: bool = True, dropout: float = 0.3,
                 head: str = "slot", pool: str = "focal", n_slot: int = 4,
                 groups_per_slot: int = 3):
        super().__init__()
        import timm

        from config import LABELS
        labels = labels or LABELS
        self.n_slot, self.groups = n_slot, groups_per_slot
        self.head_kind, self.pool_kind = head, pool

        # features_only gives the spatial map that focal pooling needs; num_classes=0
        # would have already collapsed it with the global average we are avoiding.
        self.encoder = timm.create_model(backbone, pretrained=pretrained,
                                         num_classes=0, global_pool="", in_chans=3)
        dim = self.encoder.num_features
        self.focal = FocalPool() if pool == "focal" else None
        feat = dim * (2 if pool == "focal" else 1)

        if head == "slot":
            self.head = SlotHead(feat, n_slot, labels)
        else:
            self.pool_shared = AttentionPool(feat)
            self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(feat, len(labels)))

    def forward(self, x: torch.Tensor, return_attention: bool = False):
        # x: (B, n_slot*groups, 3, H, W)
        b, n = x.shape[:2]
        fmap = self.encoder(x.flatten(0, 1))
        if fmap.ndim == 4:
            feats = self.focal(fmap) if self.focal is not None else fmap.mean((2, 3))
        else:
            feats = fmap
        feats = feats.view(b, n, -1)

        if self.head_kind == "slot":
            # Average the groups within each slot before the head, so attention is over
            # sequence type -- the thing anatomy actually distinguishes -- not over the
            # arbitrary index of a slice group.
            feats = feats.view(b, self.n_slot, self.groups, -1).mean(2)
            logits, attn = self.head(feats)
        else:
            pooled, attn = self.pool_shared(feats)
            logits = self.head(pooled)
        return (logits, attn) if return_attention else logits


def build_loss(pos_weight: torch.Tensor | None = None) -> nn.Module:
    """BCE over 12 independent binary targets -- multi-label, so no softmax: a knee can
    have an ACL tear, an effusion and a Baker's cyst at once."""
    return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
