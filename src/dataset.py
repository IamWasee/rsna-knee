"""PyTorch dataset over the preprocessed cache.

Reads the .npy volumes written by preprocess.py, not DICOM -- decoding DICOM in the
training loop would leave the GPU idle. One item is a whole study, shaped
(n_series * n_slices, 3, H, W), because labels are per study, not per slice.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import ID_COL, LABELS  # noqa: E402

# ImageNet statistics: the backbone is pretrained, so its input distribution is
# fixed even though these are greyscale MRI replicated to 3 channels.
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class KneeStudies(Dataset):
    def __init__(self, df: pd.DataFrame, cache: Path, train: bool = True,
                 n_series: int = 3, n_slices: int = 12, size: int = 224):
        self.df = df.reset_index(drop=True)
        self.cache = Path(cache)
        self.train = train
        self.shape = (n_series, n_slices, size, size)
        self.has_labels = all(c in df.columns for c in LABELS)

    def __len__(self) -> int:
        return len(self.df)

    def _load(self, study_id: str) -> np.ndarray:
        path = self.cache / f"{study_id}.npy"
        if not path.exists():
            # A missing study must not kill a 9-hour inference run.
            return np.zeros(self.shape, dtype=np.uint8)
        vol = np.load(path)
        if vol.shape != self.shape:
            fixed = np.zeros(self.shape, dtype=np.uint8)
            s = tuple(slice(0, min(a, b)) for a, b in zip(vol.shape, self.shape))
            fixed[s] = vol[s]
            vol = fixed
        return vol

    def _augment(self, vol: np.ndarray) -> np.ndarray:
        """Light augmentation only.

        No horizontal flip: left and right knees are both present, but medial and
        lateral are separate labels, and a flip swaps them. That would corrupt four
        of the twelve targets.
        """
        if np.random.rand() < 0.5:                      # brightness / contrast
            vol = np.clip(vol.astype(np.float32) * np.random.uniform(0.85, 1.15)
                          + np.random.uniform(-15, 15), 0, 255).astype(np.uint8)
        if np.random.rand() < 0.3:                      # small translation
            dx, dy = np.random.randint(-12, 13, 2)
            vol = np.roll(vol, (dy, dx), axis=(-2, -1))
        return vol

    def __getitem__(self, i: int):
        row = self.df.iloc[i]
        vol = self._load(row[ID_COL])
        if self.train:
            vol = self._augment(vol)

        n_series, n_slices, h, w = vol.shape
        x = vol.reshape(n_series * n_slices, h, w).astype(np.float32) / 255.0
        x = np.repeat(x[:, None], 3, axis=1)            # (N, 3, H, W)
        x = (x - MEAN[None, :, None, None]) / STD[None, :, None, None]
        x = torch.from_numpy(x)

        if not self.has_labels:
            return x, row[ID_COL]
        y = torch.tensor(row[LABELS].values.astype(np.float32))
        return x, y
