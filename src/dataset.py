"""PyTorch dataset over the preprocessed cache.

Reads the .npy volumes written by preprocess.py, never DICOM -- decoding in the
training loop leaves the GPU idle. One item is a whole study, because labels are
per study, not per slice.

Each group of three physically adjacent slices becomes the R/G/B channels of one
encoder input. That is deliberate: the backbone is pretrained on three-channel
images, and stacking many slices as channels (then averaging the first
convolution's weights) destroys the input interface it learned. Three adjacent
slices give a genuine ~10 mm depth neighbourhood through an untouched backbone.
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

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
GROUP = 3
AUG_ROT_DEG = 8.0
AUG_SCALE = 0.08
AUG_SHIFT = 0.05
AUG_INTENSITY = 0.10


class KneeStudies(Dataset):
    def __init__(self, df: pd.DataFrame, cache: Path, train: bool = True,
                 slots: int = 4, n_slices: int = 9, size: int = 256):
        self.df = df.reset_index(drop=True)
        self.cache = Path(cache)
        self.train = train
        self.shape = (slots, n_slices, size, size)
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
            # Never pad or crop into a mismatch. This previously took the top-left
            # 224x224 of a 256px cache and zero-filled three missing slices, which
            # cost a submission 0.06 AUC with nothing in the log to show for it.
            # A cache built with different settings than the checkpoint expects is
            # a configuration error, not something to paper over.
            raise ValueError(
                f"cache/checkpoint mismatch for {study_id}: cache has {vol.shape}, "
                f"model expects {self.shape}. Rebuild the cache with the same "
                f"--slots/--size the checkpoint was trained with."
            )
        return vol

    def _augment(self, vol: np.ndarray) -> np.ndarray:
        """Rigid jitter and an intensity scale. Neither flip, for two reasons.

        Horizontal: medial and lateral are separate labels and a mirror swaps them,
        corrupting four of the twelve targets.

        Vertical: a knee is acquired in a canonical orientation and no study looks
        like its own vertical mirror. Augmentation should cover directions the label
        is invariant to; this one moves the input off the distribution entirely. Where
        a finding sits in the frame is also information -- a Baker's cyst is identified
        by lying in the popliteal fossa, not by appearance alone.

        Zoom in only. Edge padding repeats the border outward, and the border of this
        crop is where the popliteal fossa sits, so zooming out fabricates tissue
        exactly where a Baker's cyst is looked for.
        """
        import cv2

        x = vol.astype(np.float32)
        if np.random.rand() < 0.8:
            h, w = x.shape[-2:]
            ang = np.random.uniform(-AUG_ROT_DEG, AUG_ROT_DEG)
            sc = 1.0 + np.random.uniform(0, AUG_SCALE)          # never below 1.0
            tx = np.random.uniform(-AUG_SHIFT, AUG_SHIFT) * w
            ty = np.random.uniform(-AUG_SHIFT, AUG_SHIFT) * h
            M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, sc)
            M[0, 2] += tx
            M[1, 2] += ty
            flat = x.reshape(-1, h, w)
            for i in range(flat.shape[0]):
                flat[i] = cv2.warpAffine(flat[i], M, (w, h),
                                         flags=cv2.INTER_LINEAR,
                                         borderMode=cv2.BORDER_REPLICATE)
            x = flat.reshape(x.shape)

        if np.random.rand() < 0.5:
            x *= 1.0 + np.random.uniform(-AUG_INTENSITY, AUG_INTENSITY)
        return np.clip(x, 0, 255).astype(np.uint8)

    def __getitem__(self, i: int):
        row = self.df.iloc[i]
        vol = self._load(row[ID_COL])
        if self.train:
            vol = self._augment(vol)

        slots, n_slices, h, w = vol.shape
        # (slots, 9, H, W) -> (slots*3, 3, H, W): each triplet is one RGB input.
        x = vol.reshape(slots * (n_slices // GROUP), GROUP, h, w).astype(np.float32) / 255.0
        x = (x - MEAN[None, :, None, None]) / STD[None, :, None, None]
        x = torch.from_numpy(x)

        if not self.has_labels:
            return x, row[ID_COL]
        return x, torch.tensor(row[LABELS].values.astype(np.float32))
