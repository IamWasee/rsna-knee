"""Generate submission.csv. Runs inside the offline Kaggle submission notebook.

Constraints this file exists to respect:
  - no internet, so backbone weights come from the checkpoint, never a download
  - 9 hours total, including preprocessing the hidden test set from raw DICOM
  - a malformed CSV fails after all of that, so submission.py validates strictly

    python src/infer.py --cache test_cache --weights weights/ --out submission.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import ID_COL, LABELS  # noqa: E402
from dataset import KneeStudies  # noqa: E402
from model import KneeModel  # noqa: E402
from paths import data_root  # noqa: E402
from submission import validate, write  # noqa: E402


def load_models(weights: Path, device: str) -> tuple[list, dict]:
    ckpts = sorted(weights.glob("*.pt")) if weights.is_dir() else [weights]
    if not ckpts:
        raise FileNotFoundError(f"no .pt checkpoints in {weights}")

    models, cfg = [], {}
    for p in ckpts:
        ck = torch.load(p, map_location=device, weights_only=False)
        cfg = ck["args"]
        # pretrained=False: the weights are in the checkpoint, and there is no
        # internet here to fetch them from anyway.
        m = KneeModel(cfg["backbone"], len(LABELS), pretrained=False).to(device)
        m.load_state_dict(ck["model"])
        m.eval()
        models.append(m)
        print(f"  {p.name}  gold_auc={ck.get('gold_auc', float('nan')):.3f}")
    return models, cfg


@torch.no_grad()
def predict(models: list, loader: DataLoader, device: str) -> tuple[np.ndarray, list]:
    preds, ids = [], []
    for i, (x, sid) in enumerate(loader):
        x = x.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16, enabled=(device == "cuda")):
            # Fold ensemble: average probabilities, not logits -- the folds are
            # separately calibrated and averaging logits lets a confident fold
            # dominate.
            p = torch.stack([torch.sigmoid(m(x).float()) for m in models]).mean(0)
        preds.append(p.cpu().numpy())
        ids.extend(sid)
        if (i + 1) % 25 == 0 or i + 1 == len(loader):
            print(f"  {i+1}/{len(loader)}", flush=True)
    return np.concatenate(preds), ids


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, required=True, help="preprocessed test volumes")
    ap.add_argument("--weights", type=Path, required=True, help=".pt file or directory")
    ap.add_argument("--out", type=Path, default=Path("submission.csv"))
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    root = data_root()

    test = pd.read_csv(root / "test.csv")
    print(f"{len(test)} test studies")

    models, cfg = load_models(args.weights, device)

    ds = KneeStudies(test, cache=args.cache, train=False,
                     n_series=cfg.get("n_series", 3),
                     n_slices=cfg.get("n_slices", 12),
                     size=cfg.get("size", 224))
    loader = DataLoader(ds, batch_size=args.batch, num_workers=args.workers)

    preds, ids = predict(models, loader, device)

    sub = pd.DataFrame(preds, columns=LABELS)
    sub.insert(0, ID_COL, ids)

    # Every test study must appear, in the order the sample file expects.
    sample = pd.read_csv(root / "sample_submission.csv")
    sub = sample[[ID_COL]].merge(sub, on=ID_COL, how="left")
    missing = sub[LABELS].isna().any(axis=1).sum()
    if missing:
        print(f"WARNING: {missing} studies had no prediction -- filling 0.5")
        sub[LABELS] = sub[LABELS].fillna(0.5)

    validate(sub, expected_ids=sample[ID_COL])
    write(sub, args.out)


if __name__ == "__main__":
    main()
