"""Soft bootstrapping: mix the report labels with our own ensemble's predictions.

The report labels are wrong in a particular way -- they are silent where the
report is silent, which for synovitis is most of the time -- and our image
models see the knee the report skipped. Several top teams report the same
lever: train on a mix of the two, not on either alone. Replacing the labels
with predictions was reported worse than mixing; 50/50 was reported to lift.

    target = W * report_label + (1 - W) * mean_over_arms(out_of_fold_prediction)

Every prediction used is out-of-fold: the arm that made it never trained on that
study. The 58 gold studies are kept at their report values -- training drops
them anyway, and no gold label enters this table.

W is fixed at 0.5 before any gold scoring.

    python src/pseudo_labels.py <report_labels.csv> <out.csv> <oof.csv> [<oof.csv> ...]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import ID_COL, LABELS  # noqa: E402

W = 0.5


def main(labels_csv: str, out_csv: str, *oof_csvs: str) -> None:
    lab = pd.read_csv(labels_csv).set_index(ID_COL)
    preds = []
    for p in oof_csvs:
        o = pd.read_csv(p).set_index(ID_COL)[LABELS].astype(float)
        lo, hi = float(o.min().min()), float(o.max().max())
        if lo < 0 or hi > 1:
            raise SystemExit(f"{p}: predictions outside [0,1] ({lo}, {hi}) -- not probabilities")
        preds.append(o)
    ids = preds[0].index
    for p, o in zip(oof_csvs, preds):
        if not o.index.sort_values().equals(ids.sort_values()):
            raise SystemExit(f"{p} covers a different study set")
    ens = sum(o.loc[ids] for o in preds) / len(preds)
    missing = ids.difference(lab.index)
    if len(missing):
        raise SystemExit(f"{len(missing)} OOF studies absent from {labels_csv}")
    out = lab.copy()
    out.loc[ids, LABELS] = W * lab.loc[ids, LABELS].astype(float) + (1 - W) * ens[LABELS]
    out.reset_index().to_csv(out_csv, index=False)
    print(f"{len(ids)} studies mixed ({len(preds)} arms, W={W}); "
          f"{len(out) - len(ids)} kept at report values; wrote {out_csv}")
    moved = (out.loc[ids, LABELS] > 0.5) != (lab.loc[ids, LABELS] > 0.5)
    print(f"{'label':<17}{'pos before':>11}{'after':>7}{'flipped':>9}")
    for c in LABELS:
        print(f"{c:<17}{(lab.loc[ids, c] > 0.5).mean():>11.3f}"
              f"{(out.loc[ids, c] > 0.5).mean():>7.3f}{int(moved[c].sum()):>9}")


if __name__ == "__main__":
    main(*sys.argv[1:])
