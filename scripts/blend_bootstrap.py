"""Compare arm blends on the gold 58 with a paired bootstrap over studies.

Reads the per-arm prediction files cell 45 saves -- <arm>.npy, (folds, 58, 12),
plus gold_labels.csv -- and blends them as the submission does: folds
rank-averaged inside each arm, arms percentile-ranked and averaged equally.

    .venv/bin/python scripts/blend_bootstrap.py data/audit/gold_preds \\
        --ref "A,B,C" --cand "A,B,C,D" [--cand ...]

Arm names are the .npy stems ("plane_pseudo50_ax@ax-pseudo50__single"). The
reference and each candidate are resampled on the same studies, so the
interval is for the difference. 58 studies make it wide: use it to veto a
blend that is clearly worse, not to pick the best of several.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def rank_cols(a: np.ndarray) -> np.ndarray:
    return np.column_stack([pd.Series(a[:, j]).rank(pct=True).to_numpy()
                            for j in range(a.shape[1])])


def auc(y: np.ndarray, s: np.ndarray) -> float:
    r = pd.Series(s).rank().to_numpy()
    n1 = y.sum()
    n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0) if n1 and n0 else np.nan


def macro(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.nanmean([auc(y[:, j], p[:, j]) for j in range(y.shape[1])]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("preds", type=Path)
    ap.add_argument("--ref", required=True, help="comma-separated arm names")
    ap.add_argument("--cand", action="append", required=True)
    ap.add_argument("--boots", type=int, default=2000)
    args = ap.parse_args()

    g = pd.read_csv(args.preds / "gold_labels.csv")
    y = g[g.columns[1:]].astype(int).to_numpy()
    cache: dict[str, np.ndarray] = {}

    def arm(name: str) -> np.ndarray:
        if name not in cache:
            pm = np.load(args.preds / f"{name}.npy")
            cache[name] = sum(rank_cols(f) for f in pm) / len(pm)
        return cache[name]

    def blend(names: str) -> np.ndarray:
        ns = [n.strip() for n in names.split(",") if n.strip()]
        return sum(rank_cols(arm(n)) for n in ns) / len(ns)

    ref = blend(args.ref)
    rng = np.random.default_rng(0)
    idx = [rng.integers(0, len(y), len(y)) for _ in range(args.boots)]
    print(f"reference: {macro(y, ref):.4f}")
    for c in args.cand:
        p = blend(c)
        d = np.array([macro(y[i], p[i]) - macro(y[i], ref[i]) for i in idx])
        print(f"{macro(y, p):.4f}  delta {macro(y, p) - macro(y, ref):+.4f}  "
              f"P(delta>0) {np.mean(d > 0):.2f}  90% [{np.percentile(d, 5):+.4f}, "
              f"{np.percentile(d, 95):+.4f}]   {c}")


if __name__ == "__main__":
    main()
