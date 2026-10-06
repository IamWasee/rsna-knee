"""Learned combination of arms versus the equal rank average, judged on the gold 58.

The submission averages each label's percentile rank across arms with equal
weight. This fits, per label, a logistic regression on every arm's percentile
for EVERY label -- so ACL can lean on the sagittal arms, and synovitis on
effusion -- and asks whether that beats the equal average on gold.

Training rows are the out-of-fold predictions of the six original arms against
the report labels (llm_labels_v4_blend, > 0.5). Only those six: a pseudo-label
arm's out-of-fold prediction already carries its teachers' view of that same
study's report label, which would make the stacker trust it for the wrong
reason. The regularisation is chosen by cross-validation on the training rows
over the arms' own folds; gold is touched once, at the end.

Decision, fixed 2026-10-06 before the first run: worth a submission only if
gold macro AUC beats the equal 6-arm average by >= +0.005 with paired bootstrap
P(delta > 0) >= 0.85. Prints aggregates only.

    .venv/bin/python scripts/stack_blend.py
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from config import ID_COL, LABELS  # noqa: E402

A = Path("data/audit/arms_oof")
G = Path("data/audit/gold_preds")
ARMS = {  # oof folder -> gold prediction file stem
    "sag-16ep-source/plane_s16_sag": "plane_s16_sag@sag-16ep-source",
    "cor-16ep-source/plane_s16_cor": "plane_s16_cor@cor-16ep-source",
    "ax-16ep-source/plane_s16_ax": "plane_s16_ax@ax-16ep-source",
    "sag-16ep-seed43/plane_s16s43_sag": "plane_s16s43_sag@sag-16ep-seed43",
    "cor-16ep-seed43/plane_s16s43_cor": "plane_s16s43_cor@cor-16ep-seed43",
    "ax-16ep-seed43/plane_s16s43_ax": "plane_s16s43_ax@ax-16ep-seed43",
}
CS = [0.003, 0.01, 0.03, 0.1, 0.3, 1.0]


def pct(a: np.ndarray) -> np.ndarray:
    return np.column_stack([pd.Series(a[:, j]).rank(pct=True).to_numpy() for j in range(a.shape[1])])


def macro(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean([roc_auc_score(y[:, j], p[:, j]) for j in range(y.shape[1])]))


def main() -> None:
    oofs = [pd.read_csv(A / k / "oof.csv").set_index(ID_COL) for k in ARMS]
    ids = oofs[0].index
    # Each plane groups studies into folds by its own series, so the planes' fold
    # columns differ. Every feature is still out-of-fold for its own arm; the
    # stacker's cross-validation just needs one assignment, the first arm's.
    folds = oofs[0].loc[ids, "fold"].to_numpy()
    agree = [float((o.loc[ids, "fold"].to_numpy() == folds).mean()) for o in oofs]
    print("fold agreement with the first arm: " + " ".join(f"{a:.2f}" for a in agree))
    X = np.hstack([pct(o.loc[ids, LABELS].to_numpy()) for o in oofs])      # (n, 6*12)
    rep = pd.read_csv("data/audit/llm_labels_v4_blend.csv").set_index(ID_COL)
    Y = (rep.loc[ids, LABELS].astype(float).to_numpy() > 0.5).astype(int)

    g = pd.read_csv(G / "gold_labels.csv")
    Yg = g[LABELS].astype(int).to_numpy()
    Xg = np.hstack([pct(sum(pct(f) for f in np.load(G / f"{s}.npy")) / 5) for s in ARMS.values()])

    na = len(ARMS)
    equal_g = sum(Xg[:, i * 12:(i + 1) * 12] for i in range(na)) / na

    # choose C per label by out-of-fold AUC on the report labels, over the arms' folds
    preds_g = np.zeros_like(equal_g)
    chosen = {}
    for j, c in enumerate(LABELS):
        best = (-1.0, None)
        for C in CS:
            oof = np.zeros(len(ids))
            for f in np.unique(folds):
                tr, va = folds != f, folds == f
                m = LogisticRegression(C=C, max_iter=2000).fit(X[tr], Y[tr, j])
                oof[va] = m.predict_proba(X[va])[:, 1]
            a = roc_auc_score(Y[:, j], oof)
            if a > best[0]:
                best = (a, C)
        chosen[c] = best
        m = LogisticRegression(C=best[1], max_iter=2000).fit(X, Y[:, j])
        preds_g[:, j] = m.predict_proba(Xg)[:, 1]

    eq_oof = sum(X[:, i * 12:(i + 1) * 12] for i in range(na)) / na
    print("report-label OOF, equal 6-arm average: %.4f" % macro(Y, eq_oof))
    print("report-label CV, stacker:              %.4f" % np.mean([v[0] for v in chosen.values()]))
    print("C per label: " + "  ".join(f"{c.split()[0][:4]} {v[1]}" for c, v in chosen.items()))

    rng = np.random.default_rng(0)
    idx = [rng.integers(0, len(Yg), len(Yg)) for _ in range(2000)]

    def ok(i):  # a resample must keep both classes in every label
        return all(0 < Yg[i, j].sum() < len(i) for j in range(Yg.shape[1]))

    idx = [i for i in idx if ok(i)]
    d = np.array([macro(Yg[i], preds_g[i]) - macro(Yg[i], equal_g[i]) for i in idx])
    base, new = macro(Yg, equal_g), macro(Yg, preds_g)
    print(f"\ngold: equal 6-arm {base:.4f}   stacker {new:.4f}   delta {new - base:+.4f}   "
          f"P(delta>0) {np.mean(d > 0):.2f}   90% [{np.percentile(d, 5):+.4f}, {np.percentile(d, 95):+.4f}]"
          f"   ({len(idx)} resamples)")
    print("per label (stacker - equal): " + "  ".join(
        f"{c.split()[0][:4]} {roc_auc_score(Yg[:, j], preds_g[:, j]) - roc_auc_score(Yg[:, j], equal_g[:, j]):+.3f}"
        for j, c in enumerate(LABELS)))
    verdict = new - base >= 0.005 and np.mean(d > 0) >= 0.85
    print("\nVERDICT (pre-registered: >= +0.005 and P >= 0.85):", "worth a submission" if verdict else "no")


if __name__ == "__main__":
    main()
