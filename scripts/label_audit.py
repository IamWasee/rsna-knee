"""Audit a report-derived label table against the 58 radiologist-labelled studies.

    python scripts/label_audit.py data/audit/train.csv data/audit/llm_labels_v4_blend.csv

Prints, per label, the AUC of the soft label against gold and the confusion at
0.5 -- how often the table says "present" where the radiologists said no (false
alarm) and the reverse (miss). Use it to compare a re-extracted table with the
current one before spending GPU on training from it.

What it found on llm_labels_v4_blend.csv (2026-09-27): precision 0.69, recall
0.84, macro AUC 0.893 -- 89 false alarms against 38 misses. Reading all 50
studies with an error, roughly seven in ten false alarms are a finding the
report qualifies as small, mild, trace, low-grade, grade 1-2, incipient, or
hedged ("suggests", "could represent", "suspected"). The host's criteria grade
exactly those negative: moderate or large effusion and Baker's, high-grade and
acute ligament tears, >=1 cm high-grade cartilage loss for OA, and "on the
fence" as negative. src/llm_extract.py's prompt instructs the opposite (rule 6).
Most misses are findings the report never names -- 14 of 27 synovitis cases --
which no text extractor can recover.

Caution: the 58 are the only honest ruler we have for both labels and models.
Write extraction rules from the host's published criteria, not from reading
these 58 reports, or the audit stops measuring anything.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from config import ID_COL, LABELS  # noqa: E402


def auc(y, s):
    r = pd.Series(s).rank().values
    n1 = y.sum()
    n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0) if n1 and n0 else np.nan


def main(train_csv, labels_csv):
    t = pd.read_csv(train_csv)
    lab = pd.read_csv(labels_csv).set_index(ID_COL)
    gold = t[t[LABELS].notna().all(axis=1)].set_index(ID_COL)
    missing = gold.index.difference(lab.index)
    if len(missing):
        raise SystemExit(f"{len(missing)} gold studies absent from {labels_csv}")
    L = lab.loc[gold.index]
    print(f"{'label':<17}{'gold+':>6}{'AUC':>7}{'TP':>5}{'FP':>5}{'FN':>5}"
          f"{'prec':>7}{'recall':>8}")
    tot = np.zeros(3, int)
    aucs = []
    for c in LABELS:
        y = gold[c].astype(int).values
        s = L[c].astype(float).values
        p = s > 0.5
        tp, fp, fn = int((p & (y == 1)).sum()), int((p & (y == 0)).sum()), int((~p & (y == 1)).sum())
        tot += (tp, fp, fn)
        aucs.append(auc(y, s))
        print(f"{c:<17}{y.sum():>6}{aucs[-1]:>7.3f}{tp:>5}{fp:>5}{fn:>5}"
              f"{tp / max(tp + fp, 1):>7.2f}{tp / max(tp + fn, 1):>8.2f}")
    tp, fp, fn = tot
    print(f"\nALL: TP {tp}  FP {fp}  FN {fn}   precision {tp / (tp + fp):.2f}  "
          f"recall {tp / (tp + fn):.2f}   macro AUC {np.nanmean(aucs):.3f}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
