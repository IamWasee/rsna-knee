"""Combine the keyword extractor and the model extractor.

Measured on the 58 gold studies, the two disagree in structured ways: the keyword
extractor wins the ligaments (MCL 0.793 vs 0.522, ACL 0.775 vs 0.618), the model
wins osteoarthritis and trauma (Lateral OA 0.839 vs 0.646, PF OA 0.710 vs 0.579).
Ligaments are named in nearly every report and declared normal, so they reduce to
negation, which clause-scoped regex handles well. OA and contusion are described
in varied prose, where a language model earns its keep.

Complementary errors are what makes an ensemble worth more than its parts.

    python src/ensemble.py --data train.csv --model-preds model_preds.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import ID_COL, LABELS  # noqa: E402
from report_labels import keyword_extract, macro_auc  # noqa: E402


def rank_norm(df: pd.DataFrame) -> pd.DataFrame:
    """Map each label to [0,1] by rank.

    AUC only sees ordering, and the two sources are on different scales -- the
    keyword extractor emits three discrete values (0.15/0.25/0.85), the model emits
    a continuum. Averaging raw scores lets the coarse source dominate ties; ranking
    first puts them on equal footing.
    """
    return df[LABELS].rank(pct=True)


def combine(kw: pd.DataFrame, ml: pd.DataFrame, how: str, w: float = 0.5) -> pd.DataFrame:
    if how == "mean":
        return kw[LABELS] * (1 - w) + ml[LABELS] * w
    if how == "rank_mean":
        return rank_norm(kw) * (1 - w) + rank_norm(ml) * w
    if how == "max":
        return kw[LABELS].combine(ml[LABELS], lambda a, b: pd.concat([a, b], axis=1).max(axis=1))
    raise ValueError(how)


def score(name: str, truth: pd.DataFrame, preds: pd.DataFrame, per_label: bool = False) -> float:
    s, by_label = macro_auc(truth, preds.reindex(columns=LABELS).fillna(0.5))
    print(f"{name:<28} {s:.3f}")
    if per_label:
        for c in LABELS:
            print(f"    {c:<20} {by_label[c]:.3f}")
    return s


def _diagnose(kw: pd.DataFrame, ml: pd.DataFrame) -> None:
    """Per-source positive rates, before combining.

    The ensemble output is a blend and can hide a broken source. Comparing each
    source separately against the gold prevalence is what identifies which one is
    wrong -- and whether the disagreement is real or an artifact of the combiner.
    """
    from config import GOLD_PREVALENCE

    print(f"\n{'label':<18} {'gold':>6} {'keyword':>8} {'model':>7} {'flag':>6}")
    print("-" * 50)
    for c in LABELS:
        g = GOLD_PREVALENCE[c]
        k = (kw[c] > 0.5).mean()
        m = (ml[c] > 0.5).mean()
        # 3x off the gold base rate is not sampling noise at n=3500.
        flag = ""
        if not (g / 3 < k < min(1.0, g * 3)):
            flag += "K"
        if not (g / 3 < m < min(1.0, g * 3)):
            flag += "M"
        print(f"{c:<18} {g:>6.3f} {k:>8.3f} {m:>7.3f} {flag:>6}")
    print("-" * 50)
    print("K = keyword far off gold base rate, M = model far off. Neither is proof")
    print("of a bug -- gold is 58 studies -- but a 3x gap is worth reading reports for.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True, help="train.csv")
    ap.add_argument("--model-preds", type=Path, required=True,
                    help="CSV written by local_extract.py (or llm_extract.py)")
    ap.add_argument("--out", type=Path, help="write combined predictions here")
    ap.add_argument("--method", default="mean", choices=["mean", "rank_mean", "max"],
                    help="mean keeps a probability scale, which training targets need; "
                         "rank_mean scored marginally higher but its output is a rank, "
                         "not a probability, so prevalence checks are meaningless on it")
    ap.add_argument("--weight", type=float, default=0.5, help="weight on the model source")
    ap.add_argument("--per-label", action="store_true")
    args = ap.parse_args()

    train = pd.read_csv(args.data)
    ml = pd.read_csv(args.model_preds).set_index(ID_COL)

    subset = train[train[ID_COL].isin(ml.index)].copy()
    kw = pd.DataFrame([keyword_extract(t) for t in subset["Report"]],
                      index=subset[ID_COL]).reindex(columns=LABELS)
    ml = ml.reindex(kw.index)

    labeled = subset[subset[LABELS].notna().all(axis=1)]
    if labeled.empty:
        # Unlabeled split: nothing to score, just write the combination.
        _diagnose(kw, ml)
        out = combine(kw, ml, args.method, args.weight).reset_index()
        out.to_csv(args.out or "ensemble_preds.csv", index=False)
        print(f"\nwrote {args.out or 'ensemble_preds.csv'}: {len(out)} studies "
              f"({args.method} w={args.weight}, unscored)")
        return

    truth = labeled[LABELS]
    idx = labeled[ID_COL]
    k, m = kw.loc[idx], ml.loc[idx]

    _diagnose(kw, ml)
    print(f"\nscored on {len(labeled)} gold studies\n")
    score("keyword alone", truth, k.set_index(truth.index))
    score("model alone", truth, m.set_index(truth.index))
    print()

    best, best_name, best_preds = -1.0, "", None
    for how in ("mean", "rank_mean", "max"):
        for w in ((0.3, 0.5, 0.7) if how != "max" else (0.5,)):
            preds = combine(k, m, how, w).set_index(truth.index)
            label = f"{how}" + (f" w={w}" if how != "max" else "")
            s = score(label, truth, preds)
            if s > best:
                best, best_name, best_preds = s, label, preds

    print(f"\nbest: {best_name} at {best:.3f}")
    print("\nCAUTION: picking the best of 7 combinations on 58 studies overfits. The")
    print("honest read is whether ensembling beats BOTH sources across most settings,")
    print("not which weight happened to win. Prefer rank_mean w=0.5 unless one source")
    print("is clearly stronger overall.")
    if args.per_label and best_preds is not None:
        print(f"\nper-label for {best_name}:")
        score(best_name, truth, best_preds, per_label=True)


if __name__ == "__main__":
    main()
