"""Turn multilingual radiology reports into the 12 labels, and measure how well.

The competition ships 4,407 reports but only 58 label sets. Everything downstream
depends on labelling the other 4,349 well, so this module's real job is the
*evaluation harness*: any extractor is a callable

    extract(report_text: str) -> dict[label -> float in 0..1]

and `evaluate()` scores it against the 58 gold studies with the competition metric.
Swap extractors freely; the number they have to beat is measured, not argued about.

    python src/report_labels.py --data path/to/train.csv          # score the baseline
    python src/report_labels.py --data train.csv --show-errors    # see what it got wrong
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import ID_COL, LABELS  # noqa: E402


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------
def gold_split(train: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split train.csv into the 58 labeled studies and the 4,349 unlabeled ones."""
    labeled = train[train[LABELS].notna().all(axis=1)].copy()
    unlabeled = train[train[LABELS].isna().all(axis=1)].copy()
    return labeled, unlabeled


def macro_auc(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> tuple[float, dict]:
    """Macro AUC-ROC -- the competition metric. Labels with one class are skipped."""
    from sklearn.metrics import roc_auc_score

    per_label = {}
    for c in LABELS:
        t = y_true[c].values
        if len(set(t)) < 2:
            per_label[c] = float("nan")
            continue
        per_label[c] = roc_auc_score(t, y_pred[c].values)
    valid = [v for v in per_label.values() if v == v]
    return (sum(valid) / len(valid) if valid else float("nan")), per_label


def evaluate(extract, train: pd.DataFrame, show_errors: bool = False) -> float:
    """Score an extractor against the 58 gold studies."""
    labeled, _ = gold_split(train)
    preds = pd.DataFrame(
        [extract(t) for t in labeled["Report"]], index=labeled.index
    ).reindex(columns=LABELS).fillna(0.5)

    score, per_label = macro_auc(labeled[LABELS], preds)

    print(f"\n{'label':<18} {'AUC':>6}  {'n_pos':>5}")
    print("-" * 34)
    for c in LABELS:
        auc = per_label[c]
        print(f"{c:<18} {auc:>6.3f}  {int(labeled[c].sum()):>5}")
    print("-" * 34)
    print(f"{'MACRO AUC':<18} {score:>6.3f}   (n={len(labeled)} studies)")
    print("\nReminder: 58 studies is a smoke test, not a validation set. A 0.03 move here")
    print("is noise. Use it to catch extractors that are broken, not to rank good ones.")

    if show_errors:
        _show_errors(labeled, preds)
    return score


def _show_errors(labeled: pd.DataFrame, preds: pd.DataFrame, n: int = 4) -> None:
    """Print the most confidently wrong calls -- usually missed negation."""
    print("\n" + "=" * 70)
    print("MOST CONFIDENT MISTAKES (check these for missed negation)")
    print("=" * 70)
    rows = []
    for idx in labeled.index:
        for c in LABELS:
            err = abs(labeled.loc[idx, c] - preds.loc[idx, c])
            rows.append((err, idx, c, labeled.loc[idx, c], preds.loc[idx, c]))
    rows.sort(reverse=True)
    for err, idx, c, truth, pred in rows[:n]:
        print(f"\n[{c}] truth={truth:.0f} predicted={pred:.2f}")
        print("  " + str(labeled.loc[idx, "Report"])[:400].replace("\n", " "))


# --------------------------------------------------------------------------
# Baseline extractor: multilingual terms + negation window
# --------------------------------------------------------------------------
# Deliberately simple. Its purpose is to establish a floor and to demonstrate that
# negation -- not term coverage -- is what limits this task.

TERMS: dict[str, list[str]] = {
    "ACL": [r"anterior cruciate", r"\bACL\b", r"ligamento cruzado anterior", r"\bLCA\b",
            r"voorste kruisband", r"vordere[sn]? kreuzband", r"ön çapraz", r"ligament croisé antérieur",
            r"предна кръстна", r"πρόσθιο[υς] χιαστ", r"prednj[ai] ukri[žz]en"],
    "MCL": [r"medial collateral", r"\bMCL\b", r"ligamento colateral (medial|interno)",
            r"mediale[sn]? kollateralband", r"iç yan bağ", r"ligament collatéral (médial|interne)",
            r"вътрешн[а-я]* колатерал", r"έσω πλάγιο"],
    "Medial Meniscus": [r"medial meniscus", r"menisco (interno|medial)", r"mediale meniscus",
                        r"innenmeniskus", r"medialer meniskus", r"iç menisküs",
                        r"ménisque (médial|interne)", r"медиалния менискус", r"έσω μηνίσκ",
                        r"medijalni meniskus"],
    "Lateral Meniscus": [r"lateral meniscus", r"menisco (externo|lateral)", r"laterale meniscus",
                         r"aussenmeniskus|außenmeniskus", r"dış menisküs", r"ménisque (latéral|externe)",
                         r"латералния менискус", r"έξω μηνίσκ", r"lateralni meniskus"],
    "Medial OA": [r"medial compartment (degener|osteoarthr)", r"artrosis (femorotibial )?(medial|interna)",
                  r"mediale (gonarthrose|arthrose)", r"iç kompartman", r"медиална гонартроза"],
    "Lateral OA": [r"lateral compartment (degener|osteoarthr)", r"artrosis (femorotibial )?(lateral|externa)",
                   r"laterale (gonarthrose|arthrose)", r"dış kompartman", r"латерална гонартроза"],
    "PF OA": [r"patellofemoral", r"patelofemoral", r"chondromalacia", r"condromalacia",
              r"retropatellar", r"femoropatelar", r"πατελλομηρια"],
    "Effusion": [r"effusion", r"derrame", r"erguss", r"gelenkerguss", r"efüzyon", r"épanchement",
                 r"излив", r"συλλογή", r"izliv", r"vocht in het gewricht|hydrops"],
    "Synovitis": [r"synovitis", r"sinovitis", r"synovitide", r"sinovit", r"синовит", r"υμενίτιδα",
                  r"synovial (thickening|proliferation)", r"sinovijitis"],
    "Baker's": [r"baker", r"popliteal cyst", r"quiste de baker", r"bakerzyste", r"bakerc?yste",
                r"kyste de baker", r"киста на бейкър", r"κύστη baker"],
    "Contusion": [r"contusion", r"bone (marrow )?(oedema|edema)", r"contusión", r"edema óseo",
                  r"knochenmark[öo]dem", r"kemik iliği ödem", r"contusion osseuse",
                  r"костен оток", r"οστικ[όή] οίδημα", r"kontuzij"],
    "Fracture": [r"fracture", r"fractura", r"fraktur", r"kırık", r"фрактура|счупван",
                 r"κάταγμα", r"prijelom", r"breuk"],
}

# Negation cues. Matching these near a term flips it. Nowhere near complete --
# that is the point the evaluation is meant to expose.
NEGATION = [
    r"\bno\b", r"\bnot\b", r"\bnormal\b", r"\bintact\b", r"without", r"absence of",
    r"unremarkable", r"preserved",
    r"\bsin\b", r"no hay", r"normales?", r"íntegro", r"conservad",
    r"\bgeen\b", r"normaal", r"intact",
    r"\bkein", r"unauffällig", r"regelrecht",
    r"\byok\b", r"izlenmedi", r"normal",
    r"\bpas de\b", r"absence",
    r"няма", r"нормал", r"интактн", r"б\.о\.",
    r"δεν ", r"φυσιολογικ",
    r"\bnema\b", r"uredan|uredno",
]

NEG_WINDOW = 90  # characters before a term to scan for a negation cue


def keyword_extract(report: str) -> dict[str, float]:
    """Baseline: term hit, downweighted when a negation cue sits just before it."""
    text = str(report).lower()
    out = {}
    for label, patterns in TERMS.items():
        best = 0.0
        for pat in patterns:
            for m in re.finditer(pat, text, flags=re.IGNORECASE):
                window = text[max(0, m.start() - NEG_WINDOW):m.start()]
                negated = any(re.search(n, window, flags=re.IGNORECASE) for n in NEGATION)
                best = max(best, 0.15 if negated else 0.85)
        out[label] = best if best else 0.25
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True, help="path to train.csv")
    ap.add_argument("--show-errors", action="store_true")
    args = ap.parse_args()

    train = pd.read_csv(args.data)
    labeled, unlabeled = gold_split(train)
    print(f"loaded {len(train)} studies: {len(labeled)} labeled, {len(unlabeled)} unlabeled")
    evaluate(keyword_extract, train, show_errors=args.show_errors)


if __name__ == "__main__":
    main()
