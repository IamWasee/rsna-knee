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

# Osteoarthritis labels are absent here on purpose -- SIDE/OA_TERMS below handle them.
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
# Negation cues, checked within the term's own clause -- see clause() below.
NEGATION = [
    r"\bno\b", r"\bnot\b", r"\bnormal", r"\bintact\b", r"without", r"absence of",
    r"unremarkable", r"preserved", r"\bnegative\b", r"rule[ds]? out", r"\bno evidence",
    r"\bsin\b", r"no hay", r"normales?", r"íntegro", r"conservad", r"sin signos",
    r"sin alteraciones", r"sin evidencia", r"descarta",
    r"\bgeen\b", r"normaal", r"\bintact",
    r"\bkein", r"unauffällig", r"regelrecht", r"\bohne\b",
    r"\byok\b", r"izlenmedi", r"saptanmad", r"görülmedi",
    r"\bpas de\b", r"absence", r"sans ",
    r"няма", r"нормал", r"интактн", r"б\.о\.", r"не се",
    r"δεν ", r"φυσιολογικ", r"ουδεμία",
    r"\bnema\b", r"uredan|uredno", r"bez ",
]

# Osteoarthritis is scored differently: an OA term plus a compartment word in the
# same clause. The compartment is often stated once for several findings
# ("OA femorotibial medial y lateral"), so term-adjacency alone misses most of it.
OA_TERMS = [r"osteoarthr", r"arthrosis", r"artrosis", r"gonartros", r"gonarthros",
            r"arthrose", r"\bOA\b", r"degener", r"chondromalac", r"condromalac",
            r"cartilage (loss|thinning)", r"артроз", r"αρθρίτιδα", r"artroz"]
SIDE = {
    "Medial OA": [r"medial", r"interno", r"innen", r"\biç\b", r"médial", r"медиал",
                  r"έσω", r"medijaln"],
    "Lateral OA": [r"lateral", r"externo", r"aussen|außen", r"\bdış\b", r"латерал",
                   r"έξω", r"lateraln"],
    "PF OA": [r"patellofemoral", r"patelofemoral", r"femoropatel", r"retropatell",
              r"patellar", r"rotula|rótula", r"trochlea", r"πατελ", r"patella"],
}

# Severity qualifiers. The annotated labels apply a threshold the reports do not:
# "Begleitend geringer Gelenkerguss" -- accompanying slight joint effusion -- is
# stated plainly and annotated Effusion=0. Scoring every mention at 0.85 therefore
# mismatches on every mild finding in the corpus. Graded scores also rank better,
# and AUC reads order.
MILD = [
    r"\bmild", r"\bminimal", r"\bsmall\b", r"\bslight", r"\btrace\b", r"\btiny\b",
    r"\bsubtle", r"low[- ]grade", r"grade [12]\b", r"\bearly\b",
    r"\bleve\b", r"m[ií]nim", r"peque[nñ]", r"escas", r"discret", r"ligera",
    r"\bgering", r"\bleicht", r"\bklein", r"diskret",
    r"\blicht", r"\bgeringe",
    r"\bhafif", r"\baz\b", r"minimal",
    r"\bfaible", r"\bl[ée]ger", r"\bpetit", r"discr[eè]t",
    r"минимал", r"\bлек", r"малък", r"незначител",
    r"ήπι", r"μικρ", r"ελαφρ",
    r"\bblag", r"\bmal[ia]\b", r"neznatn",
]
SEVERE = [
    r"\bsevere", r"\bmarked", r"\blarge\b", r"\bextensive", r"\bcomplete",
    r"\bfull[- ]thickness", r"high[- ]grade", r"grade [34]\b", r"\bgross\b",
    r"\bsever[oa]", r"\bgrande", r"\bimportante", r"\bmarcad", r"\bcomplet",
    r"\bausgepr[aä]gt", r"\bdeutlich", r"\bgro[sß]", r"\bvollst[aä]ndig",
    r"\bciddi", r"\bileri", r"\bb[uü]y[uü]k",
    r"\bs[ée]v[eè]re", r"\bimportant", r"\bcomplet",
    r"изразен", r"тежк", r"пълн",
    r"σοβαρ", r"μεγάλ", r"πλήρ",
    r"\bteži", r"\bveliki", r"\bpotpun",
]

CLAUSE_SPLIT = re.compile(r"[.;:\n]|(?<=\s)\d\.\s")


def clause(text: str, pos: int) -> str:
    """The sentence/clause containing position `pos`.

    Negation must be scoped to the clause, not a fixed character window. A window
    reaching backwards across a sentence boundary picks up "The PCL appears intact"
    and wrongly negates the tear described in the next sentence; a
    backwards-only window misses Spanish, where the cue trails the term
    ("menisco interno sin signos de desgarro").
    """
    starts = [0] + [m.end() for m in CLAUSE_SPLIT.finditer(text)]
    ends = [m.start() for m in CLAUSE_SPLIT.finditer(text)] + [len(text)]
    for a, b in zip(starts, ends):
        if a <= pos <= b:
            return text[a:b]
    return text[max(0, pos - 90):pos + 90]


def _negated(fragment: str) -> bool:
    return any(re.search(n, fragment, flags=re.IGNORECASE) for n in NEGATION)


def _grade(fragment: str) -> float:
    """Score a present finding by how emphatically the report states it.

    0.55 for mild -- annotators frequently threshold these to negative -- 0.95 for
    severe, 0.85 for an unqualified mention.
    """
    if any(re.search(m, fragment, flags=re.IGNORECASE) for m in MILD):
        return 0.55
    if any(re.search(v, fragment, flags=re.IGNORECASE) for v in SEVERE):
        return 0.95
    return 0.85


def keyword_extract(report: str) -> dict[str, float]:
    """Term hit, scored within its own clause so negation is read in either direction."""
    text = str(report).lower()
    out = {}

    for label, patterns in TERMS.items():
        if label in SIDE:
            continue  # handled below
        best = 0.0
        for pat in patterns:
            for m in re.finditer(pat, text, flags=re.IGNORECASE):
                frag = clause(text, m.start())
                best = max(best, 0.15 if _negated(frag) else _grade(frag))
        out[label] = best if best else 0.25

    # Osteoarthritis: OA term + compartment word co-occurring in one clause.
    for label, side_pats in SIDE.items():
        best = 0.0
        for pat in OA_TERMS:
            for m in re.finditer(pat, text, flags=re.IGNORECASE):
                frag = clause(text, m.start())
                if not any(re.search(sp, frag, flags=re.IGNORECASE) for sp in side_pats):
                    continue
                best = max(best, 0.15 if _negated(frag) else _grade(frag))
        out[label] = best if best else 0.25

    return out


def borrow_sweep(train: pd.DataFrame, target: str, donor: str) -> None:
    """Does a well-reported label's ranking improve a poorly-reported one?

    Synovitis is mentioned in 14.1% of reports but positive in 46.6% of annotated
    studies, and only 44.4% of its positives are mentioned at all -- so more than
    half are unreachable from text at any extraction quality. Effusion is well
    reported and correlates with it. Blending is done in rank space because AUC
    reads order and the two labels sit on different scales.
    """
    labeled, _ = gold_split(train)
    preds = pd.DataFrame([keyword_extract(t) for t in labeled["Report"]],
                         index=labeled.index).reindex(columns=LABELS)

    r_t = preds[target].rank(pct=True)
    r_d = preds[donor].rank(pct=True)

    print(f"\n{target} borrowing {donor}  (keyword extractor, {len(labeled)} gold)")
    print(f"{'w':>5} {target:>12} {'macro':>8}")
    print("-" * 28)
    for w in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0):
        blended = preds.copy()
        blended[target] = (1 - w) * r_t + w * r_d
        macro, per = macro_auc(labeled[LABELS], blended)
        mark = "   <- donor alone" if w == 1.0 else ""
        print(f"{w:>5.1f} {per[target]:>12.3f} {macro:>8.3f}{mark}")
    print("-" * 28)
    print("w=0 is the extractor untouched; w=1 is the donor's ranking substituted")
    print("wholesale. A real effect shows a smooth hump, not a single spike. With")
    print(f"{int(labeled[target].sum())} positives in 58 studies, treat anything under 0.05 as noise.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True, help="path to train.csv")
    ap.add_argument("--show-errors", action="store_true")
    ap.add_argument("--borrow", nargs=2, metavar=("TARGET", "DONOR"),
                    help="sweep blending DONOR's ranking into TARGET, e.g. Synovitis Effusion")
    args = ap.parse_args()

    train = pd.read_csv(args.data)
    labeled, unlabeled = gold_split(train)
    print(f"loaded {len(train)} studies: {len(labeled)} labeled, {len(unlabeled)} unlabeled")
    evaluate(keyword_extract, train, show_errors=args.show_errors)
    if args.borrow:
        borrow_sweep(train, args.borrow[0], args.borrow[1])


if __name__ == "__main__":
    main()
