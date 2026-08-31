# ============================================================
# RSNA Knee — Which report labels are best?
#
# Three public LLM-derived label sets exist alongside ours. All are scored the same
# way: macro AUC against the 58 radiologist-annotated studies. Labels cap everything
# downstream, so this is worth measuring rather than taking on recommendation.
#
# CPU, ~1 min. Attaches the competition and the three public datasets.
# ============================================================
import sys, glob, os
import pandas as pd, numpy as np

CODE = "/kaggle/working/rsna-knee"
!rm -rf $CODE && git clone -q https://github.com/IamWasee/rsna-knee.git $CODE
sys.path.insert(0, f"{CODE}/src")
from config import ID_COL, LABELS
from report_labels import keyword_extract
from sklearn.metrics import roc_auc_score

TRAIN = next(iter(glob.glob("/kaggle/input/competitions/**/train.csv", recursive=True)))
train = pd.read_csv(TRAIN)
gold = train[train[LABELS].notna().all(axis=1)].set_index(ID_COL)
print(f"{len(gold)} annotated studies\n")


def score(name, df):
    """Macro AUC of a label table against the annotated studies."""
    if ID_COL not in df.columns:
        print(f"{name:<44} no {ID_COL} column; columns: {list(df.columns)[:6]}")
        return
    df = df.set_index(ID_COL)
    cols = [c for c in LABELS if c in df.columns]
    if len(cols) < len(LABELS):
        print(f"{name:<44} only {len(cols)}/12 label columns")
        if not cols:
            return
    sub = df.reindex(gold.index)
    aucs, per = [], {}
    for c in cols:
        y, p = gold[c].values, pd.to_numeric(sub[c], errors="coerce").values
        ok = ~np.isnan(p)
        if ok.sum() < 20 or len(set(y[ok])) < 2:
            continue
        a = roc_auc_score(y[ok], p[ok]); aucs.append(a); per[c] = a
    cov = sub[cols[0]].notna().mean() if cols else 0
    print(f"{name:<44} {np.mean(aucs):.3f}   ({len(aucs)} labels, {cov:.0%} coverage)")
    return per


print(f"{'label source':<44} {'macro AUC'}")
print("-" * 70)

# Ours, computed on the spot.
kw = pd.DataFrame([keyword_extract(t) for t in train.loc[gold.index.map(
    lambda i: train[train[ID_COL] == i].index[0]), "Report"]], index=gold.index)
kw.insert(0, ID_COL, gold.index)
ours = score("ours (rule-based, in this repo)", kw.reset_index(drop=True))

# Every public label CSV that is attached.
for f in sorted(glob.glob("/kaggle/input/**/*.csv", recursive=True)):
    if "/competitions/" in f:
        continue
    try:
        df = pd.read_csv(f)
    except Exception as e:
        print(f"{os.path.basename(f):<44} unreadable: {e}")
        continue
    score(f"{os.path.basename(os.path.dirname(f))}/{os.path.basename(f)}"[:43], df)

print("-" * 70)
print("58 studies with 9-35 positives per label: differences under ~0.05 are noise.")
print("A label set only helps if it is better where OURS is weak -- check per-label,")
print("not just the macro, and consider rank-averaging two sources rather than")
print("replacing one with the other.")
