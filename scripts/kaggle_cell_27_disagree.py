# ============================================================
# RSNA Knee — where would a human's hours actually be worth something?
#
# Labelling all 4,349 studies by hand is 50-100 hours and would likely land BELOW
# the LLM extraction, which handles negation and hedging more consistently than a
# person at report 3,000. And no reader beats the coverage ceiling: kneecap
# arthritis is mentioned in only 33% of the studies that have it.
#
# But the readers do not agree with each other everywhere. Where five independent
# tables all say the same thing, a human adds nothing. Where they split, the label
# is genuinely uncertain and adjudication adds real information.
#
# This counts the split, per finding, so an hour of human time can be pointed at
# the studies where it changes something.
#
# CPU, ~2 min. Attach the competition and every label dataset.
# ============================================================
import sys, os
import numpy as np, pandas as pd

CODE = "/kaggle/working/rsna-knee"
!rm -rf $CODE && git clone -q https://github.com/IamWasee/rsna-knee.git $CODE
sys.path.insert(0, f"{CODE}/src")
from config import ID_COL, LABELS
from kaggle_paths import find, competition_file

train = pd.read_csv(competition_file("train.csv"))
gold = train[train[LABELS].notna().all(axis=1)][ID_COL]

# Only tables that are real readers. A table scoring 1.000 against the gold rows
# has the answers pasted in and is not an independent opinion.
from sklearn.metrics import roc_auc_score
g = train[train[LABELS].notna().all(axis=1)].set_index(ID_COL)
readers = {}
for f in find(suffix=".csv"):
    if os.path.getsize(f) > 40e6 or os.path.basename(f).startswith(("train", "test", "sample")):
        continue
    try:
        df = pd.read_csv(f)
    except Exception:
        continue
    if ID_COL not in df.columns or not all(c in df.columns for c in LABELS):
        continue
    df = df.drop_duplicates(subset=[ID_COL])
    sub = df.set_index(ID_COL).reindex(g.index)
    try:
        aucs = [roc_auc_score(g[c].values, pd.to_numeric(sub[c], errors="coerce").fillna(0.5))
                for c in LABELS]
    except Exception:
        continue
    macro = float(np.mean(aucs))
    name = os.path.basename(f)[:-4][:24]
    if macro > 0.995:
        print(f"skip {name}: scores {macro:.3f} on the gold rows -- it contains the answers")
        continue
    readers[name] = df.set_index(ID_COL)
    print(f"reader {name:<26} gold macro {macro:.3f}")

if len(readers) < 3:
    raise SystemExit("need at least 3 independent readers; attach more label datasets")

ids = [i for i in train[ID_COL] if i not in set(gold)]
print(f"\n{len(readers)} readers over {len(ids)} unlabelled studies\n")

print(f"{'finding':<18}{'all agree':>11}{'split':>9}{'split %':>9}   worth a human hour?")
print("-" * 72)
rows = []
for c in LABELS:
    # Each reader's call, as a yes/no at its own median so hard and soft tables
    # are comparable rather than one dominating on scale.
    calls = []
    for nm, df in readers.items():
        v = pd.to_numeric(df[c], errors="coerce").reindex(ids)
        calls.append((v > v.median()).astype(float).where(v.notna()))
    M = pd.concat(calls, axis=1)
    votes = M.sum(axis=1)
    n_read = M.notna().sum(axis=1)
    split = ((votes > 0) & (votes < n_read) & (n_read >= 3))
    pct = split.mean()
    rows.append((c, int(split.sum()), pct))
    flag = "YES -- readers genuinely disagree" if pct > 0.15 else ""
    print(f"{c:<18}{int((~split).sum()):>11}{int(split.sum()):>9}{pct:>8.1%}   {flag}")

print("-" * 72)
tot = sorted(rows, key=lambda r: -r[2])
print(f"\nMost contested: " + ", ".join(f"{c} ({p:.0%})" for c, _, p in tot[:4]))
print(f"Least contested: " + ", ".join(f"{c} ({p:.0%})" for c, _, p in tot[-3:]))
budget = sum(n for _, n, _ in tot[:3])
print(f"\nAdjudicating just the top three findings is {budget} study-decisions.")
print(f"At 20 seconds each that is {budget*20/3600:.1f} hours -- against 50-100 hours")
print("to relabel everything, most of which the readers already agree on.")
