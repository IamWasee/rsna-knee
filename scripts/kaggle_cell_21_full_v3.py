# ============================================================
# RSNA Knee — both heads, full 5 folds, then blended per label
#
# Fold 0 said slot+focal beats shared+GAP by 0.007 overall (0.804 vs 0.797), but
# not on every label: shared+GAP owns ACL by 0.033 while slot+focal owns Medial
# Meniscus by 0.043 and Fracture by 0.037. The metric is a macro average of twelve
# independent rankings, so nothing requires one head to win all twelve -- and rank
# averaging is scale-free, so two models that disagree can be combined without any
# calibration step.
#
# So: train both to full folds on identical splits, then measure the blend instead
# of guessing at it. Three candidates get scored on the same out-of-fold data:
#     each head alone, the 50/50 rank blend, and a per-label choice of weight.
#
# The per-label weights are chosen on this same OOF data, which flatters them --
# that is what the honest/optimistic split at the bottom is for. Take the 50/50
# blend as the number you can trust and the per-label one as an upper bound.
#
# Attach:
#   1. the competition
#   2. Notebook Output -> cache-v3
#   3. Dataset -> stevenleehans/rsna-knee-llm-report-labels
#   4. Model   -> metaresearch/dinov2/PyTorch/small/1
#
# GPU on, Internet on. ~7h of a 12h session. Checkpoints land per fold, so a
# timeout costs one fold rather than the run.
# ============================================================
!pip install -q timm transformers

import sys, os, glob, time
import numpy as np
import pandas as pd

CODE = "/kaggle/working/rsna-knee"
!rm -rf $CODE && git clone -q https://github.com/IamWasee/rsna-knee.git $CODE
sys.path.insert(0, f"{CODE}/src")
from kaggle_paths import find, describe
from config import LABELS

v3 = [f for f in find(suffix=".npy") if "cache_v3" in f]
if not v3:
    describe(); raise SystemExit("attach the cache-v3 notebook output")
CACHE = os.path.dirname(v3[0])

lab = find(filename="llm_labels_v4_blend.csv")
if not lab:
    describe(); raise SystemExit("attach stevenleehans/rsna-knee-llm-report-labels")

dino = [os.path.dirname(p) for p in find(filename="config.json") if "dinov2" in p.lower()]
if not dino:
    describe(); raise SystemExit("attach metaresearch/dinov2/PyTorch/small/1")

assert os.path.exists(f"{CACHE}/study_meta.csv"), "no study_meta.csv -- folds would not be scanner-grouped"
print(f"cache: {CACHE}\nlabels: {lab[0]}")

COMMON = (f'--cache "{CACHE}" --labels "{lab[0]}" --backbone "dinov2:{dino[0]}" '
          f'--size 288 --folds 5 --epochs 12 --batch 8 '
          f'--lr 1e-3 --lr-backbone 8e-6 --unfreeze-last 6 --weight-decay 0.02')

t0 = time.time()
print("\n" + "=" * 62 + "\nHEAD A: shared + GAP, 5 folds\n" + "=" * 62)
!python $CODE/src/train.py {COMMON} --head shared --pool gap --out /kaggle/working/w_shared
print(f"\nelapsed {(time.time()-t0)/3600:.2f} h")

print("\n" + "=" * 62 + "\nHEAD B: slot + focal, 5 folds\n" + "=" * 62)
!python $CODE/src/train.py {COMMON} --head slot --pool focal --out /kaggle/working/w_slot
print(f"\nelapsed {(time.time()-t0)/3600:.2f} h")

# ---------------------------------------------------------------- the blend
from sklearn.metrics import roc_auc_score

A = pd.read_csv("/kaggle/working/w_shared/oof.csv")
B = pd.read_csv("/kaggle/working/w_slot/oof.csv")
ID = A.columns[0]
B = B.set_index(ID).reindex(A[ID]).reset_index()
assert (A[ID].values == B[ID].values).all(), "the two heads scored different studies"
assert (A["fold"].values == B["fold"].values).all(), \
    "the two heads used different fold splits -- the blend would be meaningless"
print(f"\n{len(A)} out-of-fold studies, {A['fold'].nunique()} folds, splits identical")

def ranks(df, c):
    """Rank within fold. Across folds the two heads are differently calibrated, so
    pooling raw probabilities compares a 0.6 from one fold against a 0.6 from
    another. Ranking inside each fold removes that, and AUC only sees order."""
    out = np.empty(len(df))
    for f in df["fold"].unique():
        m = (df["fold"] == f).values
        out[m] = pd.Series(df.loc[m, c].values).rank(pct=True).values
    return out

def auc(y, p):
    keep = ~np.isnan(y)
    t = (y[keep] > 0.5).astype(int)
    return roc_auc_score(t, p[keep]) if len(set(t)) > 1 else np.nan

rows = []
for c in LABELS:
    y = pd.to_numeric(A[f"{c}__y"], errors="coerce").values
    ra, rb = ranks(A, c), ranks(B, c)
    a, b = auc(y, ra), auc(y, rb)
    half = auc(y, 0.5 * ra + 0.5 * rb)
    grid = {w: auc(y, w * ra + (1 - w) * rb) for w in np.arange(0, 1.01, 0.1)}
    w_best = max(grid, key=grid.get)
    rows.append((c, a, b, half, grid[w_best], w_best))

hdr = f"{'label':<18}{'shared':>8}{'slot':>8}{'50/50':>8}{'tuned':>8}{'w':>6}"
print("\n" + hdr); print("-" * len(hdr))
for c, a, b, h, t, w in rows:
    print(f"{c:<18}{a:>8.3f}{b:>8.3f}{h:>8.3f}{t:>8.3f}{w:>6.1f}")
print("-" * len(hdr))
m = lambda i: np.nanmean([r[i] for r in rows])
print(f"{'MACRO':<18}{m(1):>8.3f}{m(2):>8.3f}{m(3):>8.3f}{m(4):>8.3f}")
print(f"\n50/50 blend vs the better head alone: {m(3) - max(m(1), m(2)):+.4f}")
print(f"per-label weights add a further {m(4) - m(3):+.4f}, but they were CHOSEN on")
print("this data -- expect roughly half of that to survive on the leaderboard.")

w = {c: r[5] for c, r in zip(LABELS, rows)}
pd.DataFrame([{"label": c, "w_shared": v} for c, v in w.items()]).to_csv(
    "/kaggle/working/blend_weights.csv", index=False)
print("\nwrote /kaggle/working/blend_weights.csv")
print(f"total elapsed {(time.time()-t0)/3600:.2f} h")
