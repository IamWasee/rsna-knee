# ============================================================
# RSNA Knee — blend every arm we have, and pick the set by measurement
#
# The plane specialists are complementary in exactly the way the anatomy says
# they should be, which is the reason to expect the blend to pay:
#
#   label              sagittal   coronal    who sees it
#   Baker's              0.91      0.77      popliteal fossa -- BEHIND the knee
#   Lateral OA           0.85      0.81      outer compartment
#   Lateral Meniscus     0.75      0.72
#   MCL                  0.73      0.77      inner edge -- only the front view
#   Medial Meniscus      0.83      0.86
#   ACL                  0.75      0.78
#
# Neither is better; they are better at different things. That is what an
# ensemble is for, and it is not what four encoders on the same four sequences
# gave us -- those agreed with each other at 0.83 and the third added +0.002.
#
# Greedy forward selection on the blend, over the plane arms AND the encoder
# families together, so the two directions compete on one measurement instead of
# my judgement. Ranks are per label: AUC reads order only, and two models trained
# on different inputs are not calibrated to each other.
#
# CPU only -- this reads oof.csv files, it trains nothing.
# Attach: allslice-sag, allslice-cor, allslice-ax (when it exists), families.
# ============================================================
import sys, os, itertools
import numpy as np, pandas as pd

CODE = "/kaggle/working/rsna-knee"
!rm -rf $CODE && git clone -q https://github.com/IamWasee/rsna-knee.git $CODE
sys.path.insert(0, f"{CODE}/src")
from kaggle_paths import find, describe
from config import LABELS
from sklearn.metrics import roc_auc_score

paths = {}
for p in sorted(find(filename="oof.csv")):
    # .../allslice_sag/oof.csv -> "sag"; .../arm_convnext/oof.csv -> "convnext"
    d = os.path.basename(os.path.dirname(p))
    tag = d.replace("allslice_", "plane-").replace("arm_", "")
    if tag not in paths:
        paths[tag] = p
if len(paths) < 2:
    describe(); raise SystemExit(f"found {len(paths)} oof.csv; attach the arms")
print(f"{len(paths)} arms: {list(paths)}\n")

# Every arm must have scored the SAME studies, or the blend is comparing
# different exams and the number means nothing.
frames = {t: pd.read_csv(p) for t, p in paths.items()}
ID = list(frames.values())[0].columns[0]
common = set.intersection(*(set(f[ID]) for f in frames.values()))
base = list(frames.values())[0]
base = base[base[ID].isin(common)].sort_values(ID).reset_index(drop=True)
print(f"{len(common)} studies scored by all arms "
      f"(individual arms: {[len(f) for f in frames.values()]})")
if len(common) < 0.9 * max(len(f) for f in frames.values()):
    print("WARNING: arms disagree on which studies they scored -- check the folds")

Y = {c: pd.to_numeric(base[f"{c}__y"], errors="coerce").values for c in LABELS}
R = {}
for t, f in frames.items():
    f = f[f[ID].isin(common)].set_index(ID).reindex(base[ID]).reset_index()
    R[t] = {c: pd.Series(f[c].values).rank(pct=True).values for c in LABELS}

def per_label(tags):
    out = {}
    for c in LABELS:
        y = Y[c]; keep = ~np.isnan(y)
        p = np.mean([R[t][c] for t in tags], axis=0)
        tr = (y[keep] > 0.5).astype(int)
        out[c] = roc_auc_score(tr, p[keep]) if len(set(tr)) > 1 else np.nan
    return out

def macro(tags):
    v = [x for x in per_label(tags).values() if x == x]
    return float(np.mean(v))

print("\n" + "=" * 66 + "\nEACH ARM ALONE\n" + "=" * 66)
singles = sorted(((macro([t]), t) for t in paths), reverse=True)
for s, t in singles:
    print(f"  {t:<16}{s:.4f}")

tags = [t for _, t in singles]
print("\n" + "=" * 66 + "\nAGREEMENT (1.000 = same mistakes)\n" + "=" * 66)
print(f"{'':<16}" + "".join(f"{t[:10]:>12}" for t in tags))
for a in tags:
    print(f"{a:<16}" + "".join(
        f"{np.mean([np.corrcoef(R[a][c], R[b][c])[0,1] for c in LABELS]):>12.3f}"
        for b in tags))

print("\n" + "=" * 66 + "\nGREEDY SELECTION\n" + "=" * 66)
chosen, rest, best_set, best_score = [tags[0]], list(tags[1:]), [tags[0]], macro([tags[0]])
print(f"  1. {tags[0]:<16} {best_score:.4f}")
while rest:
    prev = macro(chosen)
    gain, t = max((macro(chosen + [x]), x) for x in rest)
    print(f"  {len(chosen)+1}. + {t:<14} {gain:.4f}  ({gain - prev:+.4f})")
    chosen.append(t); rest.remove(t)
    if gain > best_score:
        best_score, best_set = gain, list(chosen)

print(f"\nBEST SET: {best_set}  ->  {best_score:.4f}")
print(f"  against the best single arm ({tags[0]}): {best_score - macro([tags[0]]):+.4f}")

print("\n" + "=" * 66 + "\nPER LABEL, best set vs best single\n" + "=" * 66)
a, b = per_label([tags[0]]), per_label(best_set)
print(f"{'label':<18}{tags[0][:10]:>11}{'blend':>9}{'diff':>8}")
for c in LABELS:
    print(f"{c:<18}{a[c]:>11.3f}{b[c]:>9.3f}{b[c]-a[c]:>+8.3f}")

print("\nFold 0 reads about +0.01 optimistic against a 5-fold pooled number, and")
print("our held-out-to-leaderboard gap has measured ~+0.06 on two submissions.")
print("So treat this as a direction to confirm across folds, not a forecast.")
