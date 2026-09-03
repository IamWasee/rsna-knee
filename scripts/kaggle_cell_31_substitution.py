# ============================================================
# RSNA Knee — how often is a slot filled with the wrong sequence?
#
# pick_series assigns one series per slot by (plane, fluid-sensitive). When a
# study has no match it falls back to "whatever the study actually has" -- so a
# knee with no coronal fluid-sensitive series gets something else in slot 1,
# often a second sagittal.
#
# That matters because the model carries an ANATOMY PRIOR: SLOT_PRIOR tells each
# diagnosis which slots to attend to, on the assumption that slot 1 is coronal.
# For a substituted study the prior points at the wrong sequence. Nothing records
# it and nothing has ever counted it -- the same silent-substitution shape as the
# laterality flip and the corner-versus-centre crop, both of which were real bugs.
#
# Answered from train_series.csv alone. No DICOM reading, no GPU. ~1 min.
# ============================================================
import sys, os
import pandas as pd, numpy as np

CODE = "/kaggle/working/rsna-knee"
!rm -rf $CODE && git clone -q https://github.com/IamWasee/rsna-knee.git $CODE
sys.path.insert(0, f"{CODE}/src")
from config import ID_COL, SERIES_COL
from preprocess import SLOTS
from kaggle_paths import competition_file

series = pd.read_csv(competition_file("train_series.csv"))
print(f"{len(series)} series over {series[ID_COL].nunique()} studies")
print(f"slot scheme: {SLOTS}\n")

have = {}
for i, (plane, fluid) in enumerate(SLOTS):
    m = series[(series["Anatomical_Plane"] == plane)
               & (series["Fluid_Sensitive"] == fluid)]
    have[i] = set(m[ID_COL])

studies = series.groupby(ID_COL).size()
n = len(studies)
name = lambda i: f"{SLOTS[i][0]}{'-fluid' if SLOTS[i][1] else '-T1'}"

print(f"{'slot':<6}{'sequence':<20}{'present':>9}{'missing':>9}{'missing %':>11}")
print("-" * 55)
missing_counts = []
for i in range(len(SLOTS)):
    miss = n - len(have[i])
    missing_counts.append(miss)
    print(f"{i:<6}{name(i):<20}{len(have[i]):>9}{miss:>9}{miss/n:>10.1%}")
print("-" * 55)

# How many slots does each study actually have? The rest are substituted with a
# spare series, or left as zeros when there is no spare.
per_study = pd.Series(0, index=studies.index)
for i in range(len(SLOTS)):
    per_study += studies.index.isin(list(have[i])).astype(int)
n_series = series.groupby(ID_COL)[SERIES_COL].nunique()

print(f"\n{'slots matched':<16}{'studies':>9}{'share':>9}")
for k in range(len(SLOTS) + 1):
    c = int((per_study == k).sum())
    if c:
        print(f"{k}/{len(SLOTS)}{'':<12}{c:>9}{c/n:>9.1%}")

sub = (per_study < len(SLOTS))
spare = (n_series.reindex(per_study.index).fillna(0) > per_study)
filled = sub & spare
blank = sub & ~spare
print(f"\n{int(sub.sum())} studies ({sub.mean():.1%}) are missing at least one slot.")
print(f"  {int(filled.sum())} ({filled.mean():.1%}) get a SPARE series in its place --")
print(f"      the anatomy prior then points at the wrong sequence for these.")
print(f"  {int(blank.sum())} ({blank.mean():.1%}) get an all-zero slot --")
print(f"      the attention layer weighs a blank image as if it were evidence.")

print("\n" + "=" * 62)
worst = int(np.argmax(missing_counts))
if sub.mean() > 0.15:
    print(f"VERDICT: this is common ({sub.mean():.0%}), so it is worth fixing.")
    print(f"{name(worst)} is missing most often ({missing_counts[worst]/n:.0%}).")
    print("Two fixes, both cheap: record which slots were substituted so the head")
    print("can mask them, and drop the anatomy prior for a substituted slot rather")
    print("than pointing it at a sequence that is not there.")
else:
    print(f"VERDICT: rare ({sub.mean():.1%}). Real, but not where the points are.")
