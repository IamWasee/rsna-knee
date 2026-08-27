# ============================================================
# RSNA Knee — Retrain on preprocessing v2 + regraded labels
#
# Attach FOUR inputs:
#   1. the competition
#   2. Notebook Output -> the v2 preprocessing notebook (cache_v2)
#   3. Notebook Output -> the labelling notebook (model_preds_all.csv)
#   4. nothing else needed -- labels are regenerated here, not reused
#
# GPU on, Internet on. Delete Kaggle's os.walk starter cell.
# ============================================================
!pip install -q timm

import sys, glob, os
CODE = "/kaggle/working/rsna-knee"
!rm -rf $CODE && git clone -q https://github.com/IamWasee/rsna-knee.git $CODE
sys.path.insert(0, f"{CODE}/src")

import pandas as pd
TRAIN = next(iter(glob.glob("/kaggle/input/**/train.csv", recursive=True)))

# Prefer the v2 cache. Both may be attached and they have different shapes, so
# picking the wrong one fails at the first batch rather than silently.
npys = glob.glob("/kaggle/input/**/*.npy", recursive=True)
v2 = [f for f in npys if "cache_v2" in f]
CACHE = os.path.dirname((v2 or npys)[0])
print(f"cache: {CACHE}  ({len([f for f in npys if os.path.dirname(f) == CACHE])} volumes)")
if "cache_v2" not in CACHE:
    print("  WARNING: this is the v1 cache. Attach the v2 preprocessing notebook.")

# Regenerate labels with the current extractor. The saved report_labels.csv predates
# severity grading, which lifted keyword macro AUC from 0.707 to 0.725 -- it costs
# seconds to redo and the model can only be as good as its targets.
PREDS = max(glob.glob("/kaggle/input/**/model_preds_all.csv", recursive=True),
            key=lambda f: len(pd.read_csv(f)))
print(f"model preds: {PREDS} ({len(pd.read_csv(PREDS))} studies)")

!python $CODE/src/ensemble.py --data "$TRAIN" --model-preds "$PREDS" \
    --method mean --out /kaggle/working/report_labels_v2.csv

LABELS = "/kaggle/working/report_labels_v2.csv"

# v2 feeds 12 encoder inputs per study (4 slots x 3 triplets) against v1's 36,
# so each step is cheaper despite the larger images -- batch 8 fits.
!python $CODE/src/train.py --cache "$CACHE" --labels "$LABELS" \
    --epochs 6 --batch 8 --backbone resnet34 --out /kaggle/working/weights_v2

print("\ncheckpoints:", sorted(glob.glob("/kaggle/working/weights_v2/*.pt")))
print("Save this notebook's output, then attach it to the submission notebook.")
