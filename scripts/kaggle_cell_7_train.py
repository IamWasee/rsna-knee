# ============================================================
# RSNA Knee — Step 7: train the image model
#
# Attach THREE inputs:
#   1. the competition (rsna-knee-abnormality-detection)
#   2. Notebook Output -> the preprocessing notebook ("notebooka") -> cache/
#   3. Notebook Output -> the labelling notebook -> report_labels.csv
#
# GPU on. Delete Kaggle's default os.walk starter cell.
# ============================================================
!pip install -q timm

import sys, glob, os
CODE = "/kaggle/working/rsna-knee"
!rm -rf $CODE && git clone -q https://github.com/IamWasee/rsna-knee.git $CODE
sys.path.insert(0, f"{CODE}/src")

# Locate the attached inputs rather than hardcoding dataset names.
#
# Several notebooks may expose a report_labels.csv -- an earlier run stopped at
# 3,520 studies before the full one reached 4,349. Take the one with the most
# rows; silently training on the stale file would drop 800 studies and look fine.
import pandas as pd
found = glob.glob("/kaggle/input/**/report_labels.csv", recursive=True)
for f in found:
    print(f"  candidate: {f}  ({len(pd.read_csv(f))} studies)")
LABELS = max(found, key=lambda f: len(pd.read_csv(f))) if found else None

cands  = glob.glob("/kaggle/input/**/*.npy", recursive=True)
CACHE  = os.path.dirname(cands[0]) if cands else None

print("\nusing labels:", LABELS)
print("cache: ", CACHE, f"({len(cands)} volumes)" if cands else "")
if not LABELS or not CACHE:
    raise SystemExit("Attach both notebook outputs -- see the header above.")

# Measured: sharpening lifted GOLD 0.653 -> 0.725, best at epoch 5 (6 and 7 added
# nothing while derived-val kept climbing -- fitting the extractor, not anatomy).
#
# Full 5-fold run at 6 epochs: ~2.5h, fits one session. Produces the checkpoints
# the submission notebook needs.
!python $CODE/src/train.py --cache "$CACHE" --labels "$LABELS" \
    --epochs 6 --batch 4 --backbone resnet34 --out /kaggle/working/weights

import glob
print("\ncheckpoints:", sorted(glob.glob("/kaggle/working/weights/*.pt")))
print("Save this notebook's output, then attach it to the submission notebook.")
