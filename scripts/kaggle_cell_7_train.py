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

# Measured: 305s/epoch, fold0 2 epochs -> GOLD 0.653. The loop works.
#
# One fold, 8 epochs, WITH label sharpening (~45 min). The question this answers
# is narrow: does sharpening beat the 0.653 baseline? Labels were compressed
# toward 0.5, which caps the gradient BCE can provide.
!python $CODE/src/train.py --cache "$CACHE" --labels "$LABELS" \
    --only-fold 0 --epochs 8 --batch 4 --backbone resnet34 \
    --out /kaggle/working/weights

# Control, if you want the comparison rather than my reasoning (~45 min):
# !python $CODE/src/train.py --cache "$CACHE" --labels "$LABELS" \
#     --only-fold 0 --epochs 8 --batch 4 --backbone resnet34 --no-sharpen \
#     --out /kaggle/working/weights_nosharp

# Full 5-fold run once a configuration is chosen. At 305s/epoch that is ~3.5h
# for 8 epochs x 5 folds -- it fits in one 9h session.
# !python $CODE/src/train.py --cache "$CACHE" --labels "$LABELS" \
#     --epochs 8 --batch 4 --backbone resnet34 --out /kaggle/working/weights

# Then save /kaggle/working/weights as a Kaggle Dataset -- the submission
# notebook is offline and can only get weights from an attached dataset.
