# ============================================================
# RSNA Knee — THE SUBMISSION NOTEBOOK
#
# This is the one you submit. Rules it must respect:
#   - Internet OFF (Settings -> Internet)
#   - GPU on
#   - everything comes from attached inputs; nothing is downloaded
#
# Attach exactly TWO inputs:
#   1. the competition
#   2. Notebook Output -> your TRAINING notebook
#      (that one output carries both the .pt checkpoints AND src/, because the
#       training cell clones the repo into /kaggle/working, which Kaggle saves)
#
# Delete Kaggle's default os.walk starter cell.
# ============================================================
import sys, glob, os, time
from pathlib import Path
t0 = time.time()

# No pip install -- there is no internet. Verify what we need is already here.
missing = []
for mod in ("torch", "timm", "pydicom", "cv2", "pandas", "numpy", "sklearn"):
    try:
        __import__(mod)
    except ImportError:
        missing.append(mod)
if missing:
    raise SystemExit(f"NOT PREINSTALLED: {missing} -- cannot pip install offline.\n"
                     "Package these as a Kaggle Dataset and install from it.")
print("all required packages preinstalled")

# Find the code and the weights among the attached inputs.
#
# NOT a recursive glob over /kaggle/input: that walks the whole competition
# dataset -- hundreds of thousands of DICOM files -- and measured 421 seconds to
# locate two paths. Search the notebook-output trees only, bounded in depth.
def find(pattern: str, max_depth: int = 4) -> list[str]:
    roots = glob.glob("/kaggle/input/notebooks/*/*") + glob.glob("/kaggle/input/*")
    hits = []
    for r in roots:
        if "rsna-knee-abnormality-detection" in r:      # skip the DICOM archive
            continue
        for d in range(1, max_depth + 1):
            hits += glob.glob(os.path.join(r, *(["*"] * (d - 1)), pattern))
    return sorted(set(hits))

found = find("infer.py") + find("src/infer.py")
if not found:
    raise SystemExit("src/ not found -- attach the training notebook's output.")
CODE = str(Path(found[0]).parents[1])
sys.path.insert(0, f"{CODE}/src")

WEIGHTS = find("*.pt")
if not WEIGHTS:
    raise SystemExit("no .pt checkpoints found -- attach the training notebook's output.")
WDIR = str(Path(WEIGHTS[0]).parent)
print(f"code:    {CODE}")
print(f"weights: {WDIR}  ({len(WEIGHTS)} folds)")

CACHE = "/kaggle/working/test_cache"

# 1. Preprocess the hidden test set. Measured at 2.73 studies/sec with 4 workers,
#    so even 20,000 studies is ~2h of the 9h budget.
!python $CODE/src/preprocess.py --out $CACHE --split test --workers 4
print(f"preprocess: {(time.time()-t0)/60:.1f} min")

# 2. Predict and write submission.csv at the path Kaggle expects.
!python $CODE/src/infer.py --cache $CACHE --weights $WDIR --out /kaggle/working/submission.csv

import pandas as pd
sub = pd.read_csv("/kaggle/working/submission.csv")
print(sub.shape)
print(sub.head())
print(f"\ntotal: {(time.time()-t0)/60:.1f} min of the 540 available")
