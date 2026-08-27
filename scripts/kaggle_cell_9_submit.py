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
# os.walk with pruning, not glob(recursive=True): the latter descends the whole
# competition dataset -- hundreds of thousands of DICOM files -- and measured 421
# seconds. Pruning that one directory keeps it complete and fast.
def find(filename: str, suffix: str = "") -> list[str]:
    hits = []
    for root, dirs, files in os.walk("/kaggle/input"):
        dirs[:] = [d for d in dirs
                   if "rsna-knee-abnormality-detection" not in os.path.join(root, d)]
        for f in files:
            if (f == filename) or (suffix and f.endswith(suffix)):
                hits.append(os.path.join(root, f))
    return sorted(hits)


found = find("infer.py")
WEIGHTS = find("", ".pt")

if not found or not WEIGHTS:
    print("WHAT IS ACTUALLY ATTACHED:")
    for root, dirs, files in os.walk("/kaggle/input"):
        depth = root.count(os.sep) - 2
        if depth > 3 or "rsna-knee-abnormality-detection" in root:
            dirs[:] = []
            continue
        print("  " * depth + os.path.basename(root) + "/")
        for f in files[:5]:
            print("  " * (depth + 1) + f)
    raise SystemExit(
        f"missing: {'src/infer.py ' if not found else ''}"
        f"{'checkpoints' if not WEIGHTS else ''}\n"
        "Attach: Notebook Output -> your 5-fold TRAINING notebook. Its output holds\n"
        "both weights/*.pt and the cloned rsna-knee/src/."
    )

CODE = str(Path(found[0]).parents[1])
sys.path.insert(0, f"{CODE}/src")
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
