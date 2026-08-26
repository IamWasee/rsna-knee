# ============================================================
# RSNA Knee — Step 9: the SUBMISSION notebook
#
# This is the one that gets submitted. Rules for it:
#   - Internet MUST be OFF
#   - everything it needs comes from attached datasets
#   - 9 hours total, INCLUDING preprocessing the hidden test set from raw DICOM
#
# Attach: the competition, plus your weights dataset (from step 7).
# Do NOT attach the train cache -- the test studies are different studies.
# ============================================================
import sys, glob, os, time
from pathlib import Path
t0 = time.time()

# No pip install: there is no internet. Kaggle preinstalls torch, timm, pydicom, cv2.
CODE = "/kaggle/working/rsna-knee"
# The repo is not reachable offline either -- attach it as a dataset, or paste
# src/ into the notebook. This path assumes a dataset named rsna-knee-code.
CODE_DS = glob.glob("/kaggle/input/**/src/infer.py", recursive=True)
if CODE_DS:
    CODE = str(Path(CODE_DS[0]).parents[1])
sys.path.insert(0, f"{CODE}/src")
print("code:", CODE)

WEIGHTS = glob.glob("/kaggle/input/**/*.pt", recursive=True)
print(f"{len(WEIGHTS)} checkpoints found")
WDIR = str(Path(WEIGHTS[0]).parent)

CACHE = "/kaggle/working/test_cache"

# 1. Preprocess the hidden test set. This is inside the 9 hours, so it is the
#    part most likely to blow the budget -- the public test set is 3 studies but
#    the real one is not.
!python $CODE/src/preprocess.py --out $CACHE --split test --workers 4
print(f"preprocess: {(time.time()-t0)/60:.1f} min")

# 2. Predict and write submission.csv.
!python $CODE/src/infer.py --cache $CACHE --weights $WDIR --out /kaggle/working/submission.csv

import pandas as pd
sub = pd.read_csv("/kaggle/working/submission.csv")
print(sub.shape)
print(sub.head())
print(f"\ntotal: {(time.time()-t0)/60:.1f} min of the 540 available")
