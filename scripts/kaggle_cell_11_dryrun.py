# ============================================================
# RSNA Knee — Submission DRY RUN
#
# Proves the whole submission path works, using RANDOM weights. The scores are
# meaningless by design -- what is being tested is the plumbing: test-split
# preprocessing, checkpoint loading, the fold ensemble, and the exact CSV format.
#
# Run this while training is still going. CPU is fine, ~3 minutes.
# Attach only the competition.
# ============================================================
!pip install -q timm

import sys, glob, os, time, json
CODE = "/kaggle/working/rsna-knee"
!rm -rf $CODE && git clone -q https://github.com/IamWasee/rsna-knee.git $CODE
sys.path.insert(0, f"{CODE}/src")

import torch
from config import LABELS
from model import KneeModel

# A checkpoint shaped exactly like train.py writes, but never trained.
os.makedirs("/kaggle/working/fake_weights", exist_ok=True)
for fold in range(2):                       # two folds -> also tests the ensemble
    m = KneeModel("resnet34", len(LABELS), pretrained=False)
    torch.save({"model": m.state_dict(), "fold": fold, "gold_auc": 0.5,
                "per_label": {c: 0.5 for c in LABELS},
                "args": {"backbone": "resnet34", "n_series": 3,
                         "n_slices": 12, "size": 224}},
               f"/kaggle/working/fake_weights/fold{fold}.pt")
print("wrote 2 random checkpoints")

# 1. Preprocess the test split (3 public studies; the real one is hidden).
t0 = time.time()
!python $CODE/src/preprocess.py --out /kaggle/working/test_cache --split test --workers 4
print(f"preprocess: {time.time()-t0:.0f}s")

# 2. Inference + strict format validation.
!python $CODE/src/infer.py --cache /kaggle/working/test_cache \
    --weights /kaggle/working/fake_weights --out /kaggle/working/submission.csv

# 3. Confirm the file is exactly what the scorer expects.
import pandas as pd
from submission import validate
from paths import data_root

sub = pd.read_csv("/kaggle/working/submission.csv")
sample = pd.read_csv(data_root() / "sample_submission.csv")
validate(sub, expected_ids=sample["StudyInstanceUID"])
print("\nFORMAT VALID")
print(sub.to_string())
print(f"\ncolumns match sample exactly: {list(sub.columns) == list(sample.columns)}")
