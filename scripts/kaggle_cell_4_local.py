# ============================================================
# RSNA Knee — Step 4: report -> labels, FREE, on Kaggle's GPU
# No API, no cost. Settings -> Accelerator: GPU T4 x2 (or P100).
# Internet ON only to download the model weights.
# ============================================================
!pip install -q -U transformers accelerate bitsandbytes

import os, sys, glob

CODE = "/kaggle/working/rsna-knee"
!rm -rf $CODE && git clone -q https://github.com/IamWasee/rsna-knee.git $CODE
sys.path.insert(0, f"{CODE}/src")

TRAIN = next(iter(glob.glob("/kaggle/input/**/train.csv", recursive=True)), None)
if not TRAIN:
    raise SystemExit("train.csv not found -- attach the competition under Input.")
print("train.csv:", TRAIN)

!nvidia-smi --query-gpu=name,memory.total --format=csv

# 1. FREE FLOOR FIRST -- no GPU, no model, ~2 seconds. This is the number the
#    model has to beat to be worth running at all.
!python $CODE/src/report_labels.py --data "$TRAIN" --show-errors

# 2. Open-weights model on the 58 gold studies. A few minutes including download.
!python $CODE/src/local_extract.py --data "$TRAIN" --validate --four-bit --batch 8

# 3. Only once step 2 clearly beats step 1, label all 4,349. Hours, not minutes.
# !python $CODE/src/local_extract.py --data "$TRAIN" --four-bit --batch 8 \
#     --out /kaggle/working/report_labels.csv

# 4. Save /kaggle/working/report_labels.csv as a Kaggle Dataset for the training
#    notebook to attach.
