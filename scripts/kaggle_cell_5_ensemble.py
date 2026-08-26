# ============================================================
# RSNA Knee — Step 5: ensemble keyword + model
# GPU needed only for the local_extract step.
# ============================================================
import os, sys, glob

CODE = "/kaggle/working/rsna-knee"
!rm -rf $CODE && git clone -q https://github.com/IamWasee/rsna-knee.git $CODE
sys.path.insert(0, f"{CODE}/src")

TRAIN = next(iter(glob.glob("/kaggle/input/**/train.csv", recursive=True)), None)
print("train.csv:", TRAIN)

# 1. Model predictions on the 58 gold studies, saved this time so the ensemble
#    can reuse them without re-running the GPU.
!python $CODE/src/local_extract.py --data "$TRAIN" --validate --four-bit --batch 8 \
    --out /kaggle/working/model_preds_gold.csv

# 2. Compare keyword, model, and combinations of the two.
!python $CODE/src/ensemble.py --data "$TRAIN" \
    --model-preds /kaggle/working/model_preds_gold.csv --per-label

# 3. If the ensemble wins, label all 4,349 (hours) then combine.
# !python $CODE/src/local_extract.py --data "$TRAIN" --four-bit --batch 8 \
#     --out /kaggle/working/model_preds_all.csv
# !python $CODE/src/ensemble.py --data "$TRAIN" \
#     --model-preds /kaggle/working/model_preds_all.csv \
#     --out /kaggle/working/report_labels.csv
