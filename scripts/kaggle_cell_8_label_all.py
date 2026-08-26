# ============================================================
# RSNA Knee — Step 8: label all 4,349 reports, then ensemble
# GPU required. HOURS. Resumable -- just re-run if the session dies.
#
# IMPORTANT: save /kaggle/working/model_preds_all.csv as a Kaggle Dataset and
# attach it before re-running, or a fresh session starts from zero.
# ============================================================
!pip install -q -U transformers accelerate bitsandbytes

import sys, glob, os
CODE = "/kaggle/working/rsna-knee"
!rm -rf $CODE && git clone -q https://github.com/IamWasee/rsna-knee.git $CODE
sys.path.insert(0, f"{CODE}/src")

TRAIN = next(iter(glob.glob("/kaggle/input/**/train.csv", recursive=True)), None)
OUT   = "/kaggle/working/model_preds_all.csv"

# Carry over a previous partial run if you attached it as a dataset.
prev = glob.glob("/kaggle/input/**/model_preds_all.csv", recursive=True)
if prev and not os.path.exists(OUT):
    import shutil; shutil.copy(prev[0], OUT)
    import pandas as pd
    print(f"resuming from {prev[0]}: {len(pd.read_csv(OUT))} studies already done")

# batch 16 fits a T4 at 4-bit and is roughly twice as fast as batch 8.
!python $CODE/src/local_extract.py --data "$TRAIN" --four-bit --batch 16 --out $OUT

# Combine with the keyword extractor -- rank_mean w=0.5 measured 0.798 on the
# gold 58, against 0.707 keyword alone and 0.728 model alone.
!python $CODE/src/ensemble.py --data "$TRAIN" --model-preds $OUT \
    --out /kaggle/working/report_labels.csv

import pandas as pd
lab = pd.read_csv("/kaggle/working/report_labels.csv")
print(f"\n{len(lab)} studies labelled")
print("\npositive rate at 0.5 (compare against GOLD_PREVALENCE in config.py):")
for c in lab.columns[1:]:
    print(f"  {c:<18} {(lab[c] > 0.5).mean():.3f}")
