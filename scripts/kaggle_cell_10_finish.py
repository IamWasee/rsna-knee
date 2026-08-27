# ============================================================
# RSNA Knee — Step 10: finish labelling + diagnose the extractors
#
# The previous run OOM'd at 3,520/4,349. Attach that notebook's output
# (+ Add Input -> Notebook Output -> "kneee mri") so this resumes instead
# of redoing four hours.
#
# GPU on. Delete Kaggle's default os.walk starter cell.
# ============================================================
!pip install -q -U transformers accelerate bitsandbytes

import sys, glob, os, shutil
CODE = "/kaggle/working/rsna-knee"
!rm -rf $CODE && git clone -q https://github.com/IamWasee/rsna-knee.git $CODE
sys.path.insert(0, f"{CODE}/src")

TRAIN = next(iter(glob.glob("/kaggle/input/**/train.csv", recursive=True)), None)
OUT   = "/kaggle/working/model_preds_all.csv"

prev = glob.glob("/kaggle/input/**/model_preds_all.csv", recursive=True)
if prev and not os.path.exists(OUT):
    shutil.copy(prev[0], OUT)
    import pandas as pd
    print(f"resuming: {len(pd.read_csv(OUT))} studies already done")
else:
    print("NO PRIOR RUN FOUND -- this will start from zero (~4h).")
    print("Attach the previous notebook's output first if you want to resume.")

# batch 8, not 16. 16 survived four hours then OOM'd on a long batch; the code
# now halves and retries on OOM, but starting lower is cheaper than recovering.
!python $CODE/src/local_extract.py --data "$TRAIN" --four-bit --batch 8 --out $OUT

# Per-source prevalence table -- this is the diagnostic that tells us whether
# Fracture 0.882 was a broken extractor or an artifact of rank-normalisation.
!python $CODE/src/ensemble.py --data "$TRAIN" --model-preds $OUT \
    --method mean --out /kaggle/working/report_labels.csv

import pandas as pd
lab = pd.read_csv("/kaggle/working/report_labels.csv")
print(f"\n{len(lab)} studies labelled")
print(lab.describe().T[["mean", "std", "min", "50%", "max"]])
