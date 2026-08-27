# ============================================================
# RSNA Knee — Can Synovitis borrow from Effusion?
#
# Only ~16% of reports mention synovitis; 47% of annotated studies are positive.
# The extractor is guessing on most studies. Effusion is well reported and
# correlates ~0.5 with synovitis in the annotated set.
#
# Attach: competition + a notebook output with model_preds_all.csv. CPU, ~1 min.
# ============================================================
import sys, glob
CODE = "/kaggle/working/rsna-knee"
!rm -rf $CODE && git clone -q https://github.com/IamWasee/rsna-knee.git $CODE
sys.path.insert(0, f"{CODE}/src")

import pandas as pd
TRAIN = next(iter(glob.glob("/kaggle/input/**/train.csv", recursive=True)))
# The sweep needs predictions ON THE GOLD STUDIES. model_preds_all.csv holds only
# the 4,349 unlabelled ones, so scoring it has nothing to score against.
gold_preds = glob.glob("/kaggle/input/**/model_preds_gold.csv", recursive=True)
if not gold_preds:
    raise SystemExit("attach the notebook containing model_preds_gold.csv "
                     "(the --validate run), not just model_preds_all.csv")
PREDS = gold_preds[0]
print("gold preds:", PREDS, f"({len(pd.read_csv(PREDS))} studies)")

# Prints the usual comparison plus the Synovitis borrowing sweep at the end.
!python $CODE/src/ensemble.py --data "$TRAIN" --model-preds "$PREDS"

# How often is synovitis even mentioned? The 16% claim, checked on our own corpus.
import re
train = pd.read_csv(TRAIN)
pat = re.compile(r"synovit|sinovit|υμεν|синови|sinovyal|synovial", re.I)
hit = train["Report"].astype(str).str.contains(pat)
print(f"\nreports mentioning a synovitis term: {hit.mean():.1%} ({hit.sum()}/{len(train)})")
gold = train[train["Synovitis"].notna()]
print(f"gold Synovitis positive rate: {gold['Synovitis'].mean():.1%}")
print(f"of gold positives, how many reports mention it: "
      f"{hit[gold.index][gold['Synovitis'] == 1].mean():.1%}")
