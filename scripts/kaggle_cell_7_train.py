# ============================================================
# RSNA Knee — Step 7: train the image model
# GPU required. Attach the cache dataset from step 6.
# ============================================================
!pip install -q timm

import sys, glob
CODE = "/kaggle/working/rsna-knee"
!rm -rf $CODE && git clone -q https://github.com/IamWasee/rsna-knee.git $CODE
sys.path.insert(0, f"{CODE}/src")

CACHE  = "/kaggle/input/rsna-knee-cache"        # the dataset saved in step 6
LABELS = "/kaggle/input/rsna-knee-labels/report_labels.csv"   # from step 4/5

# 1. One fold, few epochs -- proves the loop runs end to end and gives a first
#    honest number on the 58 gold studies.
!python $CODE/src/train.py --cache $CACHE --labels $LABELS \
    --only-fold 0 --epochs 2 --batch 4 --backbone resnet34 --out /kaggle/working/weights

# 2. Full run once the above is sane. Kaggle caps a session at 9h, so train folds
#    across separate sessions with --only-fold rather than all five at once.
# !python $CODE/src/train.py --cache $CACHE --labels $LABELS \
#     --only-fold 1 --epochs 6 --batch 4 --backbone resnet34 --out /kaggle/working/weights

# 3. Save /kaggle/working/weights as a Kaggle Dataset -- the submission notebook
#    is offline and can only get weights from an attached dataset.
