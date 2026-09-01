# ============================================================
# RSNA Knee — DINOv2-small instead of ResNet-34
#
# Attach:
#   1. the competition
#   2. Notebook Output -> preprocessing-v2 (cache_v2)
#   3. Dataset -> stevenleehans/rsna-knee-llm-report-labels
#   4. Model   -> metaresearch/dinov2/PyTorch/small/1
#
# One fold only. A ViT is slower per step than a ResNet, and one fold answers the
# question -- the previous run's folds spanned 0.754-0.783, so a backbone worth
# adopting should clear that band on its own.
#
# GPU on, Internet on.
# ============================================================
!pip install -q timm transformers

import sys, glob, os
import pandas as pd

CODE = "/kaggle/working/rsna-knee"
!rm -rf $CODE && git clone -q https://github.com/IamWasee/rsna-knee.git $CODE
sys.path.insert(0, f"{CODE}/src")
from kaggle_paths import find, competition_file, describe

TRAIN = competition_file("train.csv")

npys = find(suffix=".npy")
v2 = [f for f in npys if "cache_v2" in f]
if not npys:
    describe(); raise SystemExit("attach the preprocessing-v2 notebook output")
CACHE = os.path.dirname(v2[0] if v2 else npys[0])

lab = find(filename="llm_labels_v4_blend.csv")
if not lab:
    describe(); raise SystemExit("attach stevenleehans/rsna-knee-llm-report-labels")
LABELS = lab[0]

# The Kaggle model mount holds config.json next to the weights.
dino = [os.path.dirname(p) for p in find(filename="config.json") if "dinov2" in p.lower()]
if not dino:
    describe(); raise SystemExit("attach the model metaresearch/dinov2/PyTorch/small/1")
DINO = f"dinov2:{dino[0]}"

print(f"cache:    {CACHE}")
print(f"labels:   {LABELS}")
print(f"backbone: {DINO}")

# Only the backbone changes from the 0.770 run. Head and pooling stay on the old
# behaviour, and the learning rates follow the public baseline's split: the head is
# new and trains fast, the encoder is only being adapted and trains 125x slower.
!python $CODE/src/train.py --cache "$CACHE" --labels "$LABELS" \
    --backbone "$DINO" --only-fold 0 --epochs 12 --batch 8 \
    --lr 1e-3 --lr-backbone 8e-6 --unfreeze-last 6 --weight-decay 0.02 \
    --head shared --pool gap \
    --out /kaggle/working/weights_dino

print("\nCompare fold 0 against the ResNet-34 run: 0.773 on the same fold.")
