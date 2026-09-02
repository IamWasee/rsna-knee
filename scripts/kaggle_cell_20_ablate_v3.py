# ============================================================
# RSNA Knee — fold-0 ablation on cache v3: what did each change buy?
#
# Three things changed at once and they must not be judged together:
#   A. the cache      (256 -> 288px, plus laterality normalisation)
#   B. the fold split (report text -> scanner fingerprint)
#   C. the head       (shared+GAP -> per-diagnosis slot attention + focal pooling)
#
# B lowers the number on purpose. Grouping by scanner removes site memorisation
# from the validation score, which one competitor measured as an inflated 0.053.
# So arm 1 below is EXPECTED to read lower than the 0.826 fold-0 of the cache_v2
# run even if the images got better -- the two numbers measure different things.
# Arm 2 is the only clean comparison here: same cache, same split, head only.
#
# Reference points, all fold 0:
#   cache_v2, 256px, report-grouped, shared+gap, DINOv2 ....... 0.826
#   (5-fold mean of that run ................................. 0.814)
#
# Attach:
#   1. the competition
#   2. Notebook Output -> the cache v3 notebook (cache_v3)
#   3. Dataset -> stevenleehans/rsna-knee-llm-report-labels
#   4. Model   -> metaresearch/dinov2/PyTorch/small/1
#
# GPU on, Internet on. ~1h30 for both arms.
# ============================================================
!pip install -q timm transformers

import sys, os
CODE = "/kaggle/working/rsna-knee"
!rm -rf $CODE && git clone -q https://github.com/IamWasee/rsna-knee.git $CODE
sys.path.insert(0, f"{CODE}/src")
from kaggle_paths import find, describe

npys = find(suffix=".npy")
v3 = [f for f in npys if "cache_v3" in f]
if not v3:
    describe(); raise SystemExit("attach the cache v3 notebook output")
CACHE = os.path.dirname(v3[0])

lab = find(filename="llm_labels_v4_blend.csv")
if not lab:
    describe(); raise SystemExit("attach stevenleehans/rsna-knee-llm-report-labels")
LABELS = lab[0]

dino = [os.path.dirname(p) for p in find(filename="config.json") if "dinov2" in p.lower()]
if not dino:
    describe(); raise SystemExit("attach metaresearch/dinov2/PyTorch/small/1")
DINO = f"dinov2:{dino[0]}"

# study_meta.csv is what makes the folds scanner-grouped. Without it train.py falls
# back to report text and says so; check here rather than reading it out of the log.
meta = os.path.join(CACHE, "study_meta.csv")
print(f"cache:  {CACHE}")
print(f"meta:   {'present' if os.path.exists(meta) else 'MISSING -- folds will not be scanner-grouped'}")
print(f"labels: {LABELS}")

COMMON = (f'--cache "{CACHE}" --labels "{LABELS}" --backbone "{DINO}" '
          f'--size 288 --epochs 12 --batch 8 --only-fold 0 '
          f'--lr 1e-3 --lr-backbone 8e-6 --unfreeze-last 6 --weight-decay 0.02')

# Arm 1 -- control. Identical model to the 0.814 run; only the cache and the fold
# split differ, so this is the new baseline everything else is measured against.
print("\n" + "=" * 60 + "\nARM 1: shared head + GAP  (control)\n" + "=" * 60)
!python $CODE/src/train.py {COMMON} --head shared --pool gap --out /kaggle/working/w_a1

# Arm 2 -- per-diagnosis attention and focal pooling. GAP averages over every slice
# and every patch, so a 5mm meniscal tear is diluted by a whole normal knee. Focal
# pooling keeps mean + top-eighth; the slot head gives each diagnosis its own query
# with an anatomy prior over the four sequence slots.
print("\n" + "=" * 60 + "\nARM 2: slot head + focal pooling\n" + "=" * 60)
!python $CODE/src/train.py {COMMON} --head slot --pool focal --out /kaggle/working/w_a2

import re, glob
print("\n" + "=" * 60)
for name, out in (("arm1 shared+gap", "w_a1"), ("arm2 slot+focal", "w_a2")):
    f = glob.glob(f"/kaggle/working/{out}/*.pt")
    print(f"{name:<20} checkpoints: {len(f)}")
print("Compare 'fold 0 best OOF macro AUC' in the two blocks above.")
print("Arm 2 minus arm 1 is the head's contribution, everything else held.")
