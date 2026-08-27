# ============================================================
# RSNA Knee — Look at what the model actually sees
#
# Attach: competition, the cache notebook, and your weights.
# GPU optional (CPU works, just slower).
# ============================================================
!pip install -q timm

import sys, glob, os
CODE = "/kaggle/working/rsna-knee"
!rm -rf $CODE && git clone -q https://github.com/IamWasee/rsna-knee.git $CODE
sys.path.insert(0, f"{CODE}/src")

cands   = glob.glob("/kaggle/input/**/*.npy", recursive=True)
CACHE   = os.path.dirname(cands[0]) if cands else None
WEIGHTS = os.path.dirname(sorted(glob.glob("/kaggle/input/**/*.pt", recursive=True))[0])
print("cache:  ", CACHE)
print("weights:", WEIGHTS)

# Three studies, with images: predictions vs radiologist truth vs the report.
!python $CODE/src/inspect_study.py --cache "$CACHE" --weights "$WEIGHTS" --n 3

# The three it got most wrong -- more informative than the ones it got right.
!python $CODE/src/inspect_study.py --cache "$CACHE" --weights "$WEIGHTS" --n 3 --worst
