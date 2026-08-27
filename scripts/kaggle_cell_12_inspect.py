# ============================================================
# RSNA Knee — Look at what the model actually sees
#
# Nothing to upload. It reads the competition's own knee MRIs -- the 58 studies
# that have real radiologist labels, so you can see who was right.
#
# Attach: the competition, the cache notebook ("notebooka"), and a notebook whose
# output contains .pt checkpoints. GPU optional.
# ============================================================
!pip install -q timm

import sys, glob, os
CODE = "/kaggle/working/rsna-knee"
!rm -rf $CODE && git clone -q https://github.com/IamWasee/rsna-knee.git $CODE
sys.path.insert(0, f"{CODE}/src")

cands = glob.glob("/kaggle/input/**/*.npy", recursive=True)
CACHE = os.path.dirname(cands[0]) if cands else None
pts   = sorted(glob.glob("/kaggle/input/**/*.pt", recursive=True))
if not pts:
    raise SystemExit("no .pt checkpoints -- attach a training notebook's output.")
WEIGHTS = os.path.dirname(pts[0])
print("cache:  ", CACHE, f"({len(cands)} volumes)")
print("weights:", WEIGHTS, f"({len(pts)} folds)")

# The three the model got MOST WRONG -- far more informative than its successes.
!python $CODE/src/inspect_study.py --cache "$CACHE" --weights "$WEIGHTS" \
    --n 3 --worst --figdir /kaggle/working/figs

# Now display the saved images. They cannot render from inside `!python`, so the
# script writes PNGs and the notebook kernel shows them here.
from IPython.display import Image, display
for f in sorted(glob.glob("/kaggle/working/figs/*.png")):
    display(Image(filename=f))
