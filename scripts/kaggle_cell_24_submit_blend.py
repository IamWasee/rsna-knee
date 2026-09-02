# ============================================================
# RSNA Knee — submit the full-v3 blend, and buy a calibration point
#
# Ten checkpoints: five folds of the shared+GAP head and five of slot+focal, all
# on cache_v3 (288px, laterality, scanner-grouped folds). infer.py rank-averages
# every .pt it finds, so ten equally weighted models IS the 50/50 blend that
# measured 0.797 out of fold.
#
# The point of this run is not the score, it is the OFFSET. We have three
# submissions in total and two of the three moves changed more than one thing at
# once, so we have no idea what a held-out gain is worth on the leaderboard. The
# published pairs say out-of-fold UNDERSTATES it -- +0.005 OOF for +0.017 LB in
# one case -- and that if true, changes which experiments are worth running.
#
# Deliberately NOT submitting the per-label tuned weights. They added +0.0005 out
# of fold, which is our own measurement agreeing with the warning: twelve weights
# fitted on the data they are scored on is how the 442-team plateau was built.
#
# Attach: the competition + Notebook Output -> full-v3. GPU on, Internet on.
# ============================================================
import sys, os, time, shutil
CODE = "/kaggle/working/rsna-knee"
!rm -rf $CODE && git clone -q https://github.com/IamWasee/rsna-knee.git $CODE
sys.path.insert(0, f"{CODE}/src")
from kaggle_paths import find, describe

t0 = time.time()
WEIGHTS = find(suffix=".pt")
if not WEIGHTS:
    describe(); raise SystemExit("attach Notebook Output -> full-v3")

# One flat directory, because infer.py takes the parent of what it is given and a
# nested layout would silently submit a 1-model ensemble as if it were ten.
WDIR = "/kaggle/working/w_all"
os.makedirs(WDIR, exist_ok=True)
for w in WEIGHTS:
    # fold0.pt exists in both arms, so keep the arm in the name.
    arm = os.path.basename(os.path.dirname(w))
    shutil.copy(w, os.path.join(WDIR, f"{arm}_{os.path.basename(w)}"))
print(f"{len(WEIGHTS)} checkpoints -> {WDIR}")
for w in sorted(os.listdir(WDIR)):
    print(f"   {w}")
if len(WEIGHTS) != 10:
    print(f"WARNING: expected 10 checkpoints (2 heads x 5 folds), found {len(WEIGHTS)}. "
          f"The blend measured at 0.797 was ten models; this is not that blend.")

# Preprocess the test set exactly as the training cache was built. Every field
# comes from the checkpoint's own manifest rather than today's defaults -- the
# mismatch that once scored 0.675 changed no tensor shape and nothing in the log.
import torch, glob
ck = torch.load(sorted(glob.glob(f"{WDIR}/*.pt"))[0], map_location="cpu",
                weights_only=False)
man, a = ck.get("cache_manifest", {}), ck.get("args", {})
SLOTS = man.get("slots", 4)
SIZE = man.get("size", 288)
CROP = man.get("crop_mm", 140.0)
ANCHORS = man.get("n_anchors", 3)
LAT = "" if man.get("laterality", False) else "--no-laterality"
print(f"\ntraining cache: slots={SLOTS} size={SIZE} crop={CROP} anchors={ANCHORS} "
      f"laterality={man.get('laterality', False)}")
print(f"head={a.get('head')} pool={a.get('pool')} backbone={a.get('backbone')}")

CACHE = "/kaggle/working/test_cache"
!python $CODE/src/preprocess.py --out $CACHE --split test --workers 4 \
    --slots $SLOTS --size $SIZE --crop-mm $CROP --anchors $ANCHORS $LAT
print(f"preprocess: {(time.time()-t0)/60:.0f} min")

!python $CODE/src/infer.py --cache $CACHE --weights $WDIR \
    --out /kaggle/working/submission.csv

import pandas as pd
if not os.path.exists("/kaggle/working/submission.csv"):
    raise SystemExit("no submission.csv -- see the error above")
sub = pd.read_csv("/kaggle/working/submission.csv")
print(f"\nsubmission: {sub.shape[0]} rows x {sub.shape[1]} cols")
print(sub.describe().T[["min", "mean", "max"]])
assert sub.iloc[:, 1:].notna().all().all(), "NaNs in the submission"
assert sub.iloc[:, 0].is_unique, "duplicate study ids in the submission"
print(f"\ntotal {(time.time()-t0)/60:.0f} min")
print("\nOut of fold this blend measured 0.797. Whatever the leaderboard says,")
print("the DIFFERENCE is the number worth writing down.")
