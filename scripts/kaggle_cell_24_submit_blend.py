# ============================================================
# RSNA Knee — SUBMISSION notebook. Internet OFF.
#
# A code competition scores by rerunning this notebook against the hidden test
# set, and that rerun has no network. So: no git clone, no pip install. src/
# comes from the dataset abdullahwasee/rsna-knee-src (published by
# scripts/sync_src.py), and every package used is already in the Kaggle image.
#
# Attach, all four:
#   1. Competition -> rsna-knee-abnormality-detection
#   2. Dataset     -> abdullahwasee/rsna-knee-src
#   3. Notebook    -> full-v3            (the ten checkpoints)
#   4. Model       -> metaresearch/dinov2/PyTorch/small/1
#      -- needed even though every weight is in the checkpoint: transformers
#         reads the architecture config from that mount.
#
# GPU on. INTERNET OFF.
#
# Ten checkpoints: five folds of shared+GAP and five of slot+focal on cache_v3.
# infer.py rank-averages every .pt it finds, so ten equal models IS the 50/50
# blend that measured 0.797 out of fold.
#
# During YOUR run the test set is a 3-study placeholder and this finishes in
# minutes. The real work happens in Kaggle's rerun after you press Submit.
# ============================================================
import sys, os, time, shutil, glob

# Find src/, and INSIST on the stamped copy. Every training notebook saves its
# own `git clone` into its output, so /kaggle/input holds several copies of this
# repo frozen at whatever commit that run cloned. The first walk found full-v3's
# copy -- which predates the rank_average fix -- and inference died on the exact
# bug the dataset exists to carry the fix for. Only SRC_VERSION.txt distinguishes
# them, so require it rather than taking the first infer.py that turns up.
stamped, fallbacks = [], []
for root, dirs, files in os.walk("/kaggle/input", followlinks=True):
    if "rsna-knee-abnormality-detection" in root:
        dirs[:] = []                      # never descend the DICOM archive
        continue
    if "infer.py" in files and "preprocess.py" in files:
        (stamped if "SRC_VERSION.txt" in files else fallbacks).append(root)

if not stamped:
    raise SystemExit(
        "no stamped src/ found. Attach the dataset abdullahwasee/rsna-knee-src.\n"
        + (f"Unstamped copies were found and deliberately NOT used:\n"
           + "\n".join(f"    {f}" for f in fallbacks)
           + "\nThose are snapshots saved inside older notebook outputs; running "
             "them would submit whatever code that run happened to clone.\n"
           if fallbacks else "")
        + "Publish it with: python scripts/sync_src.py"
    )
SRC = stamped[0]
sys.path.insert(0, SRC)
print(f"src: {SRC}")
print(f"version: {open(os.path.join(SRC, 'SRC_VERSION.txt')).read().strip()}")
for f in fallbacks:
    print(f"  ignoring unstamped copy: {f}")

from kaggle_paths import find, describe

t0 = time.time()
WEIGHTS = [w for w in find(suffix=".pt")]
if not WEIGHTS:
    describe(); raise SystemExit("attach Notebook Output -> full-v3")

# One flat directory. fold0.pt exists in BOTH arms, so keep the arm in the name --
# a plain copy would overwrite and submit nine models as though it were ten.
WDIR = "/kaggle/working/w_all"
os.makedirs(WDIR, exist_ok=True)
for w in WEIGHTS:
    arm = os.path.basename(os.path.dirname(w))
    shutil.copy(w, os.path.join(WDIR, f"{arm}_{os.path.basename(w)}"))
print(f"\n{len(WEIGHTS)} checkpoints -> {WDIR}")
for w in sorted(os.listdir(WDIR)):
    print(f"   {w}")
print(f"\nrank-averaging {len(WEIGHTS)} checkpoints. Reference points:")
print("  10 (full-v3, both heads) -> OOF 0.797, leaderboard 0.865")
print("  5  (one head, 5 folds)   -> OOF 0.789 for slot+focal")

# Preprocess the test set exactly as the training cache was built -- every field
# from the checkpoint's own manifest, never today's defaults. The mismatch that
# once scored 0.675 changed no tensor shape and left nothing in the log.
import torch
ck = torch.load(sorted(glob.glob(f"{WDIR}/*.pt"))[0], map_location="cpu",
                weights_only=False)
man, a = ck.get("cache_manifest", {}), ck.get("args", {})
SLOTS, SIZE = man.get("slots", 4), man.get("size", 288)
CROP, ANCHORS = man.get("crop_mm", 140.0), man.get("n_anchors", 3)
LAT = "" if man.get("laterality", False) else "--no-laterality"
print(f"\ntraining cache: slots={SLOTS} size={SIZE} crop={CROP} anchors={ANCHORS} "
      f"laterality={man.get('laterality', False)}")
print(f"backbone: {a.get('backbone')}")

CACHE = "/kaggle/working/test_cache"
!python $SRC/preprocess.py --out $CACHE --split test --workers 4 \
    --slots $SLOTS --size $SIZE --crop-mm $CROP --anchors $ANCHORS $LAT
print(f"preprocess: {(time.time()-t0)/60:.1f} min")

!python $SRC/infer.py --cache $CACHE --weights $WDIR \
    --out /kaggle/working/submission.csv

import pandas as pd
if not os.path.exists("/kaggle/working/submission.csv"):
    raise SystemExit("no submission.csv -- see the error above")
sub = pd.read_csv("/kaggle/working/submission.csv")
print(f"\nsubmission: {sub.shape[0]} rows x {sub.shape[1]} cols")
assert sub.iloc[:, 1:].notna().all().all(), "NaNs in the submission"
assert sub.iloc[:, 0].is_unique, "duplicate study ids"
assert sub.shape[1] == 13, f"expected 13 columns, got {sub.shape[1]}"
print(sub.head(3).to_string())
print(f"\ntotal {(time.time()-t0)/60:.1f} min")
print("Out of fold this blend measured 0.797. The DIFFERENCE from the leaderboard")
print("is the number worth writing down, whichever way it goes.")
