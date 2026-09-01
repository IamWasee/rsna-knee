# ============================================================
# RSNA Knee — THE SUBMISSION NOTEBOOK
#
# This is the one you submit. Rules it must respect:
#   - Internet OFF (Settings -> Internet)
#   - GPU on
#   - everything comes from attached inputs; nothing is downloaded
#
# Attach exactly TWO inputs:
#   1. the competition
#   2. Notebook Output -> your TRAINING notebook
#      (that one output carries both the .pt checkpoints AND src/, because the
#       training cell clones the repo into /kaggle/working, which Kaggle saves)
#
# Delete Kaggle's default os.walk starter cell.
# ============================================================
import sys, glob, os, time
from pathlib import Path
t0 = time.time()

# No pip install -- there is no internet. Verify what we need is already here.
missing = []
for mod in ("torch", "timm", "pydicom", "cv2", "pandas", "numpy", "sklearn"):
    try:
        __import__(mod)
    except ImportError:
        missing.append(mod)
if missing:
    raise SystemExit(f"NOT PREINSTALLED: {missing} -- cannot pip install offline.\n"
                     "Package these as a Kaggle Dataset and install from it.")
print("all required packages preinstalled")

# Find the code and the weights among the attached inputs.
#
# os.walk with pruning, not glob(recursive=True): the latter descends the whole
# competition dataset -- hundreds of thousands of DICOM files -- and measured 421
# seconds. Pruning that one directory keeps it complete and fast.
def find(filename: str = "", suffix: str = "") -> list[str]:
    """Locate files in the attached inputs without descending the DICOM archive.

    followlinks=True matters: Kaggle mounts notebook outputs as symlinks and
    os.walk does not follow them by default, so the default silently finds nothing
    where a recursive glob succeeds. Falls back to that glob if the walk comes up
    empty, since being slow beats failing a submission.
    """
    hits = []
    for root, dirs, files in os.walk("/kaggle/input", followlinks=True):
        dirs[:] = [d for d in dirs
                   if "rsna-knee-abnormality-detection" not in os.path.join(root, d)]
        for f in files:
            if (filename and f == filename) or (suffix and f.endswith(suffix)):
                hits.append(os.path.join(root, f))
    if not hits:
        pat = filename or f"*{suffix}"
        hits = [f for f in glob.glob(f"/kaggle/input/**/{pat}", recursive=True)
                if "rsna-knee-abnormality-detection" not in f]
    return sorted(hits)


found = find(filename="infer.py")
WEIGHTS = find(suffix=".pt")

# Kaggle's dataset uploader auto-extracts archives, and a .pt file IS a zip -- so
# uploading checkpoints to a dataset explodes each one into a directory tree
# (data.pkl, version, byteorder, data/...). Nothing is lost; rebuild the zips.
if not WEIGHTS:
    import zipfile
    marks = [os.path.dirname(p) for p in find(filename="data.pkl")]
    if marks:
        print(f"found {len(marks)} exploded checkpoints; rebuilding")
        os.makedirs("/kaggle/working/rebuilt", exist_ok=True)
        for inner in sorted(marks):
            outer = os.path.dirname(inner)                  # .../weights/fold0
            name = os.path.basename(outer)                  # fold0
            out = f"/kaggle/working/rebuilt/{name}.pt"
            # torch.save writes an uncompressed zip whose entries are prefixed by
            # an inner directory. Preserve that prefix exactly.
            with zipfile.ZipFile(out, "w", zipfile.ZIP_STORED) as z:
                for root, _, files in os.walk(outer):
                    for f in files:
                        full = os.path.join(root, f)
                        z.write(full, os.path.relpath(full, outer))
            print(f"  {name}.pt  ({os.path.getsize(out)/1e6:.1f} MB)")
        WEIGHTS = sorted(glob.glob("/kaggle/working/rebuilt/*.pt"))

if not found or not WEIGHTS:
    print("WHAT IS ACTUALLY ATTACHED:")
    for root, dirs, files in os.walk("/kaggle/input", followlinks=True):
        if "rsna-knee-abnormality-detection" in root:
            dirs[:] = []
            continue
        depth = root.count(os.sep) - 2
        if depth > 5:
            dirs[:] = []
            continue
        print("  " * depth + os.path.basename(root) + "/")
        for f in files[:6]:
            print("  " * (depth + 1) + f)
    raise SystemExit(
        f"missing: {'src/infer.py ' if not found else ''}"
        f"{'checkpoints' if not WEIGHTS else ''}\n"
        "Attach: Notebook Output -> the 5-fold TRAINING notebook version whose\n"
        "Output tab shows weights/fold0.pt .. fold4.pt AND rsna-knee/src/."
    )

# Derive the source directory from where infer.py actually is. It may sit in a
# repo checkout (rsna-knee/src/) or flat in a dataset folder (rsna_knee_src/), and
# assuming the former looked for weights/src/preprocess.py and found nothing.
SRC = os.path.dirname(found[0])
sys.path.insert(0, SRC)

# Collect every checkpoint into one flat directory. A dataset upload can nest each
# file in its own folder, and taking the parent of the first .pt would then point
# at a directory holding exactly one fold -- a 1-model ensemble submitted silently
# as if it were 5.
import shutil
WDIR = "/kaggle/working/weights_flat"
os.makedirs(WDIR, exist_ok=True)
for w in WEIGHTS:
    shutil.copy(w, os.path.join(WDIR, os.path.basename(w)))

print(f"src:     {SRC}")
print(f"weights: {WDIR}  ({len(WEIGHTS)} folds)")
for w in WEIGHTS:
    print(f"   {w}")

CACHE = "/kaggle/working/test_cache"

# Preprocess with the settings the checkpoint was TRAINED with, not today's
# defaults. A cache built to a different shape used to be silently cropped and
# zero-padded to fit, which fed the model a corner of each knee plus blank slices.
import torch
_ck = torch.load(sorted(glob.glob(f"{WDIR}/*.pt"))[0], map_location="cpu",
                 weights_only=False)
_a = _ck.get("args", {})
_man = _ck.get("cache_manifest", {})
# Prefer the manifest: it records what actually built the training cache, where
# args records only what was asked for.
SLOTS = _man.get("slots", _a.get("slots", _a.get("n_series", 4)))
SIZE = _man.get("size", _a.get("size", 256))
CROP = _man.get("crop_mm", 140.0)
print(f"training cache: slots={SLOTS} size={SIZE} crop_mm={CROP} "
      f"n_slices={_man.get('n_slices', _a.get('n_slices'))}")
print(f"checkpoint expects: slots={SLOTS} n_slices={_a.get('n_slices')} size={SIZE}")
if _a.get("n_slices") not in (9, None):
    print(f"  WARNING: this checkpoint wants {_a['n_slices']} slices per slot, but "
          f"preprocess.py now produces 9. These weights predate the v2 layout and "
          f"cannot be served correctly by the current pipeline -- retrain instead.")

# 1. Preprocess the hidden test set. Measured at 2.73 studies/sec with 4 workers,
#    so even 20,000 studies is ~2h of the 9h budget.
!python $SRC/preprocess.py --out $CACHE --split test --workers 4 \
    --slots $SLOTS --size $SIZE --crop-mm $CROP
print(f"preprocess: {(time.time()-t0)/60:.1f} min")

# 2. Predict and write submission.csv at the path Kaggle expects.
!python $SRC/infer.py --cache $CACHE --weights $WDIR --out /kaggle/working/submission.csv

import pandas as pd
if not os.path.exists("/kaggle/working/submission.csv"):
    raise SystemExit("infer.py did not produce submission.csv -- see the error above.")
sub = pd.read_csv("/kaggle/working/submission.csv")
print(sub.shape)
print(sub.head())
print(f"\ntotal: {(time.time()-t0)/60:.1f} min of the 540 available")
