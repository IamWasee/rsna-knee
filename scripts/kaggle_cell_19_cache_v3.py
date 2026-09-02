# ============================================================
# RSNA Knee — cache v3: 288px + laterality normalisation + scanner metadata
#
# Three changes over cache_288, all of them preprocessing, which is why the cache
# has to be rebuilt rather than patched:
#
#   1. 288px (was 256). Isolated as one variable by a competitor at 0.7815.
#   2. Laterality. Every knee is presented as a left one. The operation depends on
#      the plane: coronal and axial get their PIXELS mirrored, sagittal gets its
#      SLICE ORDER reversed -- there medial-lateral is the stack axis, so mirroring
#      would flip anterior/posterior and put the patella behind the knee.
#      Side comes from the Laterality tag where present, otherwise from the x of the
#      image CENTRE in patient coordinates -- not ImagePositionPatient, which is the
#      corner: a 160mm field starting 10mm past the midline is centred 70mm the
#      other side, so the corner names the wrong knee.
#   3. study_meta.csv -- a scanner fingerprint per study (manufacturer, model,
#      software, imaging frequency, coil). Without this file, folds group only on
#      report text and the model is free to memorise the site rather than the knee.
#
# Attach the competition only. CPU. ~40 min for the full run.
# ============================================================
!pip install -q pydicom opencv-python-headless

import sys, glob, json, os, numpy as np
CODE = "/kaggle/working/rsna-knee"
!rm -rf $CODE && git clone -q https://github.com/IamWasee/rsna-knee.git $CODE
sys.path.insert(0, f"{CODE}/src")

CACHE = "/kaggle/working/cache_v3"

!python $CODE/src/preprocess.py --out $CACHE --limit 40 --workers 4 \
    --size 288 --crop-mm 140 --anchors 3

man = json.load(open(f"{CACHE}/cache_manifest.json"))
print("\nmanifest:", man)
v = np.load(sorted(glob.glob(f"{CACHE}/*.npy"))[0])
print(f"shape {v.shape}  ->  {v.nbytes/1e6:.2f} MB/study, "
      f"{v.nbytes*4407/1e9:.1f} GB total")

# Did laterality actually fire? study_meta.csv records the side each study was
# resolved to and what was done about it. If every row says "L" and none were
# flipped, the tag lookup silently failed and the run is not worth 40 minutes.
import pandas as pd
meta = pd.read_csv(f"{CACHE}/study_meta.csv")
print("\ncolumns:", list(meta.columns))
for c in meta.columns:
    if c != "StudyInstanceUID":
        print(f"\n{c}:\n{meta[c].value_counts(dropna=False).head(8)}")

# Left and right knees side by side after normalisation. They should now look like
# the same anatomy in the same orientation, not mirror images of each other.
import matplotlib.pyplot as plt
names = ["Sag-FS", "Cor-FS", "Ax-FS", "Sag-T1"]
side_col = next((c for c in meta.columns if "side" in c.lower()), None)
picks = []
if side_col:
    for s in ("L", "R"):
        rows = meta[meta[side_col].astype(str).str.upper().str[0] == s]
        if len(rows):
            picks.append((s, rows.iloc[0]["StudyInstanceUID"]))
if not picks:
    picks = [("?", os.path.basename(p)[:-4]) for p in sorted(glob.glob(f"{CACHE}/*.npy"))[:2]]

for s, uid in picks:
    p = f"{CACHE}/{uid}.npy"
    if not os.path.exists(p):
        continue
    a = np.load(p)
    n = a.shape[1]
    fig, ax = plt.subplots(a.shape[0], n, figsize=(2.0*n, 2.0*a.shape[0]))
    ax = np.atleast_2d(ax)
    for r in range(a.shape[0]):
        for k in range(n):
            ax[r, k].imshow(a[r, k], cmap="gray"); ax[r, k].axis("off")
            ax[r, k].set_title(f"{names[r] if r < 4 else r} {k}", fontsize=7)
    fig.suptitle(f"original side {s} -- should read as a LEFT knee", fontsize=10)
    plt.tight_layout(); plt.show()

# Full run -- uncomment once the two knees above look like the same anatomy.
# !python $CODE/src/preprocess.py --out $CACHE --workers 4 --size 288 --crop-mm 140 --anchors 3
