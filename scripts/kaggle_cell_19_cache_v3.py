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
# A 60-study probe runs first, into its own directory so the real cache stays clean.
# The probe is CHECKED, not just eyeballed: if right knees are not being made to
# look like left ones the cell stops before spending 40 minutes building a cache
# that would quietly train the model on two mirrored anatomies.
#
# 4 slots x 9 slices x 288px = 2.99 MB/study, ~13.2 GB.
#
# Attach the competition only. CPU, Internet on. ~45 min.
# ============================================================
!pip install -q pydicom opencv-python-headless

import sys, glob, json, os
import numpy as np
import pandas as pd

CODE = "/kaggle/working/rsna-knee"
!rm -rf $CODE && git clone -q https://github.com/IamWasee/rsna-knee.git $CODE
sys.path.insert(0, f"{CODE}/src")

CACHE = "/kaggle/working/cache_v3"
PROBE = "/kaggle/working/cache_probe"
ARGS = "--size 288 --crop-mm 140 --anchors 3"

# ---------------------------------------------------------------- probe
!python $CODE/src/preprocess.py --out $PROBE --limit 60 --workers 4 {ARGS}

man = json.load(open(f"{PROBE}/cache_manifest.json"))
print("\nmanifest:", json.dumps(man))
assert man["laterality"] is True, "laterality is off in the manifest"

meta = pd.read_csv(f"{PROBE}/study_meta.csv")
n_r = int((meta["side"].astype(str).str.upper().str[0] == "R").sum())
n_l = int((meta["side"].astype(str).str.upper().str[0] == "L").sum())
print(f"\nsides resolved: {n_l} left, {n_r} right, {len(meta) - n_l - n_r} unknown")
print(f"scanners: {meta['scanner'].nunique()} distinct fingerprints in {len(meta)} studies")

if n_l < 3 or n_r < 3:
    raise SystemExit(
        f"only {n_l} left / {n_r} right resolved in 60 studies. Either the side "
        f"lookup is failing or this sample is one-sided -- either way the check "
        f"below cannot run, so do not build the full cache yet."
    )

# ---------------------------------------------------------------- the check
# If normalisation worked, an average right knee should now resemble an average
# left knee MORE than its own mirror image does. Coronal slot (index 1), middle
# slice. This is the whole claim of the change, stated as a number.
def mean_coronal(uids):
    v = [np.load(f"{PROBE}/{u}.npy")[1, 4].astype(np.float32) for u in uids
         if os.path.exists(f"{PROBE}/{u}.npy")]
    return np.mean(v, axis=0)

def corr(a, b):
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))

side0 = meta["side"].astype(str).str.upper().str[0]
L = mean_coronal(meta[side0 == "L"]["StudyInstanceUID"])
R = mean_coronal(meta[side0 == "R"]["StudyInstanceUID"])

aligned = corr(L, R)
mirrored = corr(L, R[:, ::-1])
print(f"\nmean left knee vs mean right knee, coronal:")
print(f"  as normalised : {aligned:+.4f}")
print(f"  mirrored back : {mirrored:+.4f}")
if aligned <= mirrored:
    raise SystemExit(
        f"laterality normalisation is NOT working: right knees still resemble the "
        f"mirror of left knees ({mirrored:.4f}) at least as much as left knees "
        f"themselves ({aligned:.4f}). Building the full cache would train on two "
        f"mirrored anatomies. Stopping."
    )
print(f"  -> right knees now read as left ones (+{aligned - mirrored:.4f}). Proceeding.")

# ---------------------------------------------------------------- eyes on it
import matplotlib.pyplot as plt
names = ["Sag-FS", "Cor-FS", "Ax-FS", "Sag-T1"]
for s in ("L", "R"):
    uid = meta[side0 == s]["StudyInstanceUID"].iloc[0]
    a = np.load(f"{PROBE}/{uid}.npy")
    n = a.shape[1]
    fig, ax = plt.subplots(a.shape[0], n, figsize=(1.9 * n, 1.9 * a.shape[0]))
    ax = np.atleast_2d(ax)
    for r in range(a.shape[0]):
        for k in range(n):
            ax[r, k].imshow(a[r, k], cmap="gray"); ax[r, k].axis("off")
            ax[r, k].set_title(f"{names[r] if r < 4 else r} {k}", fontsize=7)
    fig.suptitle(f"scanned as a {s} knee -- should read as a LEFT knee", fontsize=11)
    plt.tight_layout(); plt.show()

# ---------------------------------------------------------------- full build
!rm -rf $PROBE
!python $CODE/src/preprocess.py --out $CACHE --workers 4 {ARGS}

n = len(glob.glob(f"{CACHE}/*.npy"))
gb = sum(os.path.getsize(p) for p in glob.glob(f"{CACHE}/*.npy")) / 1e9
meta = pd.read_csv(f"{CACHE}/study_meta.csv")
side0 = meta["side"].astype(str).str.upper().str[0]
print(f"\n{n} studies, {gb:.1f} GB")
print(f"laterality: {(side0 == 'R').sum()} right normalised to left, "
      f"{(side0 == 'L').sum()} already left, {len(meta) - (side0.isin(['L','R'])).sum()} unknown")
print(f"scanners:   {meta['scanner'].nunique()} distinct fingerprints")
print(meta['scanner'].value_counts().head(10))
