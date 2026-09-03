# ============================================================
# RSNA Knee — ONE per-plane cache with the whole slice stack
#
# Pushed three times, once per plane, because three of these in ONE notebook is
# what killed the first attempt: 3 x 8.8 GB = 26.4 GB against a 20 GB working
# limit. Sagittal and coronal both finished; axial died at study 1,600 with the
# disk full, and a failed notebook saves no output, so all three were lost. I had
# checked that ONE plane fits and then built three into the same 20 GB.
#
# Why per-plane at all, beyond the disk: the two biggest published levers left
# both want it. All slices max-pooled beat fixed sampling by +0.037 macro and
# +0.087 on MENISCUS -- the gain lands on the small focal finding, which is the
# shape of our deficit. And separate per-plane models combined afterwards beat one
# shared network fed every plane by +0.076. We load 9 slices of ~24 and run one
# encoder across four slots: the losing side of both.
#
# 8 anchors x 3 adjacent = 24 slices, covering essentially the whole stack.
# 1 slot x 24 x 288px = 2.0 MB/study = 8.8 GB. Measured, not estimated.
#
# Attach the competition only. CPU -- no GPU quota. Internet on. ~45 min.
# ============================================================
!pip install -q pydicom opencv-python-headless

import sys, os, glob, json
import numpy as np, pandas as pd

CODE = "/kaggle/working/rsna-knee"
!rm -rf $CODE && git clone -q https://github.com/IamWasee/rsna-knee.git $CODE
sys.path.insert(0, f"{CODE}/src")

SLOT, TAG = __SLOT__, "__TAG__"
OUT = f"/kaggle/working/cache_{TAG}"
ARGS = "--size 288 --crop-mm 140 --anchors 8"

!python $CODE/src/preprocess.py --out $OUT --workers 4 --only-slot $SLOT {ARGS}

files = glob.glob(f"{OUT}/*.npy")
if not files:
    raise SystemExit(
        "nothing was written -- read preprocess.py's output above. "
        "'unrecognized arguments' means this clone of master predates the flag."
    )
gb = sum(os.path.getsize(p) for p in files) / 1e9
man = json.load(open(f"{OUT}/cache_manifest.json"))
meta = pd.read_csv(f"{OUT}/study_meta.csv")
side = meta["side"].astype(str).str.upper().str[0]
print(f"\n{len(files)} studies, {gb:.1f} GB")
print(f"manifest: {json.dumps(man)}")
print(f"laterality: {(side=='R').sum()} right normalised to left, "
      f"{(side=='L').sum()} already left, {(~side.isin(['L','R'])).sum()} unknown")
if gb > 18:
    print(f"WARNING: {gb:.1f} GB is close to the 20 GB working limit.")

# Does this plane actually carry signal for every study, or is the sequence often
# missing? A mostly-blank volume trains on nothing and says nothing about it.
blanks = []
for f in files[:400]:
    v = np.load(f)
    blanks.append(float((v.reshape(-1, v.shape[-1] * v.shape[-2]).max(1) == 0).mean()))
b = np.array(blanks)
print(f"\nblank slices over {len(b)} studies: mean {b.mean():.1%}, "
      f"{(b > 0.5).sum()} studies more than half blank")
if b.mean() > 0.2:
    print("This plane is missing for many studies -- weight it accordingly in the blend.")

import matplotlib.pyplot as plt
v = np.load(sorted(files)[0])
n = v.shape[1]; cols = 12; rows = int(np.ceil(n / cols))
fig, ax = plt.subplots(rows, cols, figsize=(1.5 * cols, 1.6 * rows))
ax = np.atleast_2d(ax)
for k in range(rows * cols):
    a = ax[k // cols, k % cols]; a.axis("off")
    if k < n:
        a.imshow(v[0, k], cmap="gray"); a.set_title(f"{k}", fontsize=6)
fig.suptitle(f"{TAG}: all {n} slices of one study", fontsize=11)
plt.tight_layout(); plt.show()
