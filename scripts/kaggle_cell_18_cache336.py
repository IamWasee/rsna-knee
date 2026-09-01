# ============================================================
# RSNA Knee — 336px cache, 3 slices per slot
#
# The public solutions use 336px with 3 slices per slot and a 130mm crop; ours is
# 256px with 9. Two independent sources report that trading slices for resolution
# wins here, and it is the shape a meniscal tear needs -- a few pixels wide at full
# resolution, gone at 256.
#
# It is also SMALLER: 4 slots x 3 slices x 336px = 1.35 MB/study (~6 GB) against the
# current 4 x 9 x 256 = 2.36 MB (~10 GB).
#
# Attach the competition only. CPU. ~40 min.
# ============================================================
!pip install -q pydicom opencv-python-headless

import sys, glob, os, numpy as np
CODE = "/kaggle/working/rsna-knee"
!rm -rf $CODE && git clone -q https://github.com/IamWasee/rsna-knee.git $CODE
sys.path.insert(0, f"{CODE}/src")

CACHE = "/kaggle/working/cache_336"

# N_ANCHORS is a module constant, so one anchor group is requested by asking for
# 3 slices; preprocess.py records what it actually did in cache_manifest.json and
# inference now refuses to run against a cache built differently.
!python $CODE/src/preprocess.py --out $CACHE --limit 20 --workers 4 \
    --size 336 --crop-mm 130 --anchors 1

import json
man = json.load(open(f"{CACHE}/cache_manifest.json"))
print("\nmanifest:", man)
v = np.load(sorted(glob.glob(f"{CACHE}/*.npy"))[0])
print(f"shape {v.shape}  ->  {v.nbytes/1e6:.2f} MB/study, "
      f"{v.nbytes*4407/1e9:.1f} GB total")

import matplotlib.pyplot as plt
names = ["Sag-FS", "Cor-FS", "Ax-FS", "Sag-T1"]
n = v.shape[1]
fig, ax = plt.subplots(v.shape[0], n, figsize=(2.2*n, 2.2*v.shape[0]))
ax = np.atleast_2d(ax)
for s in range(v.shape[0]):
    for k in range(n):
        ax[s, k].imshow(v[s, k], cmap="gray"); ax[s, k].axis("off")
        ax[s, k].set_title(f"{names[s] if s < 4 else s} {k}", fontsize=7)
plt.tight_layout(); plt.show()

# Full run -- uncomment once the images look right.
# !python $CODE/src/preprocess.py --out $CACHE --workers 4 --size 336 --crop-mm 130 --anchors 1
