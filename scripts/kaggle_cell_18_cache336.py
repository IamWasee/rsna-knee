# ============================================================
# RSNA Knee — 288px cache, resolution as the only change
#
# Resolution 256 -> 288, everything else held: 3 anchor groups, 140mm crop, 4 slots.
# This is the configuration a competitor measured at 0.7815 ("288px, 140mm crop,
# center-9 of 24") and it isolates resolution as one variable.
#
# Not 336 with one anchor group, despite that being what the top public notebooks
# use. One anchor takes the centre of the stack; on a sagittal knee the cruciates
# are central but the MENISCI ARE PERIPHERAL, and menisci are four of the twelve
# labels. Keeping three anchors preserves that coverage; 336px with three would be
# 17.5 GB against a 19.5 GB working limit.
#
# 4 slots x 9 slices x 288px = 2.99 MB/study, ~12.9 GB.
#
# Attach the competition only. CPU. ~40 min.
# ============================================================
!pip install -q pydicom opencv-python-headless

import sys, glob, os, numpy as np
CODE = "/kaggle/working/rsna-knee"
!rm -rf $CODE && git clone -q https://github.com/IamWasee/rsna-knee.git $CODE
sys.path.insert(0, f"{CODE}/src")

CACHE = "/kaggle/working/cache_288"

# preprocess.py records what it actually did in cache_manifest.json; infer.py now
# refuses to run against a cache built with different constants.
!python $CODE/src/preprocess.py --out $CACHE --limit 20 --workers 4 \
    --size 288 --crop-mm 140 --anchors 3

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
# !python $CODE/src/preprocess.py --out $CACHE --workers 4 --size 288 --crop-mm 140 --anchors 3
