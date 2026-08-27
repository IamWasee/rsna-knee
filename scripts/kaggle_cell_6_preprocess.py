# ============================================================
# RSNA Knee — Preprocessing v2: geometric slice order + physical crop
#
# Three fixes over v1, each measured by other competitors:
#   1. slices sorted by physical position, not filename (filenames are random UIDs
#      and match anatomical order ~5% of the time)
#   2. crop to a fixed 140mm field before resizing, so every study is at the same
#      physical scale (field of view spans 71 values, median 160mm)
#   3. 3 anchors x 3 adjacent slices, not 12 evenly spread (+0.018 measured)
#
# Attach the competition. GPU not needed. Delete the os.walk starter cell.
# ============================================================
!pip install -q pydicom opencv-python-headless

import sys, glob, os, numpy as np
CODE = "/kaggle/working/rsna-knee"
!rm -rf $CODE && git clone -q https://github.com/IamWasee/rsna-knee.git $CODE
sys.path.insert(0, f"{CODE}/src")

CACHE = "/kaggle/working/cache_v2"

# 1. SMOKE TEST -- 20 studies, ~2 min. Look at the pictures before the long run.
!python $CODE/src/preprocess.py --out $CACHE --limit 20 --workers 4

files = sorted(glob.glob(f"{CACHE}/*.npy"))
print(f"\n{len(files)} volumes cached")
v = np.load(files[0])
print(f"shape {v.shape} dtype {v.dtype}   (slots, 9 slices, H, W)")
print(f"empty slots: {(v.reshape(v.shape[0], -1).max(1) == 0).sum()}/{v.shape[0]}")

import matplotlib.pyplot as plt
names = ["Sag-FS", "Cor-FS", "Ax-FS", "Sag-T1"]
fig, ax = plt.subplots(v.shape[0], 9, figsize=(18, 2.1 * v.shape[0]))
ax = np.atleast_2d(ax)
for s in range(v.shape[0]):
    for k in range(9):
        ax[s, k].imshow(v[s, k], cmap="gray"); ax[s, k].axis("off")
        if k == 0:
            ax[s, k].set_ylabel(names[s] if s < 4 else f"slot{s}")
        ax[s, k].set_title(f"{names[s] if s < 4 else s} a{k//3}s{k%3}", fontsize=7)
plt.tight_layout(); plt.show()

# WHAT TO CHECK: within each row, slices 0-1-2 should look like NEIGHBOURS -- a
# smooth progression, not three unrelated views. That is the slice-ordering fix
# working. The knee should also fill the frame more than it did in v1.

# 2. Full run -- ~40 min, ~10 GB. Only after the images above look right.
# !python $CODE/src/preprocess.py --out $CACHE --workers 4
