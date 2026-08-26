# ============================================================
# RSNA Knee — Step 6: DICOM -> compact image cache
# GPU not needed. This is disk and CPU bound.
# ============================================================
!pip install -q pydicom opencv-python-headless

import sys, glob, os
CODE = "/kaggle/working/rsna-knee"
!rm -rf $CODE && git clone -q https://github.com/IamWasee/rsna-knee.git $CODE
sys.path.insert(0, f"{CODE}/src")

CACHE = "/kaggle/working/cache"

# 1. SMOKE TEST FIRST -- 20 studies, a couple of minutes. Confirms DICOM reads,
#    series selection, and shapes before committing to the full run.
!python $CODE/src/preprocess.py --out $CACHE --limit 20 --workers 4

# Look at what came out before going further.
import numpy as np
files = sorted(glob.glob(f"{CACHE}/*.npy"))
print(f"\n{len(files)} volumes cached")
if files:
    v = np.load(files[0])
    print(f"shape {v.shape} dtype {v.dtype} min {v.min()} max {v.max()}")
    print(f"all-zero series: {(v.reshape(v.shape[0], -1).max(1) == 0).sum()}/{v.shape[0]}")

    # An image you can actually look at. Blank or noise means something is wrong.
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(v.shape[0], 4, figsize=(10, 3 * v.shape[0]))
    ax = np.atleast_2d(ax)
    for s in range(v.shape[0]):
        for k in range(4):
            ax[s, k].imshow(v[s, k * 3], cmap="gray"); ax[s, k].axis("off")
    plt.tight_layout(); plt.show()

# 2. Full run -- HOURS. Only after the images above look like knees.
# !python $CODE/src/preprocess.py --out $CACHE --workers 4

# 3. Save $CACHE as a Kaggle Dataset so training runs attach it instead of
#    rebuilding it every time.
