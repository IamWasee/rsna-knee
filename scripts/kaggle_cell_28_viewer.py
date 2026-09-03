# ============================================================
# RSNA Knee — look at what the model actually sees
#
# Works on any cache that is attached: the current one (4 sequences x 9 slices) or
# the new per-plane ones (1 sequence x 24 slices). It draws exactly the pixels the
# network is fed -- after the physical crop, after the resize, after the left/right
# normalisation -- not the original scan. If something looks wrong here, the model
# is learning from something wrong.
#
# Three views:
#   1. every slice of one study, laid out as a grid
#   2. a left knee and a right knee side by side, to confirm they now read alike
#   3. the average knee across many studies -- a blurry but recognisable joint
#      means the crop and alignment are consistent; a smear means they are not
#
# CPU, ~2 min. Attach any cache notebook output (cache-v3 or cache-planes).
# ============================================================
import sys, os, glob, json
import numpy as np, pandas as pd
import matplotlib.pyplot as plt

CODE = "/kaggle/working/rsna-knee"
!rm -rf $CODE && git clone -q https://github.com/IamWasee/rsna-knee.git $CODE
sys.path.insert(0, f"{CODE}/src")
from kaggle_paths import find, describe

caches = sorted({os.path.dirname(f) for f in find(suffix=".npy")})
if not caches:
    describe(); raise SystemExit("attach a cache notebook output")
print("caches found:")
for c in caches:
    n = len(glob.glob(f"{c}/*.npy"))
    m = json.load(open(f"{c}/cache_manifest.json")) if os.path.exists(f"{c}/cache_manifest.json") else {}
    print(f"  {c}  ({n} studies, {m.get('slots','?')} seq x {m.get('n_slices','?')} slices "
          f"@ {m.get('size','?')}px)")

NAMES = ["Sagittal fluid", "Coronal fluid", "Axial fluid", "Sagittal T1"]

for CACHE in caches:
    man = json.load(open(f"{CACHE}/cache_manifest.json"))
    only = man.get("only_slot")
    label = NAMES[only] if only is not None else "all four sequences"
    print("\n" + "=" * 70)
    print(f"{os.path.basename(CACHE)} -- {label}")
    print("=" * 70)

    meta = None
    if os.path.exists(f"{CACHE}/study_meta.csv"):
        meta = pd.read_csv(f"{CACHE}/study_meta.csv")
        meta["s"] = meta["side"].astype(str).str.upper().str[0]

    files = sorted(glob.glob(f"{CACHE}/*.npy"))

    # ---- 1. every slice of one study -------------------------------------
    v = np.load(files[0])
    for s in range(v.shape[0]):
        n = v.shape[1]
        cols = min(n, 12)
        rows = int(np.ceil(n / cols))
        fig, ax = plt.subplots(rows, cols, figsize=(1.5 * cols, 1.6 * rows))
        ax = np.atleast_2d(ax)
        for k in range(rows * cols):
            a = ax[k // cols, k % cols]
            a.axis("off")
            if k < n:
                a.imshow(v[s, k], cmap="gray")
                a.set_title(f"slice {k}", fontsize=6)
        title = NAMES[only] if only is not None else NAMES[s] if s < 4 else f"seq {s}"
        fig.suptitle(f"one study, {title} -- front of knee to back", fontsize=11)
        plt.tight_layout(); plt.show()

    # ---- 2. a left knee next to a right knee ------------------------------
    if meta is not None and {"L", "R"} <= set(meta["s"]):
        mid = v.shape[1] // 2
        fig, ax = plt.subplots(1, 2, figsize=(7, 3.6))
        for j, side in enumerate(("L", "R")):
            uid = meta[meta["s"] == side]["StudyInstanceUID"].iloc[0]
            p = f"{CACHE}/{uid}.npy"
            if os.path.exists(p):
                ax[j].imshow(np.load(p)[0, mid], cmap="gray")
            ax[j].set_title(f"scanned as a {side} knee", fontsize=10)
            ax[j].axis("off")
        fig.suptitle("both should now read as the SAME side", fontsize=11)
        plt.tight_layout(); plt.show()

    # ---- 3. the average knee ---------------------------------------------
    # If the crop and alignment are consistent this is a blurry but obvious
    # joint. If it is a featureless smear, studies are not landing in the same
    # place and every downstream number is built on that.
    mid = v.shape[1] // 2
    stack = [np.load(f)[0, mid].astype(np.float32) for f in files[:300]]
    avg = np.mean(stack, axis=0)
    sd = np.std(stack, axis=0)
    fig, ax = plt.subplots(1, 2, figsize=(7, 3.6))
    ax[0].imshow(avg, cmap="gray"); ax[0].set_title(f"average of {len(stack)} knees", fontsize=10)
    ax[1].imshow(sd, cmap="magma"); ax[1].set_title("where they disagree", fontsize=10)
    for a in ax:
        a.axis("off")
    fig.suptitle("a recognisable joint = consistent alignment", fontsize=11)
    plt.tight_layout(); plt.show()
    print(f"contrast of the average image: {avg.std():.1f} "
          f"(higher means the knees line up; a smear reads near 0)")
