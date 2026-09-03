# ============================================================
# RSNA Knee — per-plane caches with the whole slice stack
#
# The two biggest levers left in the published evidence both need this cache, and
# neither is a bigger backbone:
#
#   ALL SLICES.   Dazcona et al. on MRNet, everything else held: all slices
#                 max-pooled beats fixed sampling by +0.037 macro and +0.087 on
#                 MENISCUS. The gain lands almost entirely on the small focal
#                 finding, which is the exact shape of our deficit -- Lateral
#                 Meniscus 0.710, MCL 0.732. We currently load 9 slices of roughly
#                 24, so a tear on a skipped slice is invisible and no amount of
#                 attention recovers it.
#
#   LATE FUSION.  Same paper: separate models per plane, combined afterwards,
#                 scored 0.9340 against 0.8577 for one shared network fed all
#                 three planes -- +0.076, the largest single number in any of the
#                 research. Our shared encoder across four slots is the losing
#                 configuration.
#
# Why per plane rather than one fat cache: 4 slots x 24 slices x 288px is 35 GB
# against a 20 GB limit. ONE slot x 24 slices is 8.8 GB and fits with room to
# spare -- and it is what late fusion wants anyway.
#
# 8 anchors x 3 adjacent = 24 slices, which covers essentially the whole stack.
#
# Attach the competition only. CPU, Internet on. ~2h for all three.
# ============================================================
!pip install -q pydicom opencv-python-headless

import sys, os, glob, json
import numpy as np, pandas as pd

CODE = "/kaggle/working/rsna-knee"
!rm -rf $CODE && git clone -q https://github.com/IamWasee/rsna-knee.git $CODE
sys.path.insert(0, f"{CODE}/src")

PLANES = [(0, "sag"), (1, "cor"), (2, "ax")]     # Sag-fluid, Cor-fluid, Axial
ARGS = "--size 288 --crop-mm 140 --anchors 8"    # 8 x 3 = 24 slices

# Probe one plane on 30 studies before committing two hours. A slot that is often
# missing would produce mostly-blank volumes, and that is worth knowing now.
PROBE = "/kaggle/working/probe"
!python $CODE/src/preprocess.py --out $PROBE --limit 30 --workers 4 --only-slot 0 {ARGS}
# preprocess.py exiting non-zero does not stop a notebook cell, so an empty probe
# directory here means the command failed above -- most likely this clone of
# master predates the flag being used. Say that instead of an IndexError.
probe_files = sorted(glob.glob(f"{PROBE}/*.npy"))
if not probe_files:
    raise SystemExit(
        "the probe wrote nothing -- read preprocess.py's output above. If it says "
        "'unrecognized arguments', this notebook cloned a commit of master that "
        "predates the flag; push src/ first, then re-run."
    )
v = np.load(probe_files[0])
print(f"\nshape {v.shape} -> {v.nbytes/1e6:.2f} MB/study, "
      f"{v.nbytes*4407/1e9:.1f} GB per plane")
blank = float((v.reshape(v.shape[0]*v.shape[1], -1).max(1) == 0).mean())
print(f"blank slices in this study: {blank:.0%}")
if v.nbytes * 4407 / 1e9 > 18:
    raise SystemExit("one plane already exceeds the working limit; lower --anchors")
!rm -rf $PROBE

for idx, tag in PLANES:
    out = f"/kaggle/working/cache_{tag}"
    print("\n" + "=" * 60 + f"\nplane {idx} -> {out}\n" + "=" * 60)
    !python $CODE/src/preprocess.py --out $out --workers 4 --only-slot {idx} {ARGS}
    n = len(glob.glob(f"{out}/*.npy"))
    gb = sum(os.path.getsize(p) for p in glob.glob(f"{out}/*.npy")) / 1e9
    print(f"{n} studies, {gb:.1f} GB")

print("\n" + "=" * 60)
tot = 0
for _, tag in PLANES:
    out = f"/kaggle/working/cache_{tag}"
    if os.path.exists(f"{out}/cache_manifest.json"):
        m = json.load(open(f"{out}/cache_manifest.json"))
        n = len(glob.glob(f"{out}/*.npy"))
        gb = sum(os.path.getsize(p) for p in glob.glob(f"{out}/*.npy")) / 1e9
        tot += gb
        print(f"{tag}: {n} studies, {gb:.1f} GB, slots={m['slots']} "
              f"n_slices={m['n_slices']} only_slot={m['only_slot']}")
print(f"total {tot:.1f} GB of a 20 GB working limit")
