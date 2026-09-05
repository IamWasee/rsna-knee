# ============================================================
# RSNA Knee — the whole slice stack, one plane. Independent of the bake-off.
#
# This is the largest published lever we have never tested, and it needs nothing
# from the encoder comparison running alongside it: Dazcona et al. ran exactly
# this ablation on knee MRI with everything else held, and all-slices pooling
# beat fixed sampling by +0.037 macro and +0.087 ON MENISCUS. The gain lands
# almost entirely on the small focal finding and barely moves the diffuse ones --
# which is the exact shape of our deficit. Lateral Meniscus is 0.717 for us.
#
# Today we load 9 slices of roughly 24, so a tear living in a skipped slice is
# invisible and no amount of attention recovers it. cache___TAG__ holds all 24.
#
# Geometry: 1 sequence x 24 slices -> 8 three-channel images per study, against
# 12 for the four-sequence cache. So this is CHEAPER per study, not dearer.
#
# --head shared, not slot: with a single sequence there is nothing for the
# per-diagnosis slot attention to choose between -- a softmax over one option is
# always 1.0. Attention pooling over the eight slice groups is the real choice
# here, and that is what "shared" does.
#
# Attach: competition, cache-__TAG__, stevenleehans labels, dinov2. GPU. ~1h.
# ============================================================
!pip install -q timm transformers

import sys, os, time
CODE = "/kaggle/working/rsna-knee"
!rm -rf $CODE && git clone -q https://github.com/IamWasee/rsna-knee.git $CODE
sys.path.insert(0, f"{CODE}/src")
from kaggle_paths import find, describe

TAG = "__TAG__"
hits = [f for f in find(suffix=".npy") if f"cache_{TAG}" in f]
lab = find(filename="llm_labels_v4_blend.csv")
dino = [os.path.dirname(p) for p in find(filename="config.json") if "dinov2" in p.lower()]
if not (hits and lab and dino):
    describe(); raise SystemExit(f"attach cache-{TAG}, stevenleehans labels, dinov2")
CACHE = os.path.dirname(hits[0])

import json, numpy as np
man = json.load(open(f"{CACHE}/cache_manifest.json"))
print(f"cache: {CACHE}")
print(f"manifest: {json.dumps(man)}")
if man.get("n_slices") != 24 or man.get("slots") != 1:
    raise SystemExit(f"expected 1 sequence x 24 slices, got "
                     f"{man.get('slots')} x {man.get('n_slices')} -- wrong cache")

COMMON = ["--cache", CACHE, "--labels", lab[0], "--backbone", f"dinov2:{dino[0]}",
          "--size", "288", "--slots", "1", "--n-slices", "24",
          "--only-fold", "0", "--head", "shared", "--pool", "focal"]

print("\n" + "=" * 64 + "\nREHEARSAL (gated)\n" + "=" * 64)
!python $CODE/src/train.py {" ".join(COMMON)} --batch 8 --epochs 8 \
    --grad-checkpoint --dry-run 4 --max-minutes 75 \
    --out /kaggle/working/rehearse 2>&1 | tail -8

print("\n" + "=" * 64 + f"\nTRAINING -- all 24 slices, {TAG}\n" + "=" * 64)
t0 = time.time()
!python $CODE/src/train.py {" ".join(COMMON)} --batch 8 --epochs 8 \
    --grad-checkpoint --lr 1e-3 --lr-backbone 8e-6 --unfreeze-last 6 \
    --weight-decay 0.02 --out /kaggle/working/allslice_$TAG
print(f"\nelapsed {(time.time()-t0)/60:.0f} min")

print("\n" + "=" * 64)
print("Compare against DINOv2 on the 4-sequence, 9-slice cache, fold 0: 0.804.")
print("That is NOT a fair fight -- this arm sees one sequence instead of four.")
print("The question is not whether it wins outright, it is whether MENISCUS and")
print("the other small findings improve, and whether it disagrees enough with")
print("the four-sequence model to be worth blending with it.")
