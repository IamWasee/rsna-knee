# ============================================================
# RSNA Knee — one encoder family, fold 0. Sized to fit the hour that is left.
#
# The six-family bake-off needs ~5h and the allowance does not reset until
# tomorrow evening, so this runs the single most informative arm: __TAG__.
#
# CoAtNet is the arm worth spending the last hour on. It is the family the public
# frontier notebook built its best single model from, and it is the one I wrongly
# reported as unable to fit on this hardware -- a claim measured at full precision
# on 36 images per study where training uses half precision on 12, about six times
# the real requirement. The memory probe since measured it at 6 of 15 GB without
# chunking, 2 GB with.
#
# 8 epochs, not 10: at ~5 min an epoch this has to finish inside 1.5 hours or it
# is killed and returns nothing. DINOv2 peaked at epoch 7-9 on every fold.
#
# Reference: DINOv2-small, this cache, fold 0, slot+focal = 0.804. Anything within
# ~0.02 is a keeper -- an ensemble needs members that DISAGREE, not a winner.
#
# Attach: competition, cache-v3, stevenleehans labels, dinov2. GPU. ~1h.
# ============================================================
!pip install -q timm transformers

import sys, os, time
CODE = "/kaggle/working/rsna-knee"
!rm -rf $CODE && git clone -q https://github.com/IamWasee/rsna-knee.git $CODE
sys.path.insert(0, f"{CODE}/src")
from kaggle_paths import find, describe

v3 = [f for f in find(suffix=".npy") if "cache_v3" in f]
lab = find(filename="llm_labels_v4_blend.csv")
if not (v3 and lab):
    describe(); raise SystemExit("attach cache-v3 and the stevenleehans labels")
CACHE = os.path.dirname(v3[0])

BB, BATCH, CHUNK, TAG = "__BB__", __BATCH__, __CHUNK__, "__TAG__"

# Rehearse first: the real loop, real data, real precision, three steps. It costs
# a minute and tells us the per-step cost before committing the hour -- and it is
# the same code path as training, unlike every hand-written probe I wrote this
# week, each of which drifted from what training actually does.
print("=" * 64 + "\nREHEARSAL\n" + "=" * 64)
!python $CODE/src/train.py --cache "$CACHE" --labels "{lab[0]}" \
    --backbone "$BB" --size 288 --only-fold 0 --batch $BATCH \
    --encoder-chunk $CHUNK --grad-checkpoint --dry-run 3 \
    --head slot --pool focal --out /kaggle/working/rehearse 2>&1 | tail -7

print("\n" + "=" * 64 + f"\nTRAINING {TAG}\n" + "=" * 64)
t0 = time.time()
!python $CODE/src/train.py --cache "$CACHE" --labels "{lab[0]}" \
    --backbone "$BB" --size 288 --only-fold 0 --epochs 8 --batch $BATCH \
    --encoder-chunk $CHUNK --grad-checkpoint \
    --lr 1e-3 --lr-backbone 5e-5 --weight-decay 0.02 \
    --head slot --pool focal --out /kaggle/working/arm_$TAG
print(f"\nelapsed {(time.time()-t0)/60:.0f} min")
print("\nDINOv2-small, same cache, same fold, same head: 0.804")
print("Within ~0.02 is a keeper. What an ensemble needs is disagreement.")
