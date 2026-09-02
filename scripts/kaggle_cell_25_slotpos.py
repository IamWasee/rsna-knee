# ============================================================
# RSNA Knee — position-aware head + the better Fracture reader, 5 folds
#
# Two changes at once, which I would normally refuse. Quota is 6h and this is the
# only 5-fold run that fits before Friday's reset, so the confound is deliberate
# and stated rather than accidental:
#
#   --head slotpos   The anchor groups are ordered along the slice stack, and after
#                    laterality normalisation a sagittal stack runs medial to
#                    lateral -- so the group index IS the medial-lateral axis. The
#                    old head averaged the groups before seeing them, destroying
#                    the only signal separating a medial finding from a lateral
#                    one. Four labels are medial/lateral pairs, and Lateral
#                    Meniscus (0.717) and Lateral OA (0.787) are among our worst.
#
#   --borrow         Fracture is our weakest label at 0.672, and its labels are
#                    the weakest too: pilkwang reads Fracture at 0.871 against
#                    stevenleehans' 0.793 on the 58 gold studies -- four times the
#                    noise on that label. Everywhere else stevenleehans wins, so
#                    this is one column, not a table swap.
#
# Comparable to full-v3's slot arm (0.789 over 5 folds) because the fold split is
# md5-derived and therefore identical across runs. If this lands below 0.789 the
# next run separates the two; if above, both stay and we attribute later.
#
# 10 epochs rather than 12: full-v3 peaked at epoch 7-9 on every fold, and the two
# extra epochs cost 40 minutes we do not have.
#
# Attach: competition, cache-v3, stevenleehans labels, pilkwang labels, dinov2.
# GPU, Internet on. ~2.8h.
# ============================================================
!pip install -q timm transformers

import sys, os, time
CODE = "/kaggle/working/rsna-knee"
!rm -rf $CODE && git clone -q https://github.com/IamWasee/rsna-knee.git $CODE
sys.path.insert(0, f"{CODE}/src")
from kaggle_paths import find, describe

v3 = [f for f in find(suffix=".npy") if "cache_v3" in f]
if not v3:
    describe(); raise SystemExit("attach the cache-v3 notebook output")
CACHE = os.path.dirname(v3[0])

lab = find(filename="llm_labels_v4_blend.csv")
frac = find(filename="report_labels_v2.csv")
if not lab:
    describe(); raise SystemExit("attach stevenleehans/rsna-knee-llm-report-labels")
if not frac:
    describe(); raise SystemExit("attach pilkwang/rsna-knee-llm-labels")

dino = [os.path.dirname(p) for p in find(filename="config.json") if "dinov2" in p.lower()]
if not dino:
    describe(); raise SystemExit("attach metaresearch/dinov2/PyTorch/small/1")

assert os.path.exists(f"{CACHE}/study_meta.csv"), "no study_meta.csv -- folds not scanner-grouped"
print(f"cache:    {CACHE}\nlabels:   {lab[0]}\nfracture: {frac[0]}")

t0 = time.time()
!python $CODE/src/train.py --cache "$CACHE" --labels "{lab[0]}" \
    --borrow "{frac[0]}:Fracture" \
    --backbone "dinov2:{dino[0]}" --size 288 --folds 5 --epochs 10 --batch 8 \
    --lr 1e-3 --lr-backbone 8e-6 --unfreeze-last 6 --weight-decay 0.02 \
    --head slotpos --pool focal --out /kaggle/working/w_slotpos
print(f"\nelapsed {(time.time()-t0)/3600:.2f} h")

print("\nBaselines on this cache, same folds, 5-fold pooled OOF:")
print("  shared + GAP ......... 0.786")
print("  slot + focal ......... 0.789     <- this run's control")
print("  50/50 blend of both .. 0.797     -> leaderboard 0.865")
print("\nRemember the offset: out-of-fold reads about 0.068 BELOW the leaderboard")
print("on this cache, so read a small gain here as a larger one there.")
