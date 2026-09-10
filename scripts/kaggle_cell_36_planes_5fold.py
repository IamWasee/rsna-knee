# ============================================================
# RSNA Knee — all three plane specialists, five folds, one notebook
#
# The fold-0 selection over six arms picked these three plus ConvNeXt:
#   plane-sag                 0.8079
#   + plane-cor               0.8229  (+0.0149)
#   + convnext                0.8264  (+0.0036)
#   + plane-ax                0.8278  (+0.0014)
#   + dinov2                  0.8258  (-0.0020)   harmful
#   + seresnext               0.8226  (-0.0033)   harmful
#
# ConvNeXt is left out of THIS run deliberately: it costs ~78 min a fold against
# 22 for a plane, so five folds of it is 6.5h for +0.0036. The three planes are
# 5.5h for the other +0.0199. If there is budget afterwards, the better test is
# ConvNeXt on a PLANE cache rather than on the four-sequence one it was measured
# on -- untested, and the data format is the thing that has been winning.
#
# Each plane sees a different anatomy and the split follows it rather than noise:
#   sagittal  cruciates run front-to-back        ACL 0.73, Baker's 0.91
#   coronal   MCL and medial compartment          MCL 0.77, Med Men 0.86
#   axial     looks down at the kneecap, into     PF OA 0.81, Baker's 0.94
#             the space behind the knee
#
# Fifteen checkpoints land in one output, which is what the submission notebook
# wants. ~5.5h against a 9h limit; each plane is gated at 150 min.
#
# Attach: competition, cache-sag, cache-cor, cache-ax, stevenleehans, dinov2.
# GPU, Internet on.
# ============================================================
!pip install -q timm transformers

import sys, os, time, glob, json
CODE = "/kaggle/working/rsna-knee"
!rm -rf $CODE && git clone -q https://github.com/IamWasee/rsna-knee.git $CODE
sys.path.insert(0, f"{CODE}/src")
from kaggle_paths import find, describe

lab = find(filename="llm_labels_v4_blend.csv")
dino = [os.path.dirname(p) for p in find(filename="config.json") if "dinov2" in p.lower()]
if not (lab and dino):
    describe(); raise SystemExit("attach stevenleehans labels and dinov2")

WANT = "__PLANES__".split(",")
EPOCHS = "__EPOCHS__"
caches = {}
for f in find(suffix=".npy"):
    d = os.path.dirname(f)
    for tag in WANT:
        if d.endswith(f"cache_{tag}") and tag not in caches:
            caches[tag] = d
missing = [t for t in WANT if t not in caches]
if missing:
    describe(); raise SystemExit(f"missing plane cache(s): {missing}")

for tag, c in caches.items():
    man = json.load(open(f"{c}/cache_manifest.json"))
    if man.get("slots") != 1 or man.get("n_slices") != 24:
        raise SystemExit(f"cache_{tag} is {man.get('slots')}x{man.get('n_slices')}, "
                         f"expected 1x24 -- wrong cache attached")
    print(f"{tag}: {c}  slot {man.get('only_slot')}  {len(glob.glob(c+'/*.npy'))} studies")

BB = "__BB__"
BACKBONE = f"dinov2:{dino[0]}" if BB == "dinov2" else BB
LR_BB = "8e-6" if BB == "dinov2" else "5e-5"
COMMON = ["--labels", lab[0], "--backbone", BACKBONE, "--size", "288",
          "--slots", "1", "--n-slices", "24", "--folds", "5",
          "--head", "shared", "--pool", "focal", "--batch", "8", "--epochs", EPOCHS,
          "--lr", "1e-3", "--lr-backbone", LR_BB, "--unfreeze-last", "6",
          "--weight-decay", "0.02"] + [a for a in "__EXTRA__".split() if a]

t0 = time.time()
for tag, cache in caches.items():
    print("\n" + "=" * 70 + f"\n{tag}: rehearsal\n" + "=" * 70, flush=True)
    !python $CODE/src/train.py --cache "{cache}" {" ".join(COMMON)} \
        --dry-run 4 --max-minutes 400 --out /kaggle/working/rehearse 2>&1 | tail -6

    print("\n" + "=" * 70 + f"\n{tag}: five folds\n" + "=" * 70, flush=True)
    !python $CODE/src/train.py --cache "{cache}" {" ".join(COMMON)} \
        --out /kaggle/working/plane___BBTAG___$tag
    print(f"\nelapsed {(time.time()-t0)/60:.0f} min", flush=True)

print("\n" + "=" * 70)
for tag in caches:
    n = len(glob.glob(f"/kaggle/working/plane___BBTAG___{tag}/*.pt"))
    o = os.path.exists(f"/kaggle/working/plane___BBTAG___{tag}/oof.csv")
    print(f"  plane___BBTAG___{tag}: {n} checkpoints, oof.csv {'yes' if o else 'NO'}")
print(f"\ntotal {(time.time()-t0)/3600:.1f} h")
print("Fold 0 gave 0.8278 for the three planes plus ConvNeXt. A 5-fold pooled")
print("number reads about 0.01 lower, and the leaderboard has run ~0.068 ABOVE")
print("our pooled number on two submissions. The submission is what settles it.")
