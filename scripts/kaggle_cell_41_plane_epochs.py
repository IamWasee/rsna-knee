# ============================================================
# RSNA Knee — does training LONGER still help? One plane, --sharpen-to source.
# Pushed per plane with --sub PLANE=ax --sub EPOCHS=24.
#
# 8 -> 16 epochs was the biggest single win we have found. On the 58
# radiologist-labelled studies, paired fold by fold against 8 epochs:
#   sag +0.0095   cor +0.0238   ax +0.0404        all 5/5 folds up
# and the three 16-epoch arms took the leaderboard from 0.882 to 0.909. Nobody
# has tried more than 16. This asks whether the curve is still rising.
#
# The baseline is read EXACTLY from the 16-epoch arm's own checkpoints (attach
# <plane>-16ep-source): gold_auc at each fold's kept epoch, unrounded. The log
# rounds to 3 decimals and ties there hid which epoch was actually kept.
#
# Same flags as the 16-epoch arm except --epochs; same cache, labels, seed and
# folds, so fold i here and fold i there hold the same studies.
#
# Axial at 16 epochs took 3.2h; 24 epochs ~4.7h. Gated at 420 min, killed at
# 450; each notebook has its own 12h.
#
# Attach: competition, cache-<plane>, <plane>-16ep-source, stevenleehans
# labels, dinov2. GPU, internet on.
# ============================================================
PLANE = "__PLANE__"
EPOCHS = "__EPOCHS__"
SLOT = {"sag": 0, "cor": 1, "ax": 2}[PLANE]
# Fallback only, if the 16-epoch checkpoints are not attached: the verdicts of
# cells 39/40, gold at each fold's kept epoch, rounded to 3 decimals by the log.
LOGGED16 = {"sag": [0.854, 0.852, 0.853, 0.861, 0.857],
            "cor": [0.840, 0.842, 0.850, 0.841, 0.839],
            "ax":  [0.845, 0.846, 0.862, 0.847, 0.861]}[PLANE]
!pip install -q timm transformers

import sys, os, time, glob, json, shlex
CODE = "/kaggle/working/rsna-knee"
!rm -rf $CODE && git clone -q https://github.com/IamWasee/rsna-knee.git $CODE
!git -C $CODE log --oneline -1
sys.path.insert(0, f"{CODE}/src")
from kaggle_paths import find, describe

# The model is loaded from a local directory, so nothing in training needs the
# network. The 2026-09-20 run froze in the 5-20 s window where the encoder is
# constructed, before any weight was loaded, with internet on; a hub call with
# no timeout is the leading suspect. Offline mode removes it at no cost -- the
# submission notebook has always run this way.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

lab = find(filename="llm_labels_v4_blend.csv")
dino = [os.path.dirname(p) for p in find(filename="config.json") if "dinov2" in p.lower()]
if not (lab and dino):
    describe(); raise SystemExit("attach stevenleehans labels and dinov2")

cache = None
for f in find(suffix=".npy"):
    d = os.path.dirname(f)
    if d.endswith(f"cache_{PLANE}"):
        cache = d; break
if cache is None:
    describe(); raise SystemExit(f"missing cache_{PLANE}")
man = json.load(open(f"{cache}/cache_manifest.json"))
if man.get("slots") != 1 or man.get("only_slot") != SLOT:
    raise SystemExit(f"cache_{PLANE} is not the one-sequence {PLANE} cache: {man}")
SIZE, NSL = str(man["size"]), str(man["n_slices"])
print(f"cache_{PLANE}: {SIZE}px, {NSL} slices, band {man.get('band')}")

# Identical to the 16-epoch arm except --epochs.
COMMON = ["--cache", cache, "--labels", lab[0], "--backbone", f"dinov2:{dino[0]}",
          "--size", SIZE, "--slots", "1", "--n-slices", NSL, "--folds", "5",
          "--head", "shared", "--pool", "focal", "--batch", "8", "--epochs", EPOCHS,
          "--lr", "1e-3", "--lr-backbone", "8e-6", "--unfreeze-last", "6",
          "--weight-decay", "0.02", "--sharpen-to", "source", "--seed", "42"]

def run(extra, label, ceiling):
    """train.py under a wall-clock ceiling; stop on any non-zero exit.

    ! keeps the log forwarding that has always worked here. _exit_code gives
    the status ! otherwise swallows, and the shell's timeout exits 124 on a hang.
    """
    print("\n" + "=" * 70 + f"\n{label}\n" + "=" * 70, flush=True)
    args = " ".join(shlex.quote(a) for a in COMMON + extra)
    secs = ceiling * 60
    t = time.time()
    !timeout -k 60 {secs} python -u $CODE/src/train.py {args}
    code = _exit_code
    if code == 124:
        raise SystemExit(f"{label} hit its {ceiling} min ceiling after "
                         f"{(time.time()-t)/60:.0f} min -- a hang, not a slow run")
    if code != 0:
        raise SystemExit(f"{label} exited {code}")

t0 = time.time()
run(["--dry-run", "4", "--max-minutes", "420", "--out", "/kaggle/working/rehearse"],
    "rehearsal", 20)
OUT = f"/kaggle/working/plane_e{EPOCHS}_{PLANE}"
run(["--out", OUT], f"{PLANE}: five folds, {EPOCHS} epochs, source", 450)
print(f"\nelapsed {(time.time()-t0)/60:.0f} min")

# ------------------------------------------------------------- the verdict
# Gold per fold, from the checkpoints train.py kept (best OOF epoch each).
import numpy as np, torch
g = []
for p in sorted(glob.glob(f"{OUT}/fold*.pt")):
    ck = torch.load(p, map_location="cpu", weights_only=False)
    g.append(float(ck["gold_auc"]))
    print(f"  {os.path.basename(p)}  gold {ck['gold_auc']:.3f}   OOF {ck['oof_auc']:.3f}")
if len(g) != 5:
    raise SystemExit(f"{len(g)} of 5 folds saved a checkpoint")
m = float(np.mean(g))
# Paired, fold by fold. Folds are identical across these runs (same labels, same
# seed, deterministic grouping), so fold i here and fold i of the 16-epoch arm hold
# the same studies. A mean edge over the unpaired 0.846 would call 0.847 a gain;
# seed-to-seed variance on gold has never been measured, so ask for a margin and
# for the direction to hold in most folds.
# The baseline, exactly, from the 16-epoch arm's checkpoints.
import re as _re
base_ck = {}
for p in find(suffix=".pt"):
    if f"/plane_s16_{PLANE}/" in p:
        mm = _re.search(r"fold(\d)\.pt$", p)
        if mm:
            base_ck[int(mm.group(1))] = float(torch.load(p, map_location="cpu",
                                              weights_only=False)["gold_auc"])
if sorted(base_ck) == [0, 1, 2, 3, 4]:
    BASE = [base_ck[f] for f in range(5)]
    print(f"\nbaseline: plane_s16_{PLANE} checkpoints, exact")
else:
    BASE = LOGGED16
    print(f"\nbaseline: 16-epoch checkpoints not attached ({sorted(base_ck)}); "
          f"using the logged, 3-decimal values")

d = [a - b for a, b in zip(g, BASE)]
up = sum(x > 0 for x in d)
print("\n" + "=" * 70)
print(f"{PLANE} {EPOCHS} ep: GOLD 58, mean of five folds: {m:.4f}   (fold range {min(g):.3f}-{max(g):.3f})")
print(f"  {PLANE} 16 ep, source: {np.mean(BASE):.4f}   ({'  '.join(f'{b:.4f}' for b in BASE)})")
print(f"\n{EPOCHS} ep paired against 16 ep, fold by fold: " +
      "  ".join(f"{x:+.3f}" for x in d) + f"   mean {np.mean(d):+.4f}, {up}/5 up")
print()
if np.mean(d) >= 0.005 and up >= 4:
    print(f"Longer still helps on {PLANE}. Carry {EPOCHS} epochs to the other planes.")
elif np.mean(d) > 0:
    print("Ahead, but inside what seed noise could do. The curve has flattened:")
    print("16 epochs is enough, and costs a third less.")
else:
    print(f"No better than 16 epochs -- past the peak. Keep 16.")
