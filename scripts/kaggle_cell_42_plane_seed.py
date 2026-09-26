# ============================================================
# RSNA Knee — the 16-epoch source arm again, with a different --seed.
# Pushed per plane: --sub PLANE=ax --sub SEED=43.
#
# Two jobs.
#
# 1. MEASURE SEED NOISE ON GOLD. Every verdict so far used a +0.005 bar picked
#    by judgement: seed-to-seed variation on the 58 radiologist-labelled studies
#    has never been measured. The earlier "replicate" (sharpen-source) was the
#    same seed, identical to 0.001. Here only the seed changes: GroupKFold takes
#    no seed, so fold i holds the same studies as fold i of the seed-42 arm, and
#    the paired difference is training randomness alone -- head init, batch
#    order, augmentation draws.
#
# 2. A SECOND ARM PER PLANE. Averaging two seeds of the same recipe usually
#    buys a little on the leaderboard. The 0.909 submission has one per plane.
#
# The seed-42 baseline is read exactly from the attached <plane>-16ep-source
# checkpoints. Gated at 300 min, killed at 330; 16 epochs took ~3.2h a plane.
#
# Attach: competition, cache-<plane>, <plane>-16ep-source, stevenleehans
# labels, dinov2. GPU, internet on.
# ============================================================
PLANE = "__PLANE__"
SEED = "__SEED__"
SLOT = {"sag": 0, "cor": 1, "ax": 2}[PLANE]
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

# Identical to the 16-epoch seed-42 arm except --seed.
COMMON = ["--cache", cache, "--labels", lab[0], "--backbone", f"dinov2:{dino[0]}",
          "--size", SIZE, "--slots", "1", "--n-slices", NSL, "--folds", "5",
          "--head", "shared", "--pool", "focal", "--batch", "8", "--epochs", "16",
          "--lr", "1e-3", "--lr-backbone", "8e-6", "--unfreeze-last", "6",
          "--weight-decay", "0.02", "--sharpen-to", "source", "--seed", SEED]

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
run(["--dry-run", "4", "--max-minutes", "300", "--out", "/kaggle/working/rehearse"],
    "rehearsal", 20)
OUT = f"/kaggle/working/plane_s16s{SEED}_{PLANE}"
run(["--out", OUT], f"{PLANE}: five folds, 16 epochs, source, seed {SEED}", 330)
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
# The seed-42 arm, exactly, from its own checkpoints. No fallback: this run
# measures a difference of a few thousandths, and the log's 3-decimal rounding
# with its tied epochs (up to 0.011 on ax-24 fold 1) would swamp it.
import re as _re
base_ck = {}
for p in find(suffix=".pt"):
    if f"/plane_s16_{PLANE}/" in p:
        mm = _re.search(r"fold(\d)\.pt$", p)
        if mm:
            base_ck[int(mm.group(1))] = float(torch.load(p, map_location="cpu",
                                              weights_only=False)["gold_auc"])
if sorted(base_ck) != [0, 1, 2, 3, 4]:
    raise SystemExit(f"seed-42 checkpoints for {PLANE} not attached ({sorted(base_ck)}); "
                     f"attach {PLANE}-16ep-source -- this arm is still saved in {OUT}")
BASE = [base_ck[f] for f in range(5)]
d = [a - b for a, b in zip(g, BASE)]
sd = float(np.std(d, ddof=1))
print("\n" + "=" * 70)
print(f"{PLANE}, 16 ep, source -- GOLD 58 per fold, exact")
print(f"  seed 42    " + "  ".join(f"{b:.4f}" for b in BASE) + f"   mean {np.mean(BASE):.4f}")
print(f"  seed {SEED}    " + "  ".join(f"{x:.4f}" for x in g) + f"   mean {m:.4f}")
print(f"  difference " + "  ".join(f"{x:+.4f}" for x in d) + f"   mean {np.mean(d):+.4f}")
print(f"\nseed noise on this plane: sd of the per-fold difference {sd:.4f}")
print(f"  -> sd of a five-fold mean difference between two seeds ~ {sd/np.sqrt(5):.4f}")
print(f"  the +0.005 bar used so far is {0.005/(sd/np.sqrt(5)):.1f} of those")
print("\nThe three planes' numbers are pooled offline for the real estimate;")
print("one plane's five differences is too few to trust on its own.")
