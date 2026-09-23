# ============================================================
# RSNA Knee — sagittal, 16 epochs AND --sharpen-to source, five folds.
#
# Both changes were closed on the OOF ruler and both win on the 58
# radiologist-labelled studies, which is the ruler the leaderboard uses. Same
# encoder, resolution, slices, cache and labels throughout; one flag differs
# per pair:
#
#                                         OOF ruler   gold 58
#   planes-5fold sag   8 ep, gold-sharp    0.7811     0.822
#   planes-16ep sag   16 ep, gold-sharp    0.7777     0.841   (+0.019)
#   planes-fixed sag   8 ep, source        0.7802     0.846   (+0.024)
#   sharpen-source     8 ep, source          --       0.846   (same seed rerun)
#
# Every one of five folds beat every one of the baseline's five, for both. They
# have never been combined. This run combines them, and is judged on gold: the
# number to beat is 0.846. Gold is never used for epoch selection, so it is an
# honest held-out score; n=58, so read the PAIRED gap against the table above,
# not the absolute level.
#
# ~46 min/fold from the leak-folds rehearsal (2.9 min/epoch at 8 epochs, with
# validation), ~3.9h for five, against Kaggle's 12h. Gated at 300 min and killed
# at 330 -- the last run with no ceiling hung and burned twelve hours.
#
# Attach: competition, cache-sag, stevenleehans labels, dinov2. GPU, Internet on.
# ============================================================
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
    if d.endswith("cache_sag"):
        cache = d; break
if cache is None:
    describe(); raise SystemExit("missing cache_sag")
man = json.load(open(f"{cache}/cache_manifest.json"))
if man.get("slots") != 1 or man.get("only_slot") != 0:
    raise SystemExit(f"cache_sag is not the one-sequence sagittal cache: {man}")
SIZE, NSL = str(man["size"]), str(man["n_slices"])
print(f"cache_sag: {SIZE}px, {NSL} slices, band {man.get('band')}")

# Identical to planes-fixed except --epochs.
COMMON = ["--cache", cache, "--labels", lab[0], "--backbone", f"dinov2:{dino[0]}",
          "--size", SIZE, "--slots", "1", "--n-slices", NSL, "--folds", "5",
          "--head", "shared", "--pool", "focal", "--batch", "8", "--epochs", "16",
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
run(["--dry-run", "4", "--max-minutes", "300", "--out", "/kaggle/working/rehearse"],
    "rehearsal", 20)
OUT = "/kaggle/working/plane_s16_sag"
run(["--out", OUT], "five folds, 16 epochs, source", 330)
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
# seed, deterministic grouping), so fold i here and fold i of planes-fixed hold
# the same studies. A mean edge over the unpaired 0.846 would call 0.847 a gain;
# seed-to-seed variance on gold has never been measured, so ask for a margin and
# for the direction to hold in most folds.
BASE = [0.840, 0.844, 0.852, 0.851, 0.842]   # planes-fixed sag, gold at best-OOF epoch
d = [a - b for a, b in zip(g, BASE)]
up = sum(x > 0 for x in d)
print("\n" + "=" * 70)
print(f"GOLD 58, mean of five folds: {m:.3f}   (fold range {min(g):.3f}-{max(g):.3f})")
print("  planes-5fold sag   8 ep, gold-sharp   0.822")
print("  planes-16ep sag   16 ep, gold-sharp   0.841")
print("  planes-fixed sag   8 ep, source       0.846")
print(f"  THIS              16 ep, source       {m:.3f}")
print("\npaired against planes-fixed, fold by fold: " +
      "  ".join(f"{x:+.3f}" for x in d) + f"   mean {np.mean(d):+.4f}, {up}/5 up")
print()
if np.mean(d) >= 0.005 and up >= 4:
    print("The two combine. Same settings on coronal and axial next, then submit.")
elif np.mean(d) > 0:
    print("Ahead, but inside what seed noise could do. Not enough to double the")
    print("training cost on the other planes; keep 8 epochs with source.")
else:
    print("No better than source at 8 epochs. They do not stack -- keep 8 epochs.")
