# ============================================================
# RSNA Knee — one experiment on one plane, judged on gold against our best arm.
#
# The reusable harness for every single-change test from here on. It trains
# the 16-epoch --sharpen-to source recipe (the arms behind 0.909/0.910) with ONE
# change, seed 42 kept so it shares its random draws with the baseline, and
# compares gold-58 fold by fold against the seed-42 arm's exact checkpoints.
#
#   --sub PLANE=ax             sag | cor | ax
#   --sub TAG=slotpos          names the output, plane_<TAG>_<plane>
#   --sub EXTRA="--head slotpos"   the one change, as train.py flags
#   --sub LABELFILE=llm_labels_v4_blend.csv   or another attached label table
#   --sub FOLDS=all            all | 0   (fold 0 alone: ~1/5 the cost, coarser)
#
# Bars, fixed in advance from the measured seed noise (2026-09-27): a five-fold
# paired mean must clear +0.013; a single fold, whose paired SD is ~0.008 per
# fold, must clear +0.016. Both runs share seed 42, so real noise is below
# these -- a result inside the bar means "can't tell", not "no effect".
#
# Attach: competition, cache-<plane>, <plane>-16ep-source, the label table's
# dataset, dinov2. GPU, internet on.
# ============================================================
PLANE = "__PLANE__"
TAG = "__TAG__"
EXTRA = "__EXTRA__".split()
LABELFILE = "__LABELFILE__"
FOLDS = "__FOLDS__"
SLOT = {"sag": 0, "cor": 1, "ax": 2}[PLANE]
BAR = 0.013 if FOLDS == "all" else 0.016
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

lab = find(filename=LABELFILE)
dino = [os.path.dirname(p) for p in find(filename="config.json") if "dinov2" in p.lower()]
if not (lab and dino):
    describe(); raise SystemExit(f"attach the dataset holding {LABELFILE}, and dinov2")

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

# Identical to the 16-epoch seed-42 arm, plus EXTRA.
COMMON = ["--cache", cache, "--labels", lab[0], "--backbone", f"dinov2:{dino[0]}",
          "--size", SIZE, "--slots", "1", "--n-slices", NSL, "--folds", "5",
          "--head", "shared", "--pool", "focal", "--batch", "8", "--epochs", "16",
          "--lr", "1e-3", "--lr-backbone", "8e-6", "--unfreeze-last", "6",
          "--weight-decay", "0.02", "--sharpen-to", "source", "--seed", "42"] + EXTRA \
         + (["--only-fold", "0"] if FOLDS == "0" else [])

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
run(["--dry-run", "4", "--max-minutes", "300" if FOLDS == "all" else "80", "--out", "/kaggle/working/rehearse"],
    "rehearsal", 20)
OUT = f"/kaggle/working/plane_{TAG}_{PLANE}"
run(["--out", OUT], f"{PLANE}: {TAG}, folds={FOLDS}", 330 if FOLDS == "all" else 100)
print(f"\nelapsed {(time.time()-t0)/60:.0f} min")

# ------------------------------------------------------------- the verdict
# Gold per fold, from the checkpoints train.py kept (best OOF epoch each).
import numpy as np, torch, re as _re
sys.path.insert(0, f"{CODE}/src")
from config import LABELS
new, new_pl = {}, {}
for p in sorted(glob.glob(f"{OUT}/fold*.pt")):
    ck = torch.load(p, map_location="cpu", weights_only=False)
    f = int(_re.search(r"fold(\d)", p).group(1))
    new[f], new_pl[f] = float(ck["gold_auc"]), ck.get("gold_per_label") or {}
    print(f"  fold{f}  gold {new[f]:.4f}   OOF {ck['oof_auc']:.3f} (on this run's own targets)")
want = [0, 1, 2, 3, 4] if FOLDS == "all" else [0]
if sorted(new) != want:
    raise SystemExit(f"expected folds {want}, got {sorted(new)}")
base, base_pl = {}, {}
for p in find(suffix=".pt"):
    if f"/plane_s16_{PLANE}/" in p:
        m = _re.search(r"fold(\d)\.pt$", p)
        if m and int(m.group(1)) in want:
            ck = torch.load(p, map_location="cpu", weights_only=False)
            base[int(m.group(1))] = float(ck["gold_auc"])
            base_pl[int(m.group(1))] = ck.get("gold_per_label") or {}
if sorted(base) != want:
    raise SystemExit(f"seed-42 baseline checkpoints for {PLANE} not attached ({sorted(base)})")
d = [new[f] - base[f] for f in want]
print("\n" + "=" * 70)
print(f"{PLANE}  {TAG}  ({' '.join(EXTRA) or 'no flag change'}; labels {LABELFILE})")
print("  baseline  " + "  ".join(f"{base[f]:.4f}" for f in want) + f"   mean {np.mean([base[f] for f in want]):.4f}")
print("  this      " + "  ".join(f"{new[f]:.4f}" for f in want) + f"   mean {np.mean([new[f] for f in want]):.4f}")
print("  paired    " + "  ".join(f"{x:+.4f}" for x in d) + f"   mean {np.mean(d):+.4f}")
if all(base_pl.get(f) and new_pl.get(f) for f in want):
    print("\n  per finding, mean over folds (this - baseline):")
    for c in LABELS:
        v = [new_pl[f].get(c, np.nan) - base_pl[f].get(c, np.nan) for f in want]
        print(f"    {c:<17}{np.nanmean(v):+.3f}")
print()
if np.mean(d) >= BAR:
    print(f"CLEARS the +{BAR} bar.")
elif np.mean(d) > -BAR:
    print(f"Inside +/-{BAR}: can't tell from noise.")
else:
    print(f"WORSE by more than {BAR}.")
