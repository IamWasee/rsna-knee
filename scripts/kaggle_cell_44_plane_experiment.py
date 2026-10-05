# ============================================================
# RSNA Knee — one experiment on one plane, judged on gold against our best arm.
#
# The reusable harness for every single-change test from here on. It trains
# the 16-epoch --sharpen-to source recipe (the arms behind 0.909/0.910) with ONE
# change and seed 42, and compares gold-58 fold by fold against the seed-42 arm's
# exact checkpoints. Same seed does NOT mean shared random draws once the change
# alters how the model is built: a different head consumes the RNG differently,
# so batch order and augmentation diverge from epoch 0 and the pair behaves like
# two seeds. That is exactly what the bars below were measured on.
#
#   --sub PLANE=ax             sag | cor | ax
#   --sub TAG=slotpos          names the output, plane_<TAG>_<plane>
#   --sub EXTRA="--head slotpos"   the one change, as train.py flags
#   --sub LABELFILE=llm_labels_v4_blend.csv   or another attached label table
#   --sub FOLDS=all            all | 0   (fold 0 alone: ~1/5 the cost, coarser)
#   --sub MAXMIN=330           training ceiling AND rehearsal budget, minutes
#                              (100 for FOLDS=0). Keep it under the GPU quota
#                              left: if the quota runs out first, Kaggle kills
#                              the session and every output goes with it, where
#                              this ceiling keeps the folds already done. Raise
#                              it deliberately for a slower change (more of the
#                              encoder trained), never past the quota.
#   --sub BASE=plane_s16       the arm to beat, by output folder: plane_s16 (the
#                              16-epoch source arm, attach <plane>-16ep-source) or
#                              plane_pseudo50 (attach <plane>-pseudo50). Once a
#                              change wins, the next one is judged against it.
#
# Bar, fixed in advance from the measured seed noise (2026-09-27): a five-fold
# paired mean must clear +0.013. FOLDS=0 is a SCREEN, not a verdict: one fold's
# gap carries that fold's own studies, and on this project a fold-0 gap of +0.03
# has shrunk to +0.003 at five folds. A fold-0 pass means "confirm at five
# folds"; nothing is kept or killed on fold 0 alone.
#
# Every run also carries --avg-top 3, which leaves training untouched (copying
# weights draws no random numbers) and saves the average of the three best
# epochs beside the best single one. The verdict above is on the single epoch,
# like for like with the baseline; the averaging is judged separately, inside
# the run, where no seed noise separates the two models.
#
# Attach: competition, cache-<plane>, the BASE arm's notebook, the label
# table's dataset, dinov2. GPU, internet on.
# ============================================================
PLANE = "__PLANE__"
TAG = "__TAG__"
EXTRA = "__EXTRA__".split()
LABELFILE = "__LABELFILE__"
FOLDS = "__FOLDS__"
MAXMIN = int("__MAXMIN__")
BASE = "__BASE__"
assert BASE in ("plane_s16", "plane_pseudo50"), f"unknown BASE {BASE!r}"
SLOT = {"sag": 0, "cor": 1, "ax": 2}[PLANE]
assert FOLDS in ("all", "0"), f"FOLDS must be 'all' or '0', got {FOLDS!r}"
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
          "--weight-decay", "0.02", "--sharpen-to", "source", "--seed", "42",
          "--avg-top", "3"] + EXTRA \
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

# The rehearsal projects high: 264 and 270 min projected, 218 and 210 min run
# (ax-16ep-severity, ax-slotpos). Four timed steps were also noisy enough to
# read 0.41 s/step on the same recipe and refuse ax-pseudo50 at 300 against
# 300. Twenty steps, and a budget equal to the training ceiling it guards.
CEILING = MAXMIN
t0 = time.time()
run(["--dry-run", "20", "--max-minutes", str(CEILING), "--out", "/kaggle/working/rehearse"],
    "rehearsal", 20)
OUT = f"/kaggle/working/plane_{TAG}_{PLANE}"
run(["--out", OUT], f"{PLANE}: {TAG}, folds={FOLDS}", CEILING)
print(f"\nelapsed {(time.time()-t0)/60:.0f} min")

# ------------------------------------------------------------- the verdict
# Gold per fold, from the checkpoints train.py kept (best OOF epoch each).
import numpy as np, torch, re as _re
sys.path.insert(0, f"{CODE}/src")
from config import LABELS
new, new_pl, avg, avg_pl = {}, {}, {}, {}
for p in sorted(glob.glob(f"{OUT}/fold*.pt")):
    ck = torch.load(p, map_location="cpu", weights_only=False)
    f = int(_re.search(r"fold(\d)\.pt$", p).group(1))
    if "averaged_epochs" in ck:
        new[f], new_pl[f] = float(ck["single_gold_auc"]), ck["single_gold_per_label"] or {}
        avg[f], avg_pl[f] = float(ck["gold_auc"]), ck.get("gold_per_label") or {}
        print(f"  fold{f}  gold {new[f]:.4f} single epoch {ck['single_epoch']}, "
              f"{avg[f]:.4f} averaged {ck['averaged_epochs']}   "
              f"OOF {ck['single_oof_auc']:.3f} / {ck['oof_auc']:.3f} (this run's own targets)")
    else:
        new[f], new_pl[f] = float(ck["gold_auc"]), ck.get("gold_per_label") or {}
        print(f"  fold{f}  gold {new[f]:.4f}   OOF {ck['oof_auc']:.3f} (on this run's own targets)")
want = [0, 1, 2, 3, 4] if FOLDS == "all" else [0]
if sorted(new) != want:
    raise SystemExit(f"expected folds {want}, got {sorted(new)}")
base, base_pl = {}, {}
for p in find(suffix=".pt"):
    if f"/{BASE}_{PLANE}/" in p:
        m = _re.search(r"fold(\d)\.pt$", p)
        if m and int(m.group(1)) in want:
            # two attached notebooks holding the same arm folder would let the
            # last one found win silently
            assert int(m.group(1)) not in base, f"two baselines for fold {m.group(1)}: {p}"
            ck = torch.load(p, map_location="cpu", weights_only=False)
            # an averaged arm carries its single epoch too; compare like with like
            base[int(m.group(1))] = float(ck.get("single_gold_auc", ck["gold_auc"]))
            base_pl[int(m.group(1))] = ck.get("single_gold_per_label", ck.get("gold_per_label")) or {}
if sorted(base) != want:
    raise SystemExit(f"baseline {BASE}_{PLANE} checkpoints not attached ({sorted(base)})")
d = [new[f] - base[f] for f in want]
print("\n" + "=" * 70)
print(f"{PLANE}  {TAG}  ({' '.join(EXTRA) or 'no flag change'}; labels {LABELFILE}; vs {BASE}_{PLANE})")
print("  baseline  " + "  ".join(f"{base[f]:.4f}" for f in want) + f"   mean {np.mean([base[f] for f in want]):.4f}")
print("  this      " + "  ".join(f"{new[f]:.4f}" for f in want) + f"   mean {np.mean([new[f] for f in want]):.4f}")
print("  paired    " + "  ".join(f"{x:+.4f}" for x in d) + f"   mean {np.mean(d):+.4f}")
if all(base_pl.get(f) and new_pl.get(f) for f in want):
    print("\n  per finding, mean over folds (this - baseline):")
    for c in LABELS:
        v = [new_pl[f].get(c, np.nan) - base_pl[f].get(c, np.nan) for f in want]
        print(f"    {c:<17}{np.nanmean(v):+.3f}")
print()
if FOLDS == "0":
    print(f"FOLD-0 SCREEN: {'passes' if np.mean(d) >= BAR else 'does not pass'} "
          f"the +{BAR} screen. Confirm at five folds before keeping or dropping it.")
elif np.mean(d) >= BAR:
    print(f"CLEARS the +{BAR} bar.")
elif np.mean(d) > -BAR:
    print(f"Inside +/-{BAR}: can't tell from noise.")
else:
    print(f"WORSE by more than {BAR}.")

# ------------------------------------------ the averaging, judged inside the run
if sorted(avg) == want:
    a = [avg[f] - new[f] for f in want]
    print("\n" + "=" * 70)
    print("weight averaging (--avg-top 3), same run, same training")
    print("  single    " + "  ".join(f"{new[f]:.4f}" for f in want) + f"   mean {np.mean([new[f] for f in want]):.4f}")
    print("  averaged  " + "  ".join(f"{avg[f]:.4f}" for f in want) + f"   mean {np.mean([avg[f] for f in want]):.4f}")
    print("  paired    " + "  ".join(f"{x:+.4f}" for x in a) + f"   mean {np.mean(a):+.4f}")
    print(f"  averaged vs baseline: mean {np.mean([avg[f] - base[f] for f in want]):+.4f}")
    print("  No seed noise separates these two models, but the 58 are few and which epochs\n"
          "  get averaged turns on OOF ties. Rule, fixed 2026-09-29 before any result: keep\n"
          "  averaging only if two five-fold runs each show mean >= +0.003 and at least\n"
          "  7 of their 10 folds are positive.")
