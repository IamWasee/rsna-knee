# ============================================================
# RSNA Knee — how much of the sagittal score is the scanner, not the knee?
#
# Train the SAME model on the SAME studies against the SAME targets, changing
# only how the folds are cut. Grouped keeps scanner and duplicate-report groups
# inside one fold. Random ignores them. random - grouped is the size of the
# shortcut the grouped split is there to block.
#
# Both arms run here, in one session, on one commit, with one seed. The last two
# experiments came back unreadable because each was compared against a baseline
# trained on a different flag; running the control beside the treatment is the
# only defence that has actually worked.
#
# WHAT THIS DOES *NOT* MEASURE -- read before quoting the number:
#
#  * It is not the 0.077 CV-to-leaderboard gap. That gap is between a NINE-ARM
#    BLEND (LB 0.880) and the blend's pooled OOF. This cell trains ONE sagittal
#    arm, and single-arm to blend is worth about +0.020 on its own (cell 36:
#    sag 0.8079 -> four arms 0.8278). Do not subtract 0.880 from anything here.
#  * A shortcut existing does not prove the hidden test shares scanners with
#    train. A large delta is an UPPER BOUND on how much the leaderboard could
#    be inflated, not evidence that it is.
#  * Our OOF is scored against sharpened report-derived labels; the leaderboard
#    is scored against radiologist gold. That label-semantics gap is a third
#    explanation this cell does not touch -- so the run also prints the gold
#    AUC the trainer already computes, which bounds it for free.
#
# TWO BIASES OF OPPOSING SIGN, neither fully controlled:
#  * Pooling five folds into one ranked list penalises the arm whose five models
#    are least interchangeable. Grouped folds have distinct case mixes, so their
#    models are calibrated differently and pooling costs them -- this inflates
#    random - grouped with zero leakage. Mitigated below by rank-normalising
#    within each fold before pooling, and by also reporting the mean of the
#    per-fold macro AUCs, which has no pooling artifact at all. If the two
#    disagree, believe the per-fold mean.
#  * Each fold's reported OOF is its best-of-8-epochs on its own validation set.
#    Grouped folds are within-correlated, so their epoch-to-epoch AUC is noisier
#    and their selection bonus is larger. That inflates grouped and UNDERSTATES
#    the delta.
# One seed, no error bar. Do not read a delta under about 0.02 as real.
#
# ~110 min an arm, ~3.7h for the pair against a 9h limit.
# Attach: competition, cache-sag, stevenleehans, dinov2. GPU, Internet on.
# ============================================================
!pip install -q timm transformers

import sys, os, time, glob, json, subprocess
CODE = "/kaggle/working/rsna-knee"
!rm -rf $CODE && git clone -q https://github.com/IamWasee/rsna-knee.git $CODE
sys.path.insert(0, f"{CODE}/src")
from kaggle_paths import find, describe

print(open(f"{CODE}/SRC_VERSION.txt").read() if os.path.exists(f"{CODE}/SRC_VERSION.txt")
      else subprocess.run(["git", "-C", CODE, "log", "--oneline", "-1"],
                          capture_output=True, text=True).stdout)

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
if man.get("slots") != 1:
    raise SystemExit(f"cache_sag has {man.get('slots')} sequences, expected 1")
if man.get("n_slices") % 3:
    raise SystemExit(f"cache_sag has {man.get('n_slices')} slices, not a multiple of 3")
SIZE, NSL = str(man["size"]), str(man["n_slices"])
print(f"cache_sag: {cache}\n  {SIZE}px, {NSL} slices, band {man.get('band')}, "
      f"{len(glob.glob(cache+'/*.npy'))} studies")

BACKBONE = f"dinov2:{dino[0]}"
# Copied from planes-fixed, which scored 0.8037 grouped. --sharpen-to is passed
# explicitly; it no longer has a default, because leaving it out is what made
# the last two runs unreadable.
COMMON = ["--cache", cache, "--labels", lab[0], "--backbone", BACKBONE,
          "--size", SIZE, "--slots", "1", "--n-slices", NSL, "--folds", "5",
          "--head", "shared", "--pool", "focal", "--batch", "8", "--epochs", "8",
          "--lr", "1e-3", "--lr-backbone", "8e-6", "--unfreeze-last", "6",
          "--weight-decay", "0.02", "--sharpen-to", "source", "--seed", "42"]

def run(extra, label):
    """Run train.py and STOP on failure.

    The ! magic swallows exit status, so a REFUSED budget gate or a crashed arm
    printed its error and the cell carried on to burn the next two hours. The
    rehearsal below is only a gate if a failure actually stops the cell.
    """
    print("\n" + "=" * 70 + f"\n{label}\n" + "=" * 70, flush=True)
    r = subprocess.run([sys.executable, "-u", f"{CODE}/src/train.py"]
                       + COMMON + extra)
    if r.returncode != 0:
        raise SystemExit(f"{label} exited {r.returncode} -- stopping before the "
                         f"next arm burns a session on a broken run")

t0 = time.time()
# One rehearsal. Both arms cost the same, and train.py now measures one fold and
# returns rather than repeating the identical projection five times.
# 130 min/arm, not 260: the gate has to be tight enough that a 2x regression
# trips it here rather than killing the session halfway through arm two.
for arm in ["grouped", "random"]:
    run(["--fold-grouping", arm, "--dry-run", "4", "--max-minutes", "130",
         "--out", "/kaggle/working/rehearse"], f"rehearsal, {arm} split")

for arm in ["grouped", "random"]:
    run(["--fold-grouping", arm, "--out", f"/kaggle/working/leak_{arm}"],
        f"{arm} folds, five of them")
    print(f"\nelapsed {(time.time()-t0)/60:.0f} min", flush=True)

# ---------------------------------------------------------------- scoring
import pandas as pd, numpy as np
from config import LABELS

def auc(t, s):
    r = pd.Series(s).rank().values
    n1 = t.sum(); n0 = len(t) - n1
    return (r[t == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0) if n1 and n0 else np.nan

D = {}
for arm in ["grouped", "random"]:
    p = f"/kaggle/working/leak_{arm}/oof.csv"
    if not os.path.exists(p):
        raise SystemExit(f"{arm} produced no oof.csv")
    D[arm] = pd.read_csv(p).set_index("StudyInstanceUID").sort_index()

print("\n" + "=" * 70)
if not D["grouped"].index.equals(D["random"].index):
    raise SystemExit("the two arms scored different studies -- comparison void")
stamps = {a: (d["__targets"].iloc[0] if "__targets" in d else "unstamped")
          for a, d in D.items()}
print(f"target stamps: {stamps}")
if len(set(stamps.values())) > 1:
    raise SystemExit("the two arms trained on different targets -- comparison void")
print(f"{len(D['grouped'])} studies, both arms, one target set")

def table(norm):
    """norm=True rank-normalises within each fold before pooling.

    Without it the pooled ranking is contaminated by score offsets between the
    five fold models, and those offsets are larger for grouped folds, which
    manufactures a delta out of nothing.
    """
    tot, rows = {"grouped": [], "random": []}, []
    P = {}
    for arm, d in D.items():
        P[arm] = {}
        for c in LABELS:
            v = d[c].astype(float)
            P[arm][c] = (v.groupby(d["fold"]).rank(pct=True).values if norm
                         else v.values)
    for c in LABELS:
        y = pd.to_numeric(D["grouped"][c + "__y"], errors="coerce").values
        k = ~np.isnan(y); t = (y[k] > 0.5).astype(int)
        if len(set(t)) < 2:
            continue
        a = {arm: auc(t, P[arm][c][k]) for arm in D}
        for arm in D:
            tot[arm].append(a[arm])
        rows.append((c, t.mean(), a["grouped"], a["random"]))
    return rows, np.mean(tot["grouped"]), np.mean(tot["random"])

rows_raw, g_raw, r_raw = table(norm=False)
rows_rk, g, r = table(norm=True)

# Per-fold macro. Nothing is ever compared across a fold boundary here, so this
# is the only one of the three free of cross-fold contamination -- the pooled
# number carries score offsets between the five fold models, and rank-normalising
# only trades those for a prevalence distortion, because grouped folds genuinely
# differ in per-fold positive rate and uniform ranks assert that they do not.
#
# It is also the NOISIEST of the three: it throws away every cross-fold pair,
# about four fifths of them, so its standard error is roughly 2.2x the pooled
# one. Hence the printed standard error -- the threshold has to be checked, not
# asserted.
#
# One label set for all ten (arm, fold) cells. Scoring each cell over whatever
# labels happen to be two-class there lets the two arms average over different
# labels, and the labels are nowhere near interchangeable (Baker's 0.88 against
# Lateral Meniscus 0.74). np.nanmean over ten labels and over twelve print
# identically, so that failure would be silent and worth ~0.01 per dropped
# label-fold -- half the effect we are trying to see.
usable, dropped = [], []
for c in LABELS:
    ok = True
    for arm, d in D.items():
        for _, sub in d.groupby("fold"):
            y = pd.to_numeric(sub[c + "__y"], errors="coerce").values
            k = ~np.isnan(y)
            if k.sum() == 0 or len(set((y[k] > 0.5))) < 2:
                ok = False
    (usable if ok else dropped).append(c)
if dropped:
    print(f"\nEXCLUDED from the per-fold macro (single-class in some fold): {dropped}")
print(f"per-fold macro over {len(usable)} of {len(LABELS)} labels")

per = {}
for arm, d in D.items():
    v = []
    for _, sub in d.groupby("fold"):
        a = []
        for c in usable:
            y = pd.to_numeric(sub[c + "__y"], errors="coerce").values
            k = ~np.isnan(y)          # same masking as the pooled path
            a.append(auc((y[k] > 0.5).astype(int), sub[c].values[k]))
        v.append(np.mean(a))
    per[arm] = v
if len(per["grouped"]) != len(per["random"]):
    raise SystemExit(f"fold counts differ ({len(per['grouped'])} vs "
                     f"{len(per['random'])}) -- a fold produced no oof")
pg, pr = np.mean(per["grouped"]), np.mean(per["random"])
se = {a: np.std(v, ddof=1) / np.sqrt(len(v)) for a, v in per.items()}
se_d = np.hypot(se["grouped"], se["random"])

print(f"\n{'label':<20}{'pos':>7}{'grouped':>10}{'random':>10}{'delta':>9}")
print("-" * 56)
for c, pos, a, b in rows_rk:          # the fold-ranked rows, not the raw pooled ones
    print(f"{c:<20}{pos:>6.1%} {a:>9.4f}{b:>10.4f}{b-a:>+9.4f}")
print("-" * 56)
print(f"{'MACRO pooled':<20}{'':>7}{g_raw:>9.4f}{r_raw:>10.4f}{r_raw-g_raw:>+9.4f}")
print(f"{'MACRO fold-ranked':<20}{'':>7}{g:>9.4f}{r:>10.4f}{r-g:>+9.4f}")
print(f"{'MACRO per-fold':<20}{'':>7}{pg:>9.4f}{pr:>10.4f}{pr-pg:>+9.4f}")

# The five folds behind that mean. NOT paired: grouped fold 2 and random fold 2
# hold different studies, so subtracting a row means nothing.
print(f"\nthe five folds behind the per-fold macro (unpaired -- different studies)")
print(f"{'':<20}{'grouped':>10}{'random':>10}")
for i in range(len(per["grouped"])):
    print(f"{'  fold ' + str(i):<20}{per['grouped'][i]:>10.4f}{per['random'][i]:>10.4f}")
print(f"{'  mean':<20}{pg:>10.4f}{pr:>10.4f}")
print(f"{'  std err':<20}{se['grouped']:>10.4f}{se['random']:>10.4f}")

# The teacher's own error, which neither fold scheme touches. train.py already
# saved it into every checkpoint; the cell used to tell the reader to go hunting
# for it in scrollback that may not survive.
print("\ngold AUC (the 58 radiologist-labelled studies), read from the checkpoints")
import torch
for arm in ["grouped", "random"]:
    gs = []
    for f_ in sorted(glob.glob(f"/kaggle/working/leak_{arm}/fold*.pt")):
        ck = torch.load(f_, map_location="cpu", weights_only=False)
        if ck.get("gold_auc") is not None and ck["gold_auc"] == ck["gold_auc"]:
            gs.append(float(ck["gold_auc"]))
    print(f"  {arm:<10}{np.mean(gs):.4f} over {len(gs)} folds" if gs
          else f"  {arm:<10}no gold_auc in the checkpoints")

print(f"\ntotal {(time.time()-t0)/3600:.1f} h")
print("\nHow to read this")
print("-" * 70)
print(f"  Believe the per-fold macro: {pr-pg:+.4f}, std err of the difference")
print(f"  about {se_d:.4f}. It is the only one of the three that never ranks a")
print( "  study from one fold against a study from another, and cross-fold")
print( "  contamination is the whole hazard here. It is also the noisiest, which")
print( "  is why the standard error is printed rather than a threshold asserted.")
print(f"  Pooled ({r_raw-g_raw:+.4f}) and fold-ranked ({r-g:+.4f}) are shown for")
print( "  triangulation; a per-fold delta inside two standard errors of zero is")
print( "  a null whatever they say.")
print( "")
print( "  A CLEAR POSITIVE is the trustworthy outcome. Epoch selection happens on")
print( "  each fold's own validation set, and grouped folds are within-correlated,")
print( "  so their selection bonus is the larger one -- that bias inflates grouped")
print( "  and shrinks the delta. A positive result survives it; a null is exactly")
print( "  what that bias would manufacture, so do not conclude 'no leakage' from a")
print( "  null with any confidence.")
print( "")
print( "  If positive and clear: a scanner shortcut exists. Grouped is our true")
print( "  skill. This is an UPPER BOUND on how much the leaderboard could be")
print( "  inflated -- it does not prove the hidden test shares our sites.")
print( "  If null: no shortcut this design can see. Compare the gold AUC above")
print( "  against the OOF macro -- that difference is the teacher's error, the")
print( "  third explanation for the leaderboard gap and the one still standing.")
