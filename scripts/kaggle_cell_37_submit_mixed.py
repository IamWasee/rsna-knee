# ============================================================
# RSNA Knee — SUBMISSION, weighted per ARM. Internet OFF.
#
# Nine arms, three per plane: the six that scored 0.910 plus a pseudo-label
# arm per plane (src/pseudo_labels.py: 50/50 report labels and the six arms'
# out-of-fold predictions), each as its single best epoch per fold.
#
#   plane_s16_{sag,cor,ax}@{plane}-16ep-source       seed 42   (the 0.909 set)
#   plane_s16s43_{sag,cor,ax}@{plane}-16ep-seed43    seed 43   (-> 0.910)
#   plane_pseudo50_{sag,cor,ax}@{plane}-pseudo50#single  pseudo-labels
#                                                    (ax alone added -> 0.912)
#
# Pseudo-label arms on gold, paired against their plane's seed-42 arm:
#   ax +0.0187, cor +0.0169, sag +0.0144, every fold up in all three.
# Blends on gold (cell 45, bootstrap over the 58), the 9 chosen in advance:
#   7 arms (LB 0.912)                 0.8980
#   9 arms, plane-balanced            0.8969   -0.0011, P(gain) 0.34
# The pre-registered veto was P < 0.2, so it goes to the leaderboard, which
# decides: keep if >= 0.912. Expect it near flat -- the cor and sag arms are
# strong alone but hold much of what the blend already knows.
#
# The seed runs also measured that noise. The three planes' five-fold mean gold
# differences between seeds were +0.0095, -0.0050, +0.0011: an SD of ~0.006, so
# a real gain now needs about +0.013, not the +0.005 used before. (Cell 42's own
# printout says ~0.003-0.004; it assumes the five folds move independently, and
# they don't -- a seed shifts all five together, as sagittal's five-up +0.0095
# showed.)
#
# Attach: competition, abdullahwasee/rsna-knee-src, {sag,cor,ax}-16ep-source,
#         {sag,cor,ax}-16ep-seed43, {sag,cor,ax}-pseudo50, metaresearch/dinov2.
#         GPU. INTERNET OFF.
# ============================================================
import sys, os, time, shutil, glob, json, hashlib
import numpy as np, pandas as pd

stamped, unstamped = [], []
for root, dirs, files in os.walk("/kaggle/input", followlinks=True):
    if "rsna-knee-abnormality-detection" in root:
        dirs[:] = []
        continue
    if "infer.py" in files and "preprocess.py" in files:
        (stamped if "SRC_VERSION.txt" in files else unstamped).append(root)
if not stamped:
    raise SystemExit(
        "no stamped src/ found. Attach abdullahwasee/rsna-knee-src.\n"
        + ("Unstamped copies found and deliberately NOT used:\n"
           + "\n".join(f"    {f}" for f in unstamped) + "\n"
           "Those are clones frozen inside older notebook outputs.\n" if unstamped else "")
        + "Publish with: python scripts/sync_src.py")
SRC = stamped[0]
sys.path.insert(0, SRC)
print(f"src: {SRC}\nversion: {open(os.path.join(SRC,'SRC_VERSION.txt')).read().strip()}")
from kaggle_paths import find, describe

# The arms, by name. Selecting on names rather than on whichever notebooks
# happen to be mounted is what makes the submission the measured set: attach an
# extra notebook and its arms are skipped, forget a needed one and the run stops
# instead of quietly filing a subset. The header above says which nine and why.
KEEP = {
    "plane_s16_sag@sag-16ep-source", "plane_s16_cor@cor-16ep-source",
    "plane_s16_ax@ax-16ep-source",
    "plane_s16s43_sag@sag-16ep-seed43", "plane_s16s43_cor@cor-16ep-seed43",
    "plane_s16s43_ax@ax-16ep-seed43",
    "plane_pseudo50_ax@ax-pseudo50#single",
    "plane_pseudo50_cor@cor-pseudo50#single",
    "plane_pseudo50_sag@sag-pseudo50#single",
}

import torch
arms, layouts = {}, {}
skipped = set()
# An arm trained with --avg-top saves the averaged model as fold<k>.pt and its
# best single epoch as fold<k>_single.ckpt. "<arm>#single" in KEEP selects the
# latter; it is linked in as fold<k>.pt, the name infer.py globs for.
for p in sorted(find(suffix=".pt") + find(suffix="_single.ckpt")):
    d = os.path.dirname(p)
    arm = f"{os.path.basename(d)}@{os.path.basename(os.path.dirname(d))}"
    if p.endswith("_single.ckpt"):
        arm += "#single"
    if KEEP and arm not in KEEP:
        skipped.add(arm)
        continue
    ck = torch.load(p, map_location="cpu", weights_only=False)
    man = ck.get("cache_manifest", {})
    if not man:
        # infer.py prints "parity NOT verified" and proceeds. That is the one
        # path here that yields a silently WRONG submission rather than a loud
        # failure: the layout would be rebuilt from defaults and the arm would
        # still take its full share of the vote.
        raise SystemExit(f"{arm} has no cache manifest in {os.path.basename(p)}; "
                         f"parity cannot be verified and the arm would vote anyway")
    key = hashlib.md5(json.dumps(
        # Every field infer.py's parity check compares, or two arms that differ
        # only in band collapse onto one layout, one cache gets built, and
        # check_parity refuses the odd one out -- AFTER every layout has been
        # preprocessed. Loud, but an hour late.
        {k: str(man.get(k)) for k in ("slots", "size", "crop_mm", "n_anchors",
                                      "group", "n_slices", "band", "laterality",
                                      "slot_scheme", "only_slot")},
        sort_keys=True).encode()).hexdigest()[:8]
    # "#" stays out of paths: the shell and URLs both give it a meaning
    safe = arm.replace("#", "__")
    a = arms.setdefault(arm, {"layout": key, "dir": f"/kaggle/working/arms/{safe}"})
    if a["layout"] != key:
        raise SystemExit(f"{arm} mixes checkpoints from two different caches")
    layouts.setdefault(key, man)
    os.makedirs(a["dir"], exist_ok=True)
    # symlink, not copy: infer.py globs *.pt and follows links, and copying
    # every fold of every arm spends several GB of a 20 GB working quota to
    # duplicate files that are already mounted.
    link = f"{a['dir']}/{os.path.basename(p).replace('_single.ckpt', '.pt')}"
    if not os.path.exists(link):
        os.symlink(p, link)

if skipped:
    print(f"skipped {len(skipped)} arm(s) not in KEEP: {sorted(skipped)}")
missing = KEEP - set(arms)
if missing:
    describe()
    raise SystemExit(f"KEEP names {len(KEEP)} arms and {len(missing)} are not "
                     f"attached: {sorted(missing)}. Submitting a subset of a "
                     f"measured set is not the measured set.")
if not arms:
    describe(); raise SystemExit("no checkpoints found -- attach the training notebooks")
print(f"\n{len(arms)} arm(s) over {len(layouts)} layout(s):")
for arm, a in sorted(arms.items()):
    m = layouts[a["layout"]]
    print(f"  {arm:<32} layout {a['layout']}  "
          f"{m.get('slots')}x{m.get('n_slices')} @ {m.get('size')}px "
          f"slot={m.get('only_slot')}  {len(os.listdir(a['dir']))} folds")
# Every arm was measured as five fold models. A doubled or missing fold would
# change the arm's vote without failing anywhere else.
bad = {arm: len(os.listdir(a["dir"])) for arm, a in arms.items() if len(os.listdir(a["dir"])) != 5}
if bad:
    raise SystemExit(f"arms without exactly five fold checkpoints: {bad}")

t0 = time.time()
for key, m in layouts.items():
    cache = f"/kaggle/working/test_{key}"
    lat = "" if m.get("laterality", False) else "--no-laterality"
    bnd = f"--band {m['band'][0]} {m['band'][1]}" if m.get("band") else ""
    slot = f"--only-slot {m['only_slot']}" if m.get("only_slot") is not None else ""
    print("\n" + "=" * 66 + f"\nlayout {key}: preprocess\n" + "=" * 66, flush=True)
    !python $SRC/preprocess.py --out $cache --split test --workers 4 \
        --slots {m.get('slots',4)} --size {m.get('size',288)} \
        --crop-mm {m.get('crop_mm',140.0)} --anchors {m.get('n_anchors',3)} \
        {bnd} {slot} {lat}
    print(f"elapsed {(time.time()-t0)/60:.1f} min", flush=True)

# Intermediates live in a subdirectory: left at the top level, Kaggle's submit
# dialog offered one of them instead of submission.csv, which is a single arm.
os.makedirs("/kaggle/working/parts", exist_ok=True)
subs = []
for arm, a in sorted(arms.items()):
    cache = f"/kaggle/working/test_{a['layout']}"
    wdir = a["dir"]
    out = f"/kaggle/working/parts/{arm.replace('@','_').replace('#','__')}.csv"
    print("\n" + "-" * 66 + f"\n{arm}: infer\n" + "-" * 66, flush=True)
    !python $SRC/infer.py --cache $cache --weights $wdir --out $out
    if not os.path.exists(out):
        raise SystemExit(f"{arm} produced no predictions -- see the error above")
    subs.append(out)
    print(f"elapsed {(time.time()-t0)/60:.1f} min", flush=True)

# Rank-average across ARMS, equally -- the weighting every blend number in the header used.
# infer.py already rank-averaged the folds inside each arm.
from config import ID_COL, LABELS
frames = [pd.read_csv(s) for s in subs]
base = frames[0][[ID_COL]].copy()
for c in LABELS:
    r = np.zeros(len(base))
    for f in frames:
        f = f.set_index(ID_COL).reindex(base[ID_COL]).reset_index()
        r += pd.Series(f[c].values).rank(pct=True).to_numpy()
    base[c] = r / len(frames)

base.to_csv("/kaggle/working/submission.csv", index=False)
top = [f for f in os.listdir("/kaggle/working") if f.endswith(".csv")]
assert top == ["submission.csv"], f"expected only submission.csv at top level, got {top}"
assert base[LABELS].notna().all().all(), "NaNs in the submission"
assert base[ID_COL].is_unique, "duplicate study ids"
assert base.shape[1] == 13, f"expected 13 columns, got {base.shape[1]}"
print(f"\nsubmission: {base.shape[0]} rows x {base.shape[1]} cols "
      f"from {len(frames)} arm(s)")
print(base.head(3).to_string())
print(f"\ntotal {(time.time()-t0)/60:.1f} min")
print("\nNine arms: the six that scored 0.910 plus a pseudo-label arm per plane")
print("(single epoch). Gold-58 blend 0.8969 vs 0.8980 for the 0.912 seven.")
