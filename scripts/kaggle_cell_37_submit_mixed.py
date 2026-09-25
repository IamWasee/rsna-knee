# ============================================================
# RSNA Knee — SUBMISSION, weighted per ARM. Internet OFF.
#
# Three arms: one per plane, each trained 16 epochs with --sharpen-to source.
#
#   plane_s16_sag@sag-16ep-source   1 seq x 24 slices, sagittal   DINOv2
#   plane_s16_cor@cor-16ep-source   1 seq x 24 slices, coronal    DINOv2
#   plane_s16_ax@ax-16ep-source     1 seq x 24 slices, axial      DINOv2
#
# On the 58 radiologist-labelled studies -- the only labels we have from the
# same source as the leaderboard's -- each beats the arm it replaces fold by
# fold, 5 of 5 folds, paired on the same studies:
#
#            this arm   replaces (8 ep, source)   paired mean
#   sag       0.855          0.846                  +0.0095
#   cor       0.842          0.818                  +0.0238
#   ax        0.852          0.812                  +0.0404
#
# They are the three best arms we have on gold, and on both report-label
# yardsticks the three alone beat every larger set that includes them:
#
#                                gold-yard   source-yard
#   the seven that scored 0.882    0.8036      0.8237
#   these three                    0.8132      0.8407
#   these three + slot/shared      0.8107      0.8349
#
# So the older arms are dropped rather than kept for diversity: each is now
# worse on gold and dilutes the blend on both report yardsticks.
#
# Caveat: every one of the 58 gold studies has at least one finding, so gold
# measures telling abnormal knees apart. The hidden test has normal ones too;
# the gain may shrink there. This submission is the test of that.
#
# Attach: competition, abdullahwasee/rsna-knee-src, sag-16ep-source,
#         cor-16ep-source, ax-16ep-source, metaresearch/dinov2.
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
# instead of quietly filing a subset. The header above says which three and why.
KEEP = {
    "plane_s16_sag@sag-16ep-source", "plane_s16_cor@cor-16ep-source",
    "plane_s16_ax@ax-16ep-source",
}

import torch
arms, layouts = {}, {}
skipped = set()
for p in sorted(find(suffix=".pt")):
    d = os.path.dirname(p)
    arm = f"{os.path.basename(d)}@{os.path.basename(os.path.dirname(d))}"
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
    a = arms.setdefault(arm, {"layout": key, "dir": f"/kaggle/working/arms/{arm}"})
    if a["layout"] != key:
        raise SystemExit(f"{arm} mixes checkpoints from two different caches")
    layouts.setdefault(key, man)
    os.makedirs(a["dir"], exist_ok=True)
    # symlink, not copy: infer.py globs *.pt and follows links, and copying
    # thirty-five checkpoints spends several GB of a 20 GB working quota to
    # duplicate files that are already mounted.
    link = f"{a['dir']}/{os.path.basename(p)}"
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
    out = f"/kaggle/working/parts/{arm.replace('@','_')}.csv"
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
print("\nThree arms, 16 epochs + source sharpening, one per plane. On the 58")
print("radiologist-labelled studies: sag 0.855, cor 0.842, ax 0.852, each beating")
print("the arm it replaced on 5 of 5 folds. The previous submission scored 0.882.")
