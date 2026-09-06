# ============================================================
# RSNA Knee — SUBMISSION for the mixed blend. Internet OFF.
#
# The selected set spans THREE data layouts, which the previous submission cell
# could not do -- it preprocessed the test set once, from the first checkpoint's
# manifest, and served every model from it:
#
#   plane_sag   1 sequence x 24 slices   (cache_sag)
#   plane_cor   1 sequence x 24 slices   (cache_cor)
#   w_slot      4 sequences x 9 slices   (cache_v3)
#   w_shared    4 sequences x 9 slices   (cache_v3)
#
# So: group the checkpoints by the manifest they were TRAINED with, build one
# test cache per distinct manifest, run inference per group, and rank-average
# the groups at the end. infer.py's parity check then guards each group
# separately -- serving a sagittal-trained model a coronal cache has the right
# tensor shape and the wrong anatomy, which is the silent class of failure that
# once scored 0.675.
#
# Five-fold pooled OOF for this set: 0.7998, against 0.7970 for the two
# four-sequence heads alone, which scored 0.865. So expect ~0.868: this run is
# for the CALIBRATION POINT more than the score. Two submissions have put the
# leaderboard ~0.068 above our pooled number; a third either confirms that or
# tells me the projections have been wrong.
#
# Attach: competition, abdullahwasee/rsna-knee-src, planes-5fold, full-v3,
#         metaresearch/dinov2. GPU. INTERNET OFF.
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

WANT = {"plane_sag", "plane_cor", "w_slot", "w_shared"}
groups = {}          # manifest fingerprint -> {"man":…, "dir":…, "n":…}
FLAT = "/kaggle/working/groups"
import torch
for p in sorted(find(suffix=".pt")):
    arm = os.path.basename(os.path.dirname(p))
    if arm not in WANT:
        continue
    ck = torch.load(p, map_location="cpu", weights_only=False)
    man = ck.get("cache_manifest", {})
    key = hashlib.md5(json.dumps(
        {k: man.get(k) for k in ("slots", "size", "crop_mm", "n_anchors",
                                 "n_slices", "laterality", "only_slot")},
        sort_keys=True).encode()).hexdigest()[:8]
    g = groups.setdefault(key, {"man": man, "dir": f"{FLAT}/{key}", "arms": set()})
    os.makedirs(g["dir"], exist_ok=True)
    # fold0.pt exists in every arm -- keep the arm in the name or they overwrite
    shutil.copy(p, f"{g['dir']}/{arm}_{os.path.basename(p)}")
    g["arms"].add(arm)

if not groups:
    describe(); raise SystemExit(f"no checkpoints from {sorted(WANT)} found")
missing = WANT - set().union(*(g["arms"] for g in groups.values()))
if missing:
    print(f"WARNING: {sorted(missing)} not attached. The 0.7998 set is incomplete; "
          f"this submits whatever is present.")

print(f"\n{len(groups)} distinct layout(s):")
for key, g in groups.items():
    m = g["man"]
    print(f"  {key}: {sorted(g['arms'])} -- {m.get('slots')} seq x "
          f"{m.get('n_slices')} slices @ {m.get('size')}px, "
          f"only_slot={m.get('only_slot')}, {len(os.listdir(g['dir']))} checkpoints")

t0 = time.time()
subs = []
for key, g in groups.items():
    m = g["man"]
    cache = f"/kaggle/working/test_{key}"
    lat = "" if m.get("laterality", False) else "--no-laterality"
    slot = f"--only-slot {m['only_slot']}" if m.get("only_slot") is not None else ""
    print("\n" + "=" * 66 + f"\nlayout {key}: preprocess + infer\n" + "=" * 66, flush=True)
    !python $SRC/preprocess.py --out $cache --split test --workers 4 \
        --slots {m.get('slots',4)} --size {m.get('size',288)} \
        --crop-mm {m.get('crop_mm',140.0)} --anchors {m.get('n_anchors',3)} {slot} {lat}
    # Intermediates go in a subdirectory. Left at the top level, Kaggle's submit
    # dialog offered sub_<layout>.csv instead of submission.csv -- which is one
    # layout's models, not the blend, and would have scored the wrong thing.
    os.makedirs("/kaggle/working/parts", exist_ok=True)
    out = f"/kaggle/working/parts/sub_{key}.csv"
    !python $SRC/infer.py --cache $cache --weights {g['dir']} --out $out
    if not os.path.exists(out):
        raise SystemExit(f"layout {key} produced no submission -- see above")
    subs.append(out)
    print(f"elapsed {(time.time()-t0)/60:.1f} min", flush=True)

# Rank-average ACROSS layouts. Within a layout infer.py already rank-averaged its
# folds; here the groups are combined the same way, per label, because a
# sagittal model and a four-sequence model are not calibrated to each other.
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
print(f"\nsubmission: {base.shape[0]} rows x {base.shape[1]} cols "
      f"from {len(frames)} layout(s)")
assert base[LABELS].notna().all().all(), "NaNs in the submission"
assert base[ID_COL].is_unique, "duplicate study ids"
assert base.shape[1] == 13, f"expected 13 columns, got {base.shape[1]}"
print(base.head(3).to_string())
print(f"\ntotal {(time.time()-t0)/60:.1f} min")
print("\nPooled OOF for this set: 0.7998. The two four-sequence heads alone were")
print("0.7970 and scored 0.865. Write down the DIFFERENCE, not the score.")
