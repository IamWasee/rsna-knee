# ============================================================
# RSNA Knee — SUBMISSION, weighted per ARM. Internet OFF.
#
# Seven arms across four data layouts:
#   plane_sag@planes-5fold    1 seq x 24 slices, sagittal   DINOv2, 8 epochs
#   plane_sag@planes-16ep     1 seq x 24 slices, sagittal   DINOv2, 16 epochs
#   plane_cnx_sag@...         1 seq x 24 slices, sagittal   ConvNeXt
#   plane_cor@planes-5fold    1 seq x 24 slices, coronal
#   plane_ax@planes-5fold     1 seq x 24 slices, axial
#   w_slot@full-v3            4 seq x 9 slices             slot head
#   w_shared@full-v3          4 seq x 9 slices             shared head
#
# TWO THINGS THIS CELL MUST GET RIGHT, both of which the previous version did not:
#
# 1. Average per ARM, not per layout. Three arms read the sagittal cache. Grouping
#    by layout gives them a combined 1/4 of the vote where the measurement that
#    produced 0.8032 gave them 3/7. Same checkpoints, different submission.
#
# 2. Key arms by directory AND notebook. planes-5fold and planes-16ep both write
#    "plane_sag"; keying on the directory alone keeps whichever is walked first
#    and silently drops the other.
#
# Each distinct test cache is built once, however many arms read it.
#
# Pooled OOF: 0.8032 for all seven, against 0.7998 for the four-arm set that
# scored 0.877. Internal gains have run ~4x smaller than leaderboard ones.
#
# Attach: competition, abdullahwasee/rsna-knee-src, planes-5fold, planes-16ep,
#         convnext-sag-5f, full-v3, metaresearch/dinov2. GPU. INTERNET OFF.
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

import torch
arms, layouts = {}, {}
for p in sorted(find(suffix=".pt")):
    d = os.path.dirname(p)
    arm = f"{os.path.basename(d)}@{os.path.basename(os.path.dirname(d))}"
    ck = torch.load(p, map_location="cpu", weights_only=False)
    man = ck.get("cache_manifest", {})
    key = hashlib.md5(json.dumps(
        {k: man.get(k) for k in ("slots", "size", "crop_mm", "n_anchors",
                                 "n_slices", "laterality", "only_slot")},
        sort_keys=True).encode()).hexdigest()[:8]
    a = arms.setdefault(arm, {"layout": key, "dir": f"/kaggle/working/arms/{arm}"})
    if a["layout"] != key:
        raise SystemExit(f"{arm} mixes checkpoints from two different caches")
    layouts.setdefault(key, man)
    os.makedirs(a["dir"], exist_ok=True)
    shutil.copy(p, f"{a['dir']}/{os.path.basename(p)}")

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
    slot = f"--only-slot {m['only_slot']}" if m.get("only_slot") is not None else ""
    print("\n" + "=" * 66 + f"\nlayout {key}: preprocess\n" + "=" * 66, flush=True)
    !python $SRC/preprocess.py --out $cache --split test --workers 4 \
        --slots {m.get('slots',4)} --size {m.get('size',288)} \
        --crop-mm {m.get('crop_mm',140.0)} --anchors {m.get('n_anchors',3)} {slot} {lat}
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

# Rank-average across ARMS, equally -- the weighting the 0.8032 measurement used.
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
print("\nPooled OOF for all seven arms: 0.8032; the four-arm set was 0.7998 and")
print("scored 0.877. Internal gains have run ~4x smaller than leaderboard ones,")
print("so +0.0034 here has been worth nearer +0.013 there. Record the gap.")
