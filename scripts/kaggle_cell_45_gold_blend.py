# ============================================================
# RSNA Knee — every arm, and every blend, scored on gold-58 with the
# submission's own prediction code.
#
# Why now: ax-pseudo50 cleared the bar alone (+0.0187 paired, 5/5 folds), but it
# learned from the six submitted arms' own predictions, so part of that gain may
# already be inside the blend. The OOF blend cannot settle it: pseudo50's OOF
# inherits the teachers' knowledge of each held-out study's report label (each
# teacher trained on the other folds). The 58 trained nothing and taught
# nothing, so they can.
#
# Per arm: its five fold models predict the 58 as infer.py predicts the test
# set -- same loader, same predict(), rank-averaged over folds, no TTA. The
# pixels come from the training caches, whereas the submission preprocesses the
# test DICOMs afresh; fine for comparing blends with each other.
# Arms are then rank-averaged equally, as cell 37 does. Raw per-fold
# predictions are saved so any other blend is recomputed offline, free.
# Averaged arms (--avg-top) are scored twice: as saved, and as their single
# best epoch (fold<k>_single.ckpt), named <arm>#single.
#
# Parity check, enforced: each fold model's AUC on the 58 here must be within
# 0.005 of the gold_auc its checkpoint recorded in training (fp16 on GPU there,
# fp32 here). A re-versioned cache or a changed library fails loudly.
#
# Attach: competition, dinov2, cache-sag, cache-cor, cache-ax, and the arm
# notebooks named in ARMS. CPU is enough for 58 studies. Internet on.
# ============================================================
!pip install -q timm transformers

import sys, os, json, time
from pathlib import Path
CODE = "/kaggle/working/rsna-knee"
!rm -rf $CODE && git clone -q https://github.com/IamWasee/rsna-knee.git $CODE
!git -C $CODE log --oneline -1
sys.path.insert(0, f"{CODE}/src")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader
from kaggle_paths import find, describe
from config import ID_COL, LABELS
from dataset import KneeStudies
from infer import load_models, check_parity, predict, rank_average
from paths import data_root
from train import macro_auc

ARMS = {  # arm (<output dir>@<notebook>) -> plane
    "plane_s16_sag@sag-16ep-source": "sag",
    "plane_s16_cor@cor-16ep-source": "cor",
    "plane_s16_ax@ax-16ep-source": "ax",
    "plane_s16s43_sag@sag-16ep-seed43": "sag",
    "plane_s16s43_cor@cor-16ep-seed43": "cor",
    "plane_s16s43_ax@ax-16ep-seed43": "ax",
    "plane_pseudo50_ax@ax-pseudo50": "ax",
    "plane_pseudo50_cor@cor-pseudo50": "cor",
    "plane_slotpos_ax@ax-slotpos": "ax",
    "plane_slotpos_sag@sag-slotpos": "sag",
}
SIX = [a for a in ARMS if "_s16" in a]
BLENDS = {
    "3 arms, seed 42 (LB 0.909)": [a for a in SIX if "s43" not in a],
    "6 arms (LB 0.910)": SIX,
    "6 + ax-pseudo50": SIX + ["plane_pseudo50_ax@ax-pseudo50"],
    "6 + ax-pseudo50#single": SIX + ["plane_pseudo50_ax@ax-pseudo50#single"],
    "6, pseudo50 replaces ax seed 42": [a for a in SIX if a != "plane_s16_ax@ax-16ep-source"]
                                       + ["plane_pseudo50_ax@ax-pseudo50"],
    "6 + both slotpos": SIX + ["plane_slotpos_ax@ax-slotpos",
                               "plane_slotpos_sag@sag-slotpos"],
    "6 + ax & cor pseudo50#single": SIX + ["plane_pseudo50_ax@ax-pseudo50#single",
                                           "plane_pseudo50_cor@cor-pseudo50#single"],
}
# --sub ONLY=<arm>,<arm> scores just those arms (each with its #single); empty
# scores all. Blends a run cannot complete are skipped here and recomputed
# offline from the saved predictions of every run.
ONLY = [a for a in "__ONLY__".split(",") if a]
assert all(a in ARMS for a in ONLY), f"unknown arm in ONLY: {ONLY}"
if ONLY:
    ARMS = {a: ARMS[a] for a in ONLY}

train = pd.read_csv(data_root() / "train.csv")
gold = train[train[LABELS].notna().all(axis=1)].reset_index(drop=True)
assert len(gold) == 58, f"expected 58 gold studies, got {len(gold)}"
Y = gold[LABELS].astype(int).values

caches = {}
for m in find(filename="cache_manifest.json"):
    d = os.path.dirname(m)
    for p in ("sag", "cor", "ax"):
        if os.path.basename(d) == f"cache_{p}":
            caches[p] = d
missing = {ARMS[a] for a in ARMS} - set(caches)
if missing:
    describe(); raise SystemExit(f"attach cache-{sorted(missing)}")
for p, d in caches.items():
    absent = [s for s in gold[ID_COL] if not os.path.exists(f"{d}/{s}.npy")]
    # KneeStudies serves zeros for a missing study rather than stop a test run;
    # here that would score a blank knee as if it were a prediction.
    assert not absent, f"cache_{p} lacks {len(absent)} gold studies"

# Link each arm's folds into its own folder; load_models takes every *.pt there.
dirs, recorded = {}, {}
for p in find(suffix=".pt") + find(suffix="_single.ckpt"):
    d = os.path.dirname(p)
    arm = f"{os.path.basename(d)}@{os.path.basename(os.path.dirname(d))}"
    if arm not in ARMS:
        continue
    name = os.path.basename(p)
    if name.endswith("_single.ckpt"):
        arm, name = arm + "#single", name.replace("_single.ckpt", ".pt")
    dirs.setdefault(arm, f"/kaggle/working/arms/{arm}")
    os.makedirs(dirs[arm], exist_ok=True)
    if not os.path.exists(f"{dirs[arm]}/{name}"):   # one file can be mounted twice
        os.symlink(p, f"{dirs[arm]}/{name}")
absent = set(ARMS) - set(dirs)
if absent:
    describe(); raise SystemExit(f"arms not attached: {sorted(absent)}")

OUT = Path("/kaggle/working/gold_preds")
OUT.mkdir(exist_ok=True)
gold[[ID_COL] + LABELS].to_csv(OUT / "gold_labels.csv", index=False)
device = "cuda" if torch.cuda.is_available() else "cpu"
preds, broken, t0 = {}, [], time.time()
for arm in sorted(dirs):
    plane = ARMS[arm.split("#")[0]]
    print(f"\n{arm}  ({plane}, {device})", flush=True)
    models, cfg, man = load_models(Path(dirs[arm]), device)
    assert len(models) == 5, f"{arm}: {len(models)} fold models, expected 5"
    check_parity(man, Path(caches[plane]))
    ds = KneeStudies(gold[[ID_COL]], cache=caches[plane], train=False,
                     slots=man["slots"], n_slices=man["n_slices"], size=man["size"])
    per_model, ids = predict(models, DataLoader(ds, batch_size=8, num_workers=2), device)
    assert list(ids) == list(gold[ID_COL]), f"{arm}: study order changed"
    # "#" starts a URL fragment: kaggle kernels output fetched those files empty
    np.save(OUT / f"{arm.replace('#', '__')}.npy", per_model)   # (folds, 58, 12)
    # load_models took the folder's *.pt in sorted order; re-read their records
    rec = [float(torch.load(f, map_location="cpu", weights_only=False)["gold_auc"])
           for f in sorted(Path(dirs[arm]).glob("*.pt"))]
    for k, pm in enumerate(per_model):
        here = macro_auc(Y, pm)[0]
        print(f"    fold model {k}: gold {here:.4f} here, {rec[k]:.4f} in training")
        if abs(here - rec[k]) >= 0.005:
            broken.append(f"{arm} fold {k}: {here:.4f} here vs {rec[k]:.4f} in training")
            print("    ^^^ PARITY BROKEN")
    preds[arm] = rank_average(per_model)
    print(f"  {arm}: five folds rank-averaged, gold {macro_auc(Y, preds[arm])[0]:.4f}"
          f"   ({(time.time()-t0)/60:.1f} min)", flush=True)

def blend(names):
    r = np.zeros_like(preds[names[0]])
    for a in names:
        r += np.column_stack([pd.Series(preds[a][:, j]).rank(pct=True).to_numpy()
                              for j in range(len(LABELS))])
    return r / len(names)

print("\n" + "=" * 70 + "\nper arm, folds rank-averaged (as submitted)")
for arm in sorted(preds):
    print(f"  {arm:<42} {macro_auc(Y, preds[arm])[0]:.4f}")
print("\nblends, arms rank-averaged equally (as cell 37)")
for name, names in BLENDS.items():
    if all(a in preds for a in names):
        auc, per = macro_auc(Y, blend(names))
        print(f"  {name:<34} {auc:.4f}   ({len(names)} arms)")
print(f"\n58 studies: read blend gaps with a paired bootstrap offline, not by eye.")
# Raised last, so one drifting arm does not throw away hours of scoring the rest.
if broken:
    raise SystemExit("parity broken -- these numbers are NOT the trained models':\n  "
                     + "\n  ".join(broken))
