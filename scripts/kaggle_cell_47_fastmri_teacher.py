# ============================================================
# RSNA Knee — our arms read fastMRI: soft labels for self-training. GPU.
#
# For each plane, that plane's teacher arm (its single best epoch per fold)
# predicts every fastMRI exam that has the plane; the five folds' probabilities
# are averaged, then the planes an exam has. Probabilities, not ranks: these
# become training targets (train.py --extra-labels, which uses them raw under
# --sharpen-to none -- the recipe the teachers themselves were trained with).
#
# Writes fastmri_teacher.csv: id, the 12 labels, n_planes. No gold label, no
# competition study and no fastMRI pixel leaves the notebook; only these
# per-exam probabilities.
#
# Attach: the fastMRI cache notebooks (cell 46), the teacher arm notebooks named
# in TEACHERS, dinov2. GPU, internet on (git clone).
# ============================================================
!pip install -q timm transformers

import sys, os, glob, json, time
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
from infer import load_models, check_parity, predict

TEACHERS = {  # plane -> arm (<output dir>@<notebook>), single best epochs
    "sag": "plane_combo_sag@sag-combo",
    "cor": "plane_combo_cor@cor-combo",
    "ax": "plane_combo_ax@ax-combo",
}

# fastMRI caches: every attached cache_<plane> whose manifest says split fastmri
fm = {p: [] for p in TEACHERS}
for m in find(filename="cache_manifest.json"):
    d = os.path.dirname(m)
    man = json.load(open(m))
    for p in TEACHERS:
        if os.path.basename(d) == f"cache_{p}" and man.get("split") == "fastmri":
            fm[p].append(d)
print({p: len(v) for p, v in fm.items()}, "fastMRI cache dirs per plane")
if not all(fm.values()):
    describe(); raise SystemExit("attach the fastMRI cache notebooks (cell 46)")

# teacher checkpoints: the _single.ckpt files, linked in as fold<k>.pt
dirs = {}
for p in find(suffix="_single.ckpt"):
    d = os.path.dirname(p)
    arm = f"{os.path.basename(d)}@{os.path.basename(os.path.dirname(d))}"
    for plane, want in TEACHERS.items():
        if arm == want:
            dirs.setdefault(plane, f"/kaggle/working/teach/{plane}")
            os.makedirs(dirs[plane], exist_ok=True)
            dst = f"{dirs[plane]}/{os.path.basename(p).replace('_single.ckpt', '.pt')}"
            if not os.path.exists(dst):
                os.symlink(p, dst)
if sorted(dirs) != sorted(TEACHERS):
    describe(); raise SystemExit(f"teacher arms missing: {sorted(set(TEACHERS) - set(dirs))}")

device = "cuda" if torch.cuda.is_available() else "cpu"
per_plane, t0 = {}, time.time()
for plane in TEACHERS:
    models, cfg, man = load_models(Path(dirs[plane]), device)
    assert len(models) == 5, f"{plane}: {len(models)} folds"
    paths = {}
    for d in fm[plane]:
        check_parity(man, Path(d))           # same preprocessing as the teacher saw
        for e in os.scandir(d):
            if e.name.endswith(".npy"):
                paths.setdefault(e.name[:-4], Path(e.path))
    ids = sorted(paths)
    ds = KneeStudies(pd.DataFrame({ID_COL: ids}), cache=fm[plane][0], train=False,
                     slots=man["slots"], n_slices=man["n_slices"], size=man["size"],
                     paths=paths)
    pm, got = predict(models, DataLoader(ds, batch_size=8, num_workers=2), device)
    assert list(got) == ids
    per_plane[plane] = pd.DataFrame(pm.mean(0), columns=LABELS, index=ids)  # fold mean
    print(f"{plane}: {len(ids)} exams, mean prob {pm.mean():.3f}  "
          f"({(time.time()-t0)/60:.0f} min)", flush=True)
    del models; torch.cuda.empty_cache()

allp = pd.concat(per_plane.values())
out = allp.groupby(level=0).mean()
out["n_planes"] = allp.groupby(level=0).size()
out.index.name = ID_COL
out.reset_index().to_csv("/kaggle/working/fastmri_teacher.csv", index=False)
print(f"\n{len(out)} exams labelled; planes per exam: "
      f"{out['n_planes'].value_counts().sort_index().to_dict()}")
print("mean teacher probability per label:")
print(out[LABELS].mean().round(3).to_string())
