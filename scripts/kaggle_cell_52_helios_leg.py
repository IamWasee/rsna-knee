# ============================================================
# RSNA Knee — heliosli's fine-tuned OrthoFoundation (fold 0, epoch 7, EMA best.pt):
# score it on gold-58, then predict the test set as a leg. GPU, internet ON
# (pip for the code's extra packages; a submission version will bundle wheels).
#
# Source: heliosli/rsna-orthofoundation-fold0-epoch7-20261004 (public), our private copy
# in the snap-rsna-orthofoundation-fold0-epoch7-2026 notebook output: best.pt
# (1.22 GB, full weights -- the model is rebuilt with pretrained=False and every
# tensor comes from the checkpoint), anatomy.pt, kneexnet/, cloud_code/ (their code).
# Their labels.py puts every official-gold study's group in fold -1, never trained
# on, so gold-58 is a clean ruler for this checkpoint as for our arms.
#
# 1. gold-58: a directory laid out like the competition's test split whose
#    test_series/ links to the 58 gold studies' train series; their infer.py runs
#    on it unchanged; macro AUC against the gold labels (aggregates printed only).
# 2. test: their infer.py on the real test split -> /kaggle/working/helios_test.csv,
#    timed, for the hybrid blend.
# Reference: our 9-arm team 0.9003, sag-combo 0.8985 on gold.
# ============================================================
import os, sys, json, time, glob, shutil, subprocess
from pathlib import Path
import numpy as np, pandas as pd

T0 = time.time()
W = Path("/kaggle/working")
# notebook outputs mount deep (/kaggle/input/notebooks/<owner>/<slug>/...); walk for receipt.json,
# never descending into the competition's DICOM tree
src = []
for root, dirs, files in os.walk("/kaggle/input", followlinks=True):   # outputs mount as symlinks
    dirs[:] = [d for d in dirs if d not in ("competitions", "rsna-knee-abnormality-detection", "train_series",
                                            "test_series", "cloud_code", "kneexnet")]
    if root.count(os.sep) > 9:
        dirs[:] = []
    if "receipt.json" in files:
        src.append(Path(root))
src = [s for s in src if ((s / "best.pt").is_file() or (s / "weight_parts.json").is_file()) and (s / "cloud_code").is_dir()]
if not src:
    for root, dirs, files in os.walk("/kaggle/input", followlinks=True):
        dirs[:] = [d for d in dirs if d not in ("competitions", "rsna-knee-abnormality-detection", "cloud_code")]
        depth = root.count(os.sep) - 2
        if depth > 7:
            dirs[:] = []
            continue
        print("  " * depth + root, files[:5])
    raise SystemExit("attach the snap-rsna-orthofoundation-fold0-epoch7-2026 output (tree above)")
SRC = src[0]
rec = json.loads((SRC / "receipt.json").read_text())
print("checkpoint", SRC, {k: rec[k] for k in ("fold", "epoch_one_based", "weights", "validation_macro_auc_weak_reference")})

# their code + assets in one writable place: infer.py looks for anatomy.pt next to the checkpoint
import hashlib
H = Path("/tmp/helios"); shutil.rmtree(H, ignore_errors=True); H.mkdir()
shutil.copytree(SRC / "cloud_code", H / "cloud_code")
# the dataset now ships best.pt as 128 MB parts; rebuild it and check the assembled hash
if (SRC / "best.pt").is_file():
    os.symlink(SRC / "best.pt", H / "best.pt")
else:
    parts = json.loads((SRC / "weight_parts.json").read_text())
    h = hashlib.sha256()
    with open(H / "best.pt", "wb") as dst:
        for part in parts["parts"]:
            blob = (SRC / part["name"]).read_bytes()
            assert len(blob) == part["bytes"] and hashlib.sha256(blob).hexdigest() == part["sha256"], part["name"]
            dst.write(blob); h.update(blob)
    assert h.hexdigest() == parts["assembled_sha256"] == rec["files"]["best.pt"]["sha256"], "assembled best.pt differs"
    del blob
    print("best.pt assembled from", len(parts["parts"]), "parts")
# the helper weights may sit anywhere in this dataset version: place each receipt file by
# name where infer.py looks (next to best.pt; kneexnet/ beside it)
(H / "kneexnet").mkdir()
everywhere = {}
for root, dirs, files in os.walk(SRC, followlinks=True):
    dirs[:] = [d for d in dirs if d not in ("cloud_code", "dependencies")]
    for f in files:
        everywhere.setdefault(f, Path(root) / f)
for name in rec["files"]:
    if name in ("best.pt", "cloud_code.zip"):
        continue
    hit = everywhere.get(Path(name).name)
    if hit is None:
        print("snapshot files:", sorted(everywhere)[:60])
        raise SystemExit(f"{name} not found in the snapshot")
    os.symlink(hit, H / name)
    print(f"  {name} <- {hit.relative_to(SRC)}")
for name, meta in rec["files"].items():
    p = H / name if (H / name).exists() else (H / "cloud_code" / name)
    if name == "cloud_code.zip":
        continue
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 24), b""):
            h.update(b)
    assert h.hexdigest() == meta["sha256"], f"{name} differs from heliosli's receipt"
print("receipt hashes verified", f"{(time.time()-T0)/60:.1f} min")

# extra packages their code imports
wheels = sorted(str(p) for p in SRC.rglob("*.whl"))
if wheels:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-index", "--no-deps", *wheels], check=False)
# their environment pins timm 1.0.25 (DINOv3 ViTs); --no-deps so torch is not touched. MONAI is vendored.
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "timm==1.0.25"], check=False, timeout=900)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "h5py", "psutil"], check=False, timeout=900)
print("timm", subprocess.run([sys.executable, "-c", "import timm; print(timm.__version__)"],
                             capture_output=True, text=True).stdout.strip())

# competition root
COMP = next(Path(p).parent for p in ["/kaggle/input/competitions/rsna-knee-abnormality-detection/test.csv",
                                      "/kaggle/input/rsna-knee-abnormality-detection/test.csv"] if Path(p).exists())
print("competition", COMP, sorted(p.name for p in COMP.iterdir())[:12])
train = pd.read_csv(COMP / "train.csv", dtype={"StudyInstanceUID": str})
LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion",
          "Synovitis", "Baker's", "Contusion", "Fracture"]
gold = train[train[LABELS].notna().all(axis=1)]
print(f"{len(gold)} gold studies")

# a test-shaped directory over the gold studies
G = Path("/tmp/golddata"); shutil.rmtree(G, ignore_errors=True); G.mkdir()
test_cols = list(pd.read_csv(COMP / "test.csv", nrows=1).columns)
gold[[c for c in test_cols if c in gold.columns]].to_csv(G / "test.csv", index=False)
if (COMP / "test_series.csv").exists() and (COMP / "train_series.csv").exists():
    ser = pd.read_csv(COMP / "train_series.csv", dtype=str)
    ser[ser["StudyInstanceUID"].isin(gold["StudyInstanceUID"])].to_csv(G / "test_series.csv", index=False)
(G / "test_series").mkdir()
for u in gold["StudyInstanceUID"]:
    os.symlink(COMP / "train_series" / u, G / "test_series" / u)
ss = pd.read_csv(COMP / "sample_submission.csv", nrows=1)
pd.DataFrame({"StudyInstanceUID": gold["StudyInstanceUID"], **{c: 0.5 for c in ss.columns[1:]}}).to_csv(
    G / "sample_submission.csv", index=False)


def run(data, out, budget_s):
    t = time.time()
    env = dict(os.environ, PYTHONPATH=str(H / "cloud_code"))
    r = subprocess.run([sys.executable, "-m", "training.infer", "--data", str(data), "--checkpoint", str(H / "best.pt"),
                        "--output", str(out), "--anatomy-weights", str(H / "anatomy.pt")],
                       env=env, cwd=str(H / "cloud_code"), timeout=budget_s)
    assert r.returncode == 0, f"heliosli infer failed on {data}"
    subs = sorted(Path(out).rglob("submission.csv"))
    assert subs, f"no submission.csv under {out}"
    return pd.read_csv(subs[0], dtype={"StudyInstanceUID": str}), time.time() - t


from sklearn.metrics import roc_auc_score
# caches go to /tmp: full-resolution competition volumes must not land in the notebook output
pred, secs = run(G, Path("/tmp/helios_gold"), 3 * 3600)
print("gold native cache:", subprocess.run(["du", "-sh", "/tmp/helios_gold"], capture_output=True, text=True).stdout.strip())
pred = pred.set_index("StudyInstanceUID").loc[gold["StudyInstanceUID"]]
per = {c: roc_auc_score(gold[c].astype(int), pred[c]) for c in LABELS}
print("\n" + "=" * 66)
print(f"heliosli OrthoFoundation fold0 on gold-58: macro AUC {np.mean(list(per.values())):.4f}  "
      f"({secs/60:.1f} min for {len(gold)} studies, {secs/len(gold):.1f} s/study)")
for c, a in per.items():
    print(f"  {c:<18} {a:.3f}")
print("ours for reference: 9-arm team 0.9003 | sag-combo 0.8985")
np.save(W / "helios_gold_pred.npy", pred[LABELS].values)
gold[["StudyInstanceUID"]].to_csv(W / "helios_gold_ids.csv", index=False)
shutil.rmtree("/tmp/helios_gold", ignore_errors=True)
shutil.rmtree(G, ignore_errors=True)

# bf16 autocast on a T4 may be emulated: project the test run from the gold timing first
n_test = len(pd.read_csv(COMP / "test.csv"))
proj = secs / len(gold) * n_test
print(f"projected test run: {proj/60:.0f} min for {n_test} studies")
tsecs = None
if proj > 0.8 * 4 * 3600:
    print("SKIPPING test run: projection exceeds 80% of its 4 h budget")
else:
    tpred, tsecs = run(COMP, Path("/tmp/helios_test"), 4 * 3600)
    test_ids = set(pd.read_csv(COMP / "test.csv", dtype={"StudyInstanceUID": str})["StudyInstanceUID"])
    assert set(tpred["StudyInstanceUID"]) == test_ids and tpred["StudyInstanceUID"].is_unique, "test coverage"
    tpred.to_csv(W / "helios_test.csv", index=False)
    shutil.rmtree("/tmp/helios_test", ignore_errors=True)
if tsecs:
    print(f"\ntest: {len(tpred)} studies in {tsecs/60:.1f} min ({tsecs/max(1,len(tpred)):.1f} s/study)")
json.dump({"gold_macro": float(np.mean(list(per.values()))), "gold_per_label": per,
           "gold_seconds": secs, "test_seconds": tsecs, "projected_test_seconds": proj},
          open(W / "helios_leg.json", "w"), indent=1)
print(f"total {(time.time()-T0)/60:.0f} min")
