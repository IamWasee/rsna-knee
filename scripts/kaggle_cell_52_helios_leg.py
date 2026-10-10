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
src = []
for pat in ("/kaggle/input/*/snapshot/datasets/*/receipt.json", "/kaggle/input/*/*/snapshot/datasets/*/receipt.json",
            "/kaggle/input/*/*/*/snapshot/datasets/*/receipt.json"):
    src += [Path(p).parent for p in glob.glob(pat)]
src = [s for s in src if (s / "best.pt").is_file() and (s / "cloud_code").is_dir()]
assert len(src) >= 1, "attach the snap-rsna-orthofoundation-fold0-epoch7-2026 output"
SRC = src[0]
rec = json.loads((SRC / "receipt.json").read_text())
print("checkpoint", SRC, {k: rec[k] for k in ("fold", "epoch_one_based", "weights", "validation_macro_auc_weak_reference")})

# their code + assets in one writable place: infer.py looks for anatomy.pt next to the checkpoint
H = Path("/tmp/helios"); shutil.rmtree(H, ignore_errors=True); H.mkdir()
shutil.copytree(SRC / "cloud_code", H / "cloud_code")
for f in ("best.pt", "anatomy.pt"):
    os.symlink(SRC / f, H / f)
os.symlink(SRC / "kneexnet", H / "kneexnet")
import hashlib
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
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "monai", "segmentation-models-pytorch", "ultralytics",
                "h5py", "psutil"], check=False, timeout=1200)

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
G = W / "golddata"; shutil.rmtree(G, ignore_errors=True); G.mkdir()
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
pred, secs = run(G, W / "helios_gold", 3 * 3600)
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

tpred, tsecs = run(COMP, W / "helios_test", 4 * 3600)
tpred.to_csv(W / "helios_test.csv", index=False)
print(f"\ntest: {len(tpred)} studies in {tsecs/60:.1f} min ({tsecs/max(1,len(tpred)):.1f} s/study)")
json.dump({"gold_macro": float(np.mean(list(per.values()))), "gold_per_label": per,
           "gold_seconds": secs, "test_seconds": tsecs, "test_rows": len(tpred)},
          open(W / "helios_leg.json", "w"), indent=1)
for d in ("golddata", "helios_gold", "helios_test"):
    for f in (W / d).rglob("native_test_cache"):
        shutil.rmtree(f, ignore_errors=True)
print(f"total {(time.time()-T0)/60:.0f} min")
