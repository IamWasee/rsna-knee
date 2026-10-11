# ============================================================
# RSNA Knee — LEG RUNNER: several public pipelines, each in its own process,
# rank-blended. GPU (2x T4), internet OFF (a submission notebook).
#
# Each leg is a public notebook's code cells, kept in our private dataset
# abdullahwasee/hybrid-legs, executed by nbconvert as a separate kernel. A
# separate process means one leg's models and globals are gone, GPU memory
# included, before the next leg starts -- pasting two 20-model pipelines into one
# kernel shares neither safely.
#
#   --sub LEGS=pub954:0.8,fast2xt4:0.2     notebook:weight, run in that order
#
# A leg's submission.csv is moved to legs/<name>.csv as soon as it finishes, so
# the next leg cannot be mistaken for it. Every leg must succeed and cover every
# test study; otherwise submission.csv is removed and the run fails -- never a
# silent subset of the blend.
# Budget: legs run in order with what is left of BUDGET_H; a leg that overruns
# is killed and the run fails (loudly) instead of hitting Kaggle's 9 h wall.
# ============================================================
import os, sys, json, time, glob, shutil, subprocess
import numpy as np, pandas as pd

LEGS = [(n, float(w)) for n, w in (x.split(":") for x in "__LEGS__".split(","))]
assert abs(sum(w for _, w in LEGS) - 1) < 1e-6, LEGS
BUDGET_H = 8.5
T0 = time.time()
W = "/kaggle/working"
SUB = f"{W}/submission.csv"
OUT = f"{W}/legs"
os.makedirs(OUT, exist_ok=True)


def find_leg(name):
    # Inputs mount at several depths and as symlinks (datasets/<owner>/<slug>, notebooks/...):
    # walk with followlinks, never into the competition's DICOM tree.
    for root, dirs, files in os.walk("/kaggle/input", followlinks=True):
        dirs[:] = [d for d in dirs if d not in ("competitions", "rsna-knee-abnormality-detection",
                                                "train_series", "test_series")]
        if f"{name}.ipynb" in files:
            return os.path.join(root, f"{name}.ipynb")
    raise FileNotFoundError(f"{name}.ipynb: attach abdullahwasee/hybrid-legs")


def refuse(msg):
    if os.path.exists(SUB):
        os.remove(SUB)
    raise RuntimeError(msg)


try:
    timing = {}
    for name, w in LEGS:
        left = BUDGET_H * 3600 - (time.time() - T0)
        if left < 600:
            refuse(f"no time left for leg {name}")
        src = find_leg(name)
        nb = f"{W}/_leg_{name}.ipynb"
        shutil.copy(src, nb)
        if os.path.exists(SUB):
            os.remove(SUB)
        t = time.time()
        print(f"\n{'=' * 66}\nleg {name} (weight {w}): starting, {left/3600:.2f} h left\n{'=' * 66}", flush=True)
        r = subprocess.run([sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute",
                            "--ExecutePreprocessor.timeout=-1", "--ExecutePreprocessor.kernel_name=python3",
                            "--output", f"_leg_{name}_done.ipynb", nb], cwd=W, timeout=left)
        timing[name] = round(time.time() - t)
        if r.returncode != 0:
            refuse(f"leg {name} failed (exit {r.returncode}) -- see _leg_{name}_done.ipynb")
        if not os.path.exists(SUB):
            refuse(f"leg {name} wrote no submission.csv")
        shutil.move(SUB, f"{OUT}/{name}.csv")
        print(f"leg {name}: done in {timing[name]/60:.1f} min", flush=True)
        # the next leg starts on clean GPUs; the finished kernel's process is gone
        subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader"])

    ID = "StudyInstanceUID"
    frames = {n: pd.read_csv(f"{OUT}/{n}.csv", dtype={ID: str}) for n, _ in LEGS}
    base = frames[LEGS[0][0]]
    labels = [c for c in base.columns if c != ID]
    if len(labels) != 12 or not base[ID].is_unique:
        refuse(f"leg {LEGS[0][0]}: bad columns or duplicate ids")
    ranks = {}
    for n, df in frames.items():
        if set(df[ID]) != set(base[ID]) or not df[ID].is_unique or sorted(df.columns) != sorted(base.columns):
            refuse(f"leg {n} covers different studies or columns")
        df = df.set_index(ID).loc[base[ID]]
        if df[labels].isna().any().any() or (df[labels].nunique() <= 1).any():
            refuse(f"leg {n} has NaN or a constant column")
        ranks[n] = df[labels].rank(method="average", pct=True).values
    print("\nmean spearman between legs:")
    for a in ranks:
        print("  " + "  ".join(f"{np.mean([pd.Series(ranks[a][:, j]).corr(pd.Series(ranks[b][:, j]), method='spearman') for j in range(12)]):.3f}"
                               for b in ranks), f"  {a}")
    blend = sum(w * ranks[n] for n, w in LEGS)
    final = base[[ID]].copy()
    final[labels] = pd.DataFrame(blend).rank(method="average", pct=True).values
    assert final[labels].notna().all().all() and final.shape == (len(base), 13)
    final.to_csv(SUB, index=False)
    for f in glob.glob(f"{W}/*.csv"):          # only submission.csv at the top level
        if f != SUB:
            shutil.move(f, f"{OUT}/{os.path.basename(f)}")
    print(f"\nsubmission: {len(final)} studies from legs {LEGS}")
    print(f"leg minutes: { {k: round(v/60, 1) for k, v in timing.items()} }, total {(time.time()-T0)/60:.1f} min")
except BaseException:
    if os.path.exists(SUB):
        os.remove(SUB)
    raise
