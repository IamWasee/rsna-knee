# ============================================================
# RSNA Knee — three claims worth settling before spending GPU on them
#
# 1. PER-LABEL label quality. I have repeated for weeks that our rule-based labels
#    beat the public ones on ACL, Fracture, Contusion and Baker's. The label
#    benchmark only ever printed the MACRO, so that claim has no log behind it.
#    Print the full twelve-row matrix and settle it. A published per-label table
#    also suggests pilkwang's Fracture column (0.871) beats stevenleehans' (0.793)
#    -- Fracture is our worst image label, so borrowing one column may be worth
#    more than blending whole tables, which is measured to gain nothing.
#
# 2. FRACTURE TARGET VARIANCE. The claim is that Fracture fails not because the
#    reader is imprecise but because the target is nearly degenerate -- one public
#    reader assigns ~97% of studies the same value, which carries almost no
#    gradient no matter how good the images are. Check the distribution directly.
#
# 3. THE 140mm CROP. A public notebook claims every series here is acquired at a
#    150mm field of view. If true, our fixed 140mm physical crop is close to a
#    no-op and the only real lever is pixel count -- which would change how much a
#    384px cache is worth.
#
# CPU, ~3 min. Attach the competition plus every label dataset you can:
#   stevenleehans/rsna-knee-llm-report-labels
#   pilkwang/rsna-knee-llm-labels
#   lixin73/rsna-knee-llm-report-labels-sol56
#   laymond/rsna-knee-abnormality-qwen3-8b-weak-labels
#   yunusgmsoy/rsna-knee-llm-labels-4-source-merged
#   rayanbabur/rsna-knee-2026-calibrated-soft-targets
# ============================================================
!pip install -q pydicom

import sys, glob, os
import numpy as np, pandas as pd

CODE = "/kaggle/working/rsna-knee"
!rm -rf $CODE && git clone -q https://github.com/IamWasee/rsna-knee.git $CODE
sys.path.insert(0, f"{CODE}/src")
from config import ID_COL, LABELS
from sklearn.metrics import roc_auc_score

TRAIN = next(iter(glob.glob("/kaggle/input/competitions/**/train.csv", recursive=True)))
train = pd.read_csv(TRAIN)
gold = train[train[LABELS].notna().all(axis=1)].set_index(ID_COL)
print(f"{len(gold)} annotated studies\n")

# ---------------------------------------------------------------- 1. per label
def per_label(df):
    if ID_COL not in df.columns:
        return None
    sub = df.set_index(ID_COL).reindex(gold.index)
    out = {}
    for c in LABELS:
        if c not in sub:
            continue
        y = gold[c].values
        p = pd.to_numeric(sub[c], errors="coerce").values
        ok = ~np.isnan(p)
        if ok.sum() < 20 or len(set(y[ok])) < 2:
            continue
        out[c] = roc_auc_score(y[ok], p[ok])
    return out or None

tables = {}
for f in sorted(glob.glob("/kaggle/input/**/*.csv", recursive=True)):
    if os.path.getsize(f) > 40e6 or os.path.basename(f) in ("train.csv", "test.csv"):
        continue
    try:
        df = pd.read_csv(f)
    except Exception:
        continue
    got = per_label(df)
    if got and len(got) >= 10:
        tables[os.path.basename(f)[:-4][:26]] = got

if not tables:
    raise SystemExit("no usable label tables found -- attach the datasets listed above")

names = list(tables)
w = max(len(n) for n in names) + 2
print(f"{'label':<18}" + "".join(f"{n[:11]:>13}" for n in names) + f"{'best':>16}")
print("-" * (18 + 13 * len(names) + 16))
wins = {n: 0 for n in names}
for c in LABELS:
    row = {n: tables[n].get(c) for n in names}
    have = {n: v for n, v in row.items() if v is not None}
    best = max(have, key=have.get) if have else None
    if best:
        wins[best] += 1
    print(f"{c:<18}" + "".join(
        f"{(f'{row[n]:.3f}' if row[n] is not None else '--'):>13}" for n in names)
        + f"{(best or '')[:15]:>16}")
print("-" * (18 + 13 * len(names) + 16))
print(f"{'MACRO':<18}" + "".join(
    f"{np.mean(list(tables[n].values())):>13.3f}" for n in names))
print(f"\nlabels won: " + ", ".join(f"{n}={k}" for n, k in wins.items() if k))
print("A per-label pick is only worth taking where the gap is large: on 58 studies")
print("the 95% interval on a single-label AUC is roughly +/-0.10.")

# ---------------------------------------------------------------- 2. Fracture
print("\n" + "=" * 60 + "\nTARGET VARIANCE -- is a label degenerate?\n" + "=" * 60)
print(f"{'source':<28}{'label':<18}{'modal share':>12}{'std':>8}")
for n in names:
    for f in glob.glob("/kaggle/input/**/*.csv", recursive=True):
        if os.path.basename(f)[:-4][:26] != n:
            continue
        try:
            df = pd.read_csv(f)
        except Exception:
            continue
        for c in ("Fracture", "PF OA", "Synovitis"):
            if c not in df:
                continue
            v = pd.to_numeric(df[c], errors="coerce").dropna()
            if not len(v):
                continue
            share = v.value_counts(normalize=True).iloc[0]
            flag = "  <-- near-degenerate, almost no gradient" if share > 0.9 else ""
            print(f"{n:<28}{c:<18}{share:>12.1%}{v.std():>8.3f}{flag}")
        break

# ---------------------------------------------------------------- 3. the crop
print("\n" + "=" * 60 + "\nFIELD OF VIEW -- is the 140mm crop doing anything?\n" + "=" * 60)
import pydicom
from paths import data_root
root = data_root()
series = pd.read_csv(root / "train_series.csv")
rows, seen = [], 0
for r in series.sample(min(400, len(series)), random_state=0).itertuples():
    d = root / "train_series" / getattr(r, ID_COL) / getattr(r, "SeriesInstanceUID")
    fs = sorted(glob.glob(str(d / "*.dcm")))
    if not fs:
        continue
    try:
        ds = pydicom.dcmread(fs[len(fs) // 2], stop_before_pixels=True, force=True)
        ps = float(ds.PixelSpacing[0]); cols = int(ds.Columns)
    except Exception:
        continue
    rows.append((ps * cols, ps, cols)); seen += 1
    if seen >= 300:
        break

fov = np.array([r[0] for r in rows])
print(f"{len(fov)} series sampled")
print(f"field of view mm: min {fov.min():.0f}  p10 {np.percentile(fov,10):.0f}  "
      f"median {np.median(fov):.0f}  p90 {np.percentile(fov,90):.0f}  max {fov.max():.0f}")
print(f"share within 145-155mm: {((fov>=145)&(fov<=155)).mean():.1%}")
print(f"share smaller than our 140mm crop: {(fov < 140).mean():.1%}")
if np.percentile(fov, 10) > 140 and np.percentile(fov, 90) < 165:
    print("-> the FOV is near-constant, so the 140mm crop mostly just trims a margin.")
    print("   Resolution, not physical extent, is the lever. A 384px cache is worth more.")
else:
    print("-> the FOV varies materially, so the physical crop IS normalising scale")
    print("   across scanners and is doing real work. Keep it.")
