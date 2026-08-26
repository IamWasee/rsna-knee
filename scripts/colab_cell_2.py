# ============================================================
# RSNA Knee — Step 2: label structure, languages, imaging volume
# Paste into a NEW Colab cell (step 1 already ran, creds in place).
# ============================================================
!pip install -q --upgrade kaggle
!pip install -q langdetect pydicom

import os, glob, pandas as pd, numpy as np
COMP = "rsna-knee-abnormality-detection"
D = f"/content/{COMP}"
pd.set_option("display.width", 220)

LABELS = ["ACL","MCL","Medial Meniscus","Lateral Meniscus","Medial OA","Lateral OA",
          "PF OA","Effusion","Synovitis","Baker's","Contusion","Fracture"]

train = pd.read_csv(f"{D}/train.csv")

# ---------- A. Exactly how is the supervision distributed? ----------
print("="*70); print("A. LABEL STRUCTURE"); print("="*70)
lab = train[LABELS]
n_lab_per_row = lab.notna().sum(axis=1)
print(f"total studies:            {len(train)}")
print(f"rows with >=1 label:      {(n_lab_per_row > 0).sum()}")
print(f"rows with all 12 labels:  {(n_lab_per_row == 12).sum()}")
print(f"rows with 0 labels:       {(n_lab_per_row == 0).sum()}")
print("\ndistribution of labels-per-row (nonzero only):")
print(n_lab_per_row[n_lab_per_row > 0].value_counts().sort_index().to_string())

print("\nnon-null count per label:")
for c in LABELS:
    print(f"  {c:<18} n={lab[c].notna().sum():<5} pos={int(lab[c].sum()):<4} rate={lab[c].mean():.3f}")

print("\nunique values present:", sorted(pd.unique(lab.values.ravel())[:10], key=str))

# ---------- B. What language is each report in? ----------
print("\n" + "="*70); print("B. REPORT LANGUAGES"); print("="*70)
from langdetect import detect, DetectorFactory
DetectorFactory.seed = 0

def lang(t):
    try: return detect(str(t)[:600])
    except Exception: return "??"

sample = train["Report"].sample(min(600, len(train)), random_state=0)
print(sample.map(lang).value_counts().to_string())

L = train["Report"].astype(str).str.len()
print(f"\nreport length: min={L.min()} p25={L.quantile(.25):.0f} median={L.median():.0f} "
      f"p75={L.quantile(.75):.0f} max={L.max()}")
print(f"missing reports: {train['Report'].isna().sum()}")

import re
ph = re.findall(r"\[[A-Z_]+\]", " ".join(train["Report"].astype(str).head(2000)))
print("de-id placeholders:", pd.Series(ph).value_counts().head(10).to_dict())

print("\n---- 2 more full reports ----")
for i in [10, 200]:
    print(f"\n[{i}] {str(train['Report'].iloc[i])[:900]}")

# ---------- C. Do the 58 labeled rows differ from the rest? ----------
print("\n" + "="*70); print("C. LABELED SUBSET"); print("="*70)
has = n_lab_per_row > 0
print(f"labeled report length  median={L[has].median():.0f}")
print(f"unlabeled report length median={L[~has].median():.0f}")
print("\nlabeled-subset study UIDs (first 5):")
print(train.loc[has, "StudyInstanceUID"].head().to_string(index=False))

# ---------- D. How much imaging is there? ----------
print("\n" + "="*70); print("D. IMAGING VOLUME"); print("="*70)
for f in ["train_series.csv", "test_series.csv"]:
    !kaggle competitions download -c $COMP -f "$f" -p $D 2>&1 | tail -1
!cd $D && unzip -o -q '*.zip' 2>/dev/null; true

for f in ["train_series.csv", "test_series.csv"]:
    p = f"{D}/{f}"
    if os.path.exists(p):
        s = pd.read_csv(p)
        print(f"\n{f}: shape={s.shape}")
        print("columns:", list(s.columns))
        print(s.head(5).to_string())
        for c in s.columns:
            if s[c].dtype == object and s[c].nunique() < 60:
                print(f"\n  value counts for '{c}':")
                print("  " + s[c].value_counts().head(25).to_string().replace("\n", "\n  "))
        # slices ~1.84 MB each -> estimate total download size
        if "SeriesInstanceUID" in s.columns:
            per_study = s.groupby("StudyInstanceUID").size()
            print(f"\n  series per study: median={per_study.median():.0f} max={per_study.max()}")
            print(f"  total series in {f}: {len(s)}")
            for slices in (20, 30, 40):
                gb = len(s) * slices * 1.84 / 1024
                print(f"    if ~{slices} slices/series -> ~{gb:.0f} GB")

!df -h /content | tail -1
