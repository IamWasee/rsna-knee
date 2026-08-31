# ============================================================
# RSNA Knee — Train on the published label table
#
# Attach THREE inputs:
#   1. the competition
#   2. Notebook Output -> the v2 preprocessing notebook (cache_v2)
#   3. Dataset -> stevenleehans/rsna-knee-llm-report-labels
#
# Measured on the 58 annotated studies: our labels 0.725, this table 0.893. Only the
# label source changes from the previous run, so the difference is attributable.
#
# GPU on, Internet on. Delete Kaggle's os.walk starter cell.
# ============================================================
!pip install -q timm

import sys, glob, os
CODE = "/kaggle/working/rsna-knee"
!rm -rf $CODE && git clone -q https://github.com/IamWasee/rsna-knee.git $CODE
sys.path.insert(0, f"{CODE}/src")

import pandas as pd


def find(filename: str = "", suffix: str = "") -> list[str]:
    """Search the attached inputs without descending the DICOM archive.

    A recursive glob over /kaggle/input walks hundreds of thousands of DICOM files
    -- it spent 623 seconds here before failing. followlinks=True matters because
    Kaggle mounts notebook outputs as symlinks.
    """
    hits = []
    for root, dirs, files in os.walk("/kaggle/input", followlinks=True):
        dirs[:] = [d for d in dirs
                   if "rsna-knee-abnormality-detection" not in os.path.join(root, d)]
        for f in files:
            if (filename and f == filename) or (suffix and f.endswith(suffix)):
                hits.append(os.path.join(root, f))
                if suffix and len(hits) > 50:      # enough to identify the dir
                    return hits
    return hits


def competition_file(name: str) -> str:
    """train.csv sits one level inside the competition mount; '**' from there would
    walk train_series/ as well, which is the archive we are trying not to touch."""
    for root, dirs, files in os.walk("/kaggle/input/competitions", followlinks=True):
        dirs[:] = [d for d in dirs if d not in ("train_series", "test_series")]
        if name in files:
            return os.path.join(root, name)
    raise SystemExit(f"{name} not found -- is the competition attached?")


TRAIN = competition_file("train.csv")

# Prefer the v2 cache. Both may be attached and their tensor shapes differ, so
# training on the wrong one now raises at the first batch instead of reshaping.
npys = find(suffix=".npy")
v2 = [f for f in npys if "cache_v2" in f]
if not npys:
    print("NO .npy CACHE FOUND. What is attached:")
    for root, dirs, files in os.walk("/kaggle/input", followlinks=True):
        if "rsna-knee-abnormality-detection" in root:
            dirs[:] = []
            continue
        depth = root.count(os.sep) - 2
        if depth > 4:
            dirs[:] = []
            continue
        print("  " * depth + os.path.basename(root) + "/")
        for f in files[:5]:
            print("  " * (depth + 1) + f)
    raise SystemExit(
        "Attach the Preprocessing v2 notebook's output. If it is attached, its\n"
        "LATEST version has no cache_v2 -- pin the version that produced it\n"
        "(Version History -> ... -> Pin as default version), then re-add the input."
    )

CACHE = os.path.dirname(v2[0] if v2 else npys[0])
print(f"cache: {CACHE}")
if "cache_v2" not in CACHE:
    print("  WARNING: this looks like the v1 cache (3 series, 12 slices, 224px).")
    print("  Training will raise a shape mismatch. Attach the v2 notebook instead.")

# Label source. Measured on the 58 annotated studies:
#   ours (rule-based)            0.725
#   steven llm_labels_v4_blend   0.893
# Labels cap everything downstream, so prefer the published table when attached and
# fall back to regenerating ours only if it is not.
best = find(filename="llm_labels_v4_blend.csv")
PREDS = None
if best:
    LABELS_CSV = best[0]
    print(f"labels: {LABELS_CSV}  (published, 0.893 on the annotated 58)")
else:
    cands = find(filename="model_preds_all.csv")
    if not cands:
        raise SystemExit("no label table found -- attach the labels dataset.")
    PREDS = max(cands, key=lambda f: len(pd.read_csv(f)))
    print(f"labels: regenerating from {PREDS} (ours, 0.725)")

if PREDS is not None:
    !python $CODE/src/ensemble.py --data "$TRAIN" --model-preds "$PREDS" \
        --method mean --out /kaggle/working/report_labels_v2.csv
    LABELS_CSV = "/kaggle/working/report_labels_v2.csv"

LABELS = LABELS_CSV

# 6 -> 12 epochs, nothing else changed. At six every fold was still improving
# (fold2 0.746->0.747, fold3 0.742->0.743, fold4 0.756->0.757), so the schedule was
# cutting training off rather than converging it. Six was tuned when the labels were
# noisy; a cleaner target has more signal to extract before it starts memorising.
#
# Sharpening also stays on for this run despite reducing spread (0.349 -> 0.312) on
# this label table -- turning it off is a second variable and gets its own run.
!python $CODE/src/train.py --cache "$CACHE" --labels "$LABELS" \
    --epochs 12 --batch 8 --backbone resnet34 \
    --head shared --pool gap \
    --out /kaggle/working/weights_v2

print("\ncheckpoints:", sorted(glob.glob("/kaggle/working/weights_v2/*.pt")))
print("Save this notebook's output, then attach it to the submission notebook.")
