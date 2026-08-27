# ============================================================
# RSNA Knee — Is there a metadata shortcut?
#
# Predicts the 12 findings from acquisition metadata ONLY -- no pixels.
# If this scores near the image model, the signal is protocol, not anatomy.
#
# Attach: the competition + a notebook output containing report_labels.csv.
# CPU is fine. ~2 min without --deep, ~20 min with.
# ============================================================
import sys, glob, os
CODE = "/kaggle/working/rsna-knee"
!rm -rf $CODE && git clone -q https://github.com/IamWasee/rsna-knee.git $CODE
sys.path.insert(0, f"{CODE}/src")

import pandas as pd
found = glob.glob("/kaggle/input/**/report_labels.csv", recursive=True)
LABELS = max(found, key=lambda f: len(pd.read_csv(f)))
print("labels:", LABELS, f"({len(pd.read_csv(LABELS))} studies)")

# Fast: series counts, planes, sequence mix, protocol recipe.
!python $CODE/src/metadata_probe.py --labels "$LABELS"

# Slower: adds scanner, TR/TE, slice geometry from one DICOM header per series.
# !python $CODE/src/metadata_probe.py --labels "$LABELS" --deep
