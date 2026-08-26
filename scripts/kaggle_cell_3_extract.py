# ============================================================
# RSNA Knee — Step 3: report -> labels with Claude
# Run in a KAGGLE notebook with Internet ON (Settings -> Internet).
# This is NOT the submission notebook; the submission one stays offline
# and just reads the dataset this produces.
#
# Add your Anthropic key via Kaggle Add-ons -> Secrets, named ANTHROPIC_API_KEY.
# ============================================================
!pip install -q anthropic

import os, sys
from kaggle_secrets import UserSecretsClient
os.environ["ANTHROPIC_API_KEY"] = UserSecretsClient().get_secret("ANTHROPIC_API_KEY")

# Pull the pipeline code. llm_extract.py is 300 lines -- clone it, don't paste it.
REPO = ""   # e.g. "https://github.com/yourname/rsna-knee.git"
if not REPO:
    raise SystemExit("Set REPO to your GitHub repo URL first.")

CODE = "/kaggle/working/rsna-knee"
!rm -rf $CODE && git clone -q $REPO $CODE
sys.path.insert(0, f"{CODE}/src")

TRAIN = "/kaggle/input/rsna-knee-abnormality-detection/train.csv"

# 1. What will the full run cost?
!python $CODE/src/llm_extract.py --data $TRAIN --estimate

# 2. Validate on the 58 gold studies FIRST. ~$0.50, and it tells you whether the
#    prompt works before you spend on 4,349 reports.
!python $CODE/src/llm_extract.py --data $TRAIN --validate

# 3. Only after the AUC looks right, run the full batch.
# !python $CODE/src/llm_extract.py --data $TRAIN --all \
#     --out /kaggle/working/report_labels.csv

# 4. Save /kaggle/working/report_labels.csv as a Kaggle Dataset. The offline
#    submission notebook attaches that dataset instead of calling any API.
