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
REPO = "github.com/IamWasee/rsna-knee.git"
CODE = "/kaggle/working/rsna-knee"

# Private repo? Add a GitHub token (repo scope) as a Kaggle Secret named
# GITHUB_TOKEN. If the repo is public this is skipped automatically.
try:
    gh = UserSecretsClient().get_secret("GITHUB_TOKEN")
    url = f"https://{gh}@{REPO}"
except Exception:
    url = f"https://{REPO}"

!rm -rf $CODE && git clone -q $url $CODE
sys.path.insert(0, f"{CODE}/src")
print("code cloned:", os.path.isdir(f"{CODE}/src"))

# Kaggle adds a "competitions/" level for attached competition inputs.
import glob
TRAIN = next(iter(glob.glob("/kaggle/input/**/train.csv", recursive=True)), None)
if not TRAIN:
    raise SystemExit("train.csv not found -- attach the competition under Input.")
print("train.csv:", TRAIN)

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
