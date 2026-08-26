# ============================================================
# RSNA Knee — Step 1: what does the data actually look like?
# Paste this whole thing into ONE Colab cell and run it.
# Runtime: CPU is fine.
# ============================================================
!pip install -q kaggle pydicom

# --- Kaggle credentials (via Colab Secrets — never paste the token in a cell) -----
# One-time setup in Colab:
#   1. Left sidebar -> the key icon (Secrets) -> "+ Add new secret"
#   2. Name:  KAGGLE_API_TOKEN
#   3. Value: your KGAT_... token
#   4. Toggle "Notebook access" ON
import os

token = None
try:
    from google.colab import userdata
    token = userdata.get("KAGGLE_API_TOKEN")
except Exception as e:
    print("Could not read Colab secret:", e)

if token:
    os.environ["KAGGLE_API_TOKEN"] = token
    os.makedirs("/root/.kaggle", exist_ok=True)
    with open("/root/.kaggle/access_token", "w") as f:
        f.write(token.strip())
    os.chmod("/root/.kaggle/access_token", 0o600)
    print("credentials in place (new-style KGAT token)")
elif os.path.exists("/root/.kaggle/kaggle.json"):
    print("using existing legacy kaggle.json")
else:
    raise SystemExit(
        "No credentials. Add KAGGLE_API_TOKEN as a Colab Secret (key icon in the "
        "left sidebar), enable Notebook access, then re-run this cell."
    )

# Verify auth before doing anything expensive.
!kaggle competitions list -s knee 2>&1 | head -5

COMP = "rsna-knee-abnormality-detection"
os.makedirs(f"/content/{COMP}", exist_ok=True)

# --- How big is it, and what files exist? ---------------------------------
print("\n" + "="*70)
print("FILE LIST (note the sizes — this decides Colab vs Kaggle)")
print("="*70)
!kaggle competitions files -c $COMP
!df -h /content | tail -1

# --- Grab only the small metadata files -----------------------------------
print("\n" + "="*70)
print("DOWNLOADING SMALL FILES ONLY")
print("="*70)
for fname in ["train.csv", "sample_submission.csv", "train_labels.csv", "test.csv"]:
    !kaggle competitions download -c $COMP -f "$fname" -p /content/$COMP 2>/dev/null

!cd /content/$COMP && unzip -o -q '*.zip' 2>/dev/null; ls -la /content/$COMP

# --- Look at every CSV ----------------------------------------------------
import pandas as pd, glob
pd.set_option("display.width", 200)

for csv in sorted(glob.glob(f"/content/{COMP}/*.csv")):
    df = pd.read_csv(csv)
    print("\n" + "="*70)
    print(f"{os.path.basename(csv)}   shape={df.shape}")
    print("="*70)
    print("columns:", list(df.columns))
    print(df.head(3).to_string(max_colwidth=70))

    # binary label columns -> prevalence
    for c in df.columns:
        if df[c].dropna().isin([0, 1]).all() and df[c].nunique() <= 2:
            print(f"  {c:<20} positive rate {df[c].mean():.4f}  (n={int(df[c].sum())})")

    # long free-text columns are the radiology reports
    for c in df.columns:
        if df[c].dtype == object:
            L = df[c].astype(str).str.len()
            if L.mean() > 100:
                print(f"\n  >>> '{c}' is report text (mean length {L.mean():.0f} chars)")
                print("  ---- SAMPLE REPORT ----")
                print("  " + str(df[c].iloc[0])[:1200].replace("\n", "\n  "))
