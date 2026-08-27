# ============================================================
# RSNA Knee — Pull the training output via the Kaggle API
#
# Attaching a notebook's output mounts whichever version was attached, and a
# later junk version can shadow a good one. The API fetches the output files
# directly, which avoids the UI version trap entirely.
#
# Run in a HELPER notebook: Internet ON, CPU is fine, no inputs needed.
# Needs your KAGGLE_API_TOKEN as a Kaggle Secret (Add-ons -> Secrets).
#
# Then Save & Run All, and attach THIS notebook to the submission notebook.
# ============================================================
!pip install -q --upgrade kaggle

import os, glob, shutil
from kaggle_secrets import UserSecretsClient

token = UserSecretsClient().get_secret("KAGGLE_API_TOKEN")
os.environ["KAGGLE_API_TOKEN"] = token
os.makedirs("/root/.kaggle", exist_ok=True)
with open("/root/.kaggle/access_token", "w") as f:
    f.write(token.strip())
os.chmod("/root/.kaggle/access_token", 0o600)

KERNEL = "abdullahwasee/notebookd"
DEST = "/kaggle/working/from_notebookd"
os.makedirs(DEST, exist_ok=True)

!kaggle kernels output $KERNEL -p $DEST

# What actually came down?
print("\n=== downloaded ===")
pts = sorted(glob.glob(f"{DEST}/**/*.pt", recursive=True))
srcs = sorted(glob.glob(f"{DEST}/**/infer.py", recursive=True))
for f in sorted(glob.glob(f"{DEST}/**/*", recursive=True))[:40]:
    if os.path.isfile(f):
        print(f"  {os.path.relpath(f, DEST)}  ({os.path.getsize(f)/1e6:.1f} MB)")

print(f"\ncheckpoints: {len(pts)}")
print(f"src/infer.py: {len(srcs)}")

if not pts:
    raise SystemExit(
        "No .pt files came down -- the API returned a version without output.\n"
        "Fall back to the UI: delete the junk version so the good one is latest."
    )

# Flatten into the layout the submission notebook expects.
os.makedirs("/kaggle/working/weights", exist_ok=True)
for p in pts:
    shutil.copy(p, f"/kaggle/working/weights/{os.path.basename(p)}")
if srcs:
    src_root = os.path.dirname(os.path.dirname(srcs[0]))
    shutil.copytree(src_root, "/kaggle/working/rsna-knee", dirs_exist_ok=True)
else:
    # No src in the output? Clone it -- the code is on GitHub anyway.
    !git clone -q https://github.com/IamWasee/rsna-knee.git /kaggle/working/rsna-knee

shutil.rmtree(DEST, ignore_errors=True)   # keep the saved output small
print("\nready:", sorted(glob.glob("/kaggle/working/weights/*.pt")))
print("src   :", os.path.isdir("/kaggle/working/rsna-knee/src"))
print("\nNow Save & Run All, then attach THIS notebook to the submission notebook.")
