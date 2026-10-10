# ============================================================
# RSNA Knee — HYBRID, step 2 of 4: put our team's submission aside. Internet OFF.
#
# The hybrid notebook is four parts, pushed as one notebook:
#   1. cell 37 -- our nine arms write /kaggle/working/submission.csv
#   2. this cell -- move it to hybrid/ours.csv, free the disk cell 37 used
#   3. the public 0.954 notebook (rabari9999/rsna-knee-lb-0-954-rank-127-silver),
#      its code cells verbatim: CoAtNet 224 + 384 SWA (nartaa) and the 2.5D
#      ConvNeXt reader (goodpjw2008), fused into its own submission.csv
#   4. cell 48b -- rank-blend the two, weight __W__ on ours
#
# Ours runs first because it is the cheaper half: if it fails it fails in
# minutes, not after the public pipeline's hours.
# ============================================================
import os, shutil, glob, time
import pandas as pd

H = "/kaggle/working/hybrid"
os.makedirs(H, exist_ok=True)
src = "/kaggle/working/submission.csv"
# If cell 37 stopped (a SystemExit most likely halts the run anyway) there is
# no submission.csv here, and cell 48b then refuses to file
# the public half on its own.
if not os.path.exists(src):
    raise RuntimeError("cell 37 wrote no submission.csv -- our team did not run")
ours = pd.read_csv(src)
assert ours.shape[1] == 13 and ours.notna().all().all(), ours.shape
shutil.move(src, f"{H}/ours.csv")
if os.path.isdir("/kaggle/working/parts"):
    shutil.move("/kaggle/working/parts", f"{H}/parts")
# test caches: several GB of the 20 GB working quota the public pipeline needs next
for d in glob.glob("/kaggle/working/test_*"):
    shutil.rmtree(d, ignore_errors=True)
open(f"{H}/stashed_at.txt", "w").write(str(time.time()))
print(f"ours stashed: {len(ours)} studies -> {H}/ours.csv")
print(f"working dir now: {sorted(os.listdir('/kaggle/working'))}")
