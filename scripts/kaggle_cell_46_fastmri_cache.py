# ============================================================
# RSNA Knee — fastMRI clinical knee DICOMs -> our plane caches. CPU, internet on.
#
# Streams one fastMRI knee DICOM archive straight from its download link and
# writes cache_{sag,cor,ax}/<id>.npy with the competition caches' exact
# preprocessing (src/fastmri_cache.py). No GPU quota; nothing touches a laptop.
#
#   --sub SRC=<signed archive URL>   NEVER commit it: the link is the access, the
#                                    data may not be redistributed (NYU DSA), and
#                                    this repo is cloned publicly. Pass it only here.
#   --sub SHARD=0 --sub SHARDS=4     batch1 (~134 GB, ~8,200 exams) in 4 notebooks,
#                                    ~2,050 exams / ~12 GB each, under the 20 GB
#                                    output limit; batch2 (~30 GB) in 1.
#   --sub LIMIT=0                    or e.g. 20 for a smoke test
#
# The output stays private to this account (fastMRI DSA: internal research only).
# ============================================================
SRC = "__SRC__"
SHARD, SHARDS, LIMIT = int("__SHARD__"), int("__SHARDS__"), int("__LIMIT__")
assert 0 <= SHARD < SHARDS

# fastMRI stores pixels as lossless JPEG 2000; pydicom needs a decoder for it.
!pip install -q pylibjpeg pylibjpeg-openjpeg

import os, sys, time
CODE = "/kaggle/working/rsna-knee"
!rm -rf $CODE && git clone -q https://github.com/IamWasee/rsna-knee.git $CODE
!git -C $CODE log --oneline -1

OUT = "/kaggle/working/fm"
t0 = time.time()
!timeout -k 60 39600 python -u $CODE/src/fastmri_cache.py --src "$SRC" --out $OUT --work /tmp/fm_work --size 288 --anchors 8 --band 0.15 0.85 --shard $SHARD --shards $SHARDS --limit $LIMIT
code = _exit_code
print(f"exit {code} after {(time.time()-t0)/60:.0f} min")
!rm -rf $CODE
!du -sh $OUT/cache_* 2>/dev/null
if code != 0:
    raise SystemExit(f"fastmri_cache.py exited {code}")
