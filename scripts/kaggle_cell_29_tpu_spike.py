# ============================================================
# RSNA Knee — does our training loop run on a TPU at all?
#
# TIME-BOXED. The only question is whether one fold runs end to end. Not tuned,
# not fast, not compared -- just running.
#
# What this is NOT for, since I said otherwise before checking: a TPU is not a
# graphics card with eight times the memory. Each of its chips has roughly the
# working space of the T4 we already use; the eight are for running eight copies
# at once, not for fitting one bigger model. So this does NOT lift the ceiling
# that stopped a large encoder -- --encoder-chunk is what does that.
#
# What it IS for: a second weekly allowance we have never touched. 20 hours a
# week, zero used, seven weeks left -- 140 hours sitting idle next to a graphics
# allowance we burned 28 of 30 hours of. This buys experiments, not capacity.
#
# Three things differ from the GPU path, all inside Backend in train.py:
#   half precision is bfloat16 (no gradient scaler needed -- bfloat16 keeps
#   float32's exponent range); the optimiser step goes through xm.optimizer_step,
#   which also flushes the queued graph; and reading the loss every step forces
#   that flush, which is why per-step logging can dominate the runtime.
#
# SET THE ACCELERATOR TO TPU, not GPU. Internet on. Target: under 40 minutes.
# ============================================================
import sys, os, time
CODE = "/kaggle/working/rsna-knee"
!rm -rf $CODE && git clone -q https://github.com/IamWasee/rsna-knee.git $CODE
sys.path.insert(0, f"{CODE}/src")

# Everything that touches the TPU runs in a SUBPROCESS, never in this notebook.
# A TPU has exactly one owner: importing torch_xla here claims it, and the
# training process launched afterwards then aborts with a core dump trying to
# claim it again. That is what killed the first attempt -- not the platform.
probe = "/kaggle/working/probe_tpu.py"
open(probe, "w").write("""
import torch, torch_xla, torch_xla.core.xla_model as xm
d = xm.xla_device()
print(f"torch {torch.__version__}, torch_xla {torch_xla.__version__}")
print(f"device {d} -> {xm.xla_real_devices([str(d)])[0]}")
print(f"visible cores: {len(xm.get_xla_supported_devices())}")
a = torch.randn(256, 256, device=d)
print(f"matmul on TPU: {(a @ a).sum().item():.1f}")
""")
!python $probe
print()

!pip install -q timm transformers
from kaggle_paths import find, describe
v3 = [f for f in find(suffix=".npy") if "cache_v3" in f]
lab = find(filename="llm_labels_v4_blend.csv")
dino = [os.path.dirname(p) for p in find(filename="config.json") if "dinov2" in p.lower()]
if not (v3 and lab and dino):
    describe(); raise SystemExit("attach cache-v3, the stevenleehans labels, and dinov2")
CACHE = os.path.dirname(v3[0])

# Two epochs of one fold. Enough to prove the loop closes and the loss falls;
# not enough to compare against anything, and not offered as a comparison.
t0 = time.time()
!python $CODE/src/train.py --cache "$CACHE" --labels "{lab[0]}" \
    --backbone "dinov2:{dino[0]}" --device tpu --size 288 \
    --only-fold 0 --epochs 2 --batch 8 --workers 2 \
    --lr 1e-3 --lr-backbone 8e-6 --head slot --pool focal \
    --out /kaggle/working/tpu_probe
mins = (time.time() - t0) / 60
print(f"\nwall clock: {mins:.1f} min for 2 epochs of 1 fold")

import glob
ok = bool(glob.glob("/kaggle/working/tpu_probe/*.pt"))
print("\n" + "=" * 60)
if ok:
    print("VERDICT: the loop runs on TPU.")
    print(f"For reference, the same 2 epochs on a T4 take about 5.5 minutes.")
    print(f"This took {mins:.1f}. If that ratio is under ~2x, the 20 free hours a")
    print("week are worth having. If it is 5x or worse, they are not -- the")
    print("allowance is only useful if an experiment finishes in a sitting.")
else:
    print("VERDICT: no checkpoint written -- read the error above.")
    print("Two hours was the budget for this. If the failure is not obvious,")
    print("stop here; the GPU path is not blocked on it.")
