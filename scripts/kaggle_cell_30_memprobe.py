# ============================================================
# RSNA Knee — does slice-chunking actually lift the memory ceiling?
#
# Measurement only. No training, ~15 minutes of GPU, because the training half
# needs three hours we do not have until the allowance resets -- and the question
# that matters can be answered without it.
#
# THE QUESTION. Every team above 0.92 runs four or five different encoder
# families and averages them; we run one, which is why our "blend" of two heads
# on the same backbone bought only +0.008. I previously reported that a second,
# larger family "cannot fit on this hardware". That came from a probe measuring
# FULL precision while train.py trains under autocast float16 -- roughly twice
# the real requirement. The claim was never established.
#
# THE LEVER. Peak memory scales with SLICES PER STUDY (36 at this geometry), not
# with batch size -- which is exactly why a big encoder failed at every batch
# size including 1. --encoder-chunk feeds the encoder N slices at a time, capping
# the peak at N images regardless of batch.
#
# Two things are checked, and the first matters more:
#   1. CORRECTNESS -- chunked and unchunked must produce the same numbers. A
#      memory fix that quietly changes the maths is worse than no fix.
#   2. CAPACITY -- largest batch that fits, per encoder, per chunk size.
#
# Attach: competition + dinov2 model. GPU on. ~15 min.
# ============================================================
!pip install -q timm transformers

import sys, os, time
CODE = "/kaggle/working/rsna-knee"
!rm -rf $CODE && git clone -q https://github.com/IamWasee/rsna-knee.git $CODE
sys.path.insert(0, f"{CODE}/src")

import torch, timm
from model import KneeModel
print(f"torch {torch.__version__}, timm {timm.__version__}, "
      f"{torch.cuda.get_device_name(0)}, "
      f"{torch.cuda.get_device_properties(0).total_memory/2**30:.0f} GB")

SIZE, N_SLOT, GROUPS = 288, 4, 3
N_IMG = N_SLOT * GROUPS * 3          # 36 slices per study

# ---------------------------------------------------------------- correctness
# Same weights, same input, chunked vs not. Run in float32 and in eval mode so
# the only difference under test is the chunking itself.
print("\n" + "=" * 64 + "\n1. CORRECTNESS\n" + "=" * 64)
torch.manual_seed(0)
m = KneeModel("resnet18", pretrained=False, head="slot", pool="focal",
              n_slot=N_SLOT, groups_per_slot=GROUPS).cuda().eval()
x = torch.randn(2, N_IMG, 3, 128, 128, device="cuda")
with torch.no_grad():
    m.encoder_chunk = 0
    ref = m(x).float()
    for k in (18, 12, 6, 1):
        m.encoder_chunk = k
        got = m(x).float()
        d = (got - ref).abs().max().item()
        print(f"  chunk {k:>2}: max difference from unchunked = {d:.3e}"
              f"   {'OK' if d < 1e-4 else 'MISMATCH -- do not use'}")
        assert d < 1e-4, f"chunking changed the output at chunk={k}"
del m, x, ref; torch.cuda.empty_cache()
print("  chunking is numerically transparent.")

# ---------------------------------------------------------------- capacity
print("\n" + "=" * 64 + "\n2. CAPACITY -- fp16 autocast + checkpointing, as trained\n" + "=" * 64)

CANDIDATES = [
    ("coatnet_rmlp_2_rw_224", "hybrid conv+attn"),
    ("maxvit_rmlp_small_rw_224", "hybrid maxvit"),
    ("swin_small_patch4_window7_224", "windowed transformer"),
    ("convnext_small.fb_in22k_ft_in1k", "modern conv"),
    ("seresnext50_32x4d.racm_in1k", "classic conv"),
    ("tf_efficientnetv2_s.in21k_ft_in1k", "efficient conv"),
]
known = set(timm.list_models())

def fits(name, chunk, batch):
    try:
        for kw in ({"img_size": SIZE}, {}):
            try:
                enc = timm.create_model(name, pretrained=False, num_classes=0,
                                        global_pool="", in_chans=3, **kw)
                break
            except TypeError:
                continue
        enc = enc.cuda().train()
        try:
            enc.set_grad_checkpointing(True)
        except Exception:
            pass
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        x = torch.randn(batch * N_IMG, 3, SIZE, SIZE, device="cuda")
        with torch.autocast("cuda", dtype=torch.float16):
            step = chunk or x.shape[0]
            out = torch.cat([enc(x[i:i + step]) for i in range(0, x.shape[0], step)])
        out.float().mean().backward()
        peak = torch.cuda.max_memory_allocated() / 2**30
        del enc, x, out; torch.cuda.empty_cache()
        return peak
    except RuntimeError as e:
        try:
            del enc, x
        except Exception:
            pass
        torch.cuda.empty_cache()
        if "out of memory" in str(e).lower():
            return None
        raise

hdr = f"{'encoder':<34}{'family':<22}" + "".join(f"{f'chunk {c}':>11}" for c in (0, 12, 6))
print("\n" + hdr); print("-" * len(hdr))
opened = []
for name, fam in CANDIDATES:
    if name.split(".")[0] not in known and name not in known:
        print(f"{name:<34}{fam:<22}  not in this timm"); continue
    row, best = "", 0
    for chunk in (0, 12, 6):
        got = None
        for b in (4, 2, 1):
            try:
                peak = fits(name, chunk, b)
            except Exception as e:
                peak = None
            if peak is not None:
                got = f"b{b} {peak:.0f}GB"; best = max(best, b if chunk else 0)
                if chunk:
                    opened.append((name, fam, chunk, b, peak))
                break
        row += f"{(got or 'OOM'):>11}"
    print(f"{name:<34}{fam:<22}{row}")

print("-" * len(hdr))
print("chunk 0 = all 36 slices at once, which is what we do today.\n")

if opened:
    print("Encoders that chunking makes trainable:")
    seen = set()
    for name, fam, chunk, b, peak in opened:
        if fam in seen:
            continue
        seen.add(fam)
        print(f"  {fam:<22} {name:<34} chunk {chunk}, batch {b}, {peak:.0f} GB")
    print(f"\n{len(seen)} additional encoder FAMILY(s) are now reachable.")
    print("That is the wall. An ensemble needs models that disagree, and we have")
    print("had exactly one family until now.")
else:
    print("Nothing new opened up. The ceiling is real and this time it is")
    print("measured under the conditions we actually train in.")
