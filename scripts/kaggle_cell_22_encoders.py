# ============================================================
# RSNA Knee — encoder bake-off, second attempt
#
# The first attempt tested one encoder out of three and told us almost nothing.
# Three faults, all mine:
#
#   1. CoAtNet printed NOTHING -- no result, no error. The probe only printed on
#      success or on a non-OOM exception, so an encoder that ran out of memory at
#      every batch size from 8 down to 1 was dropped in silence. That was the
#      actual finding and it was invisible.
#   2. The SwinV2 name was not a real timm model.
#   3. ConvNeXt-base fit only at batch 1 -- 3,479 steps an epoch, 203 minutes --
#      and scored 0.791 against DINOv2's 0.804. At batch 1 that is not a verdict
#      on the encoder, it is a verdict on the batch size.
#
# The cause of all the memory pressure is structural: every slice of a study goes
# through the encoder, so one study is 4 slots x 9 slices = 36 IMAGES. "Batch 8"
# means 288 images per step. DINOv2-small (22M) survives that; a base-size CNN
# cannot, at any batch size. Gradient checkpointing recomputes activations instead
# of storing them -- about 30% slower per step, and the difference between "this
# encoder is untestable" and "this encoder is testable".
#
# Attach: competition, cache-v3, stevenleehans labels. GPU, Internet. ~3h.
# ============================================================
!pip install -q timm transformers

import sys, os, time
CODE = "/kaggle/working/rsna-knee"
!rm -rf $CODE && git clone -q https://github.com/IamWasee/rsna-knee.git $CODE
sys.path.insert(0, f"{CODE}/src")
from kaggle_paths import find, describe

v3 = [f for f in find(suffix=".npy") if "cache_v3" in f]
if not v3:
    describe(); raise SystemExit("attach the cache-v3 notebook output")
CACHE = os.path.dirname(v3[0])
lab = find(filename="llm_labels_v4_blend.csv")
if not lab:
    describe(); raise SystemExit("attach stevenleehans/rsna-knee-llm-report-labels")
print(f"cache: {CACHE}\nmeta:  {os.path.exists(CACHE + '/study_meta.csv')}")

import torch, timm
SIZE, N_IMG = 288, 36

# Names are guesses against whatever timm version Kaggle ships, so ask it rather
# than trusting the list. Anything that does not exist is reported and skipped.
WANT = ["coatnet_rmlp_2_rw_224", "coatnet_rmlp_1_rw_224", "coatnet_2_rw_224",
        "convnext_small.fb_in22k_ft_in1k", "convnext_tiny.fb_in22k_ft_in1k",
        "swin_small_patch4_window7_224", "maxvit_rmlp_small_rw_224"]
known = set(timm.list_models())
CANDIDATES = [n for n in WANT if n.split(".")[0] in known or n in known]
print(f"\n{len(CANDIDATES)}/{len(WANT)} candidate names exist in timm "
      f"{timm.__version__}: {CANDIDATES}")
for n in WANT:
    if n not in CANDIDATES:
        print(f"  not a model in this timm: {n}")

def probe(name):
    """Largest batch that fits, with checkpointing on.

    Returns 0 and SAYS WHY in every failing case. The first version printed
    nothing on out-of-memory, and the second crashed because its `finally` block
    touched a model the `try` had already deleted. No `del` inside the try now,
    and no cleanup that references anything conditionally bound.
    """
    m = None
    # coatnet_rmlp bakes its relative-position table to the training resolution:
    # at 288 it asks for 18x18 and finds 14x14. timm can rebuild the table if the
    # model takes img_size, so offer it and fall back for models that do not.
    for kwargs in ({"img_size": SIZE}, {}):
        try:
            m = timm.create_model(name, pretrained=True, num_classes=0,
                                  global_pool="", in_chans=3, **kwargs)
            if kwargs:
                print(f"  (built {name} with img_size={SIZE})")
            break
        except TypeError:
            continue
        except Exception as e:
            print(f"  SKIP {name}: build failed -- {type(e).__name__}: {str(e)[:100]}")
            return 0
    if m is None:
        print(f"  SKIP {name}: could not be built at any setting")
        return 0

    m = m.cuda().train()
    try:
        m.set_grad_checkpointing(True); ckpt = "checkpointed"
    except Exception:
        ckpt = "NO checkpointing"

    fitted, last = 0, ""
    for b in (8, 6, 4, 3, 2, 1):
        try:
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            x = torch.randn(b * N_IMG, 3, SIZE, SIZE, device="cuda")
            y = m(x)
            y.float().mean().backward()
            print(f"  OK   {name}: batch {b} ({b*N_IMG} images), {ckpt}, peak "
                  f"{torch.cuda.max_memory_allocated()/2**30:.1f} GB, "
                  f"map {tuple(y.shape)}")
            fitted = b
        except RuntimeError as e:
            last = str(e)[:110]
            if "out of memory" not in last.lower():
                print(f"  SKIP {name}: {last}")
                break
        for prm in m.parameters():
            prm.grad = None
        if fitted:
            break
    if not fitted and last and "out of memory" in last.lower():
        print(f"  SKIP {name}: out of memory even at batch 1 ({ckpt})")
    del m
    torch.cuda.empty_cache()
    return fitted


print("\nprobing:")
viable = []
for n in CANDIDATES:
    # Two probe bugs in two attempts have each cost a run, so one bad candidate
    # is not allowed to end the cell.
    try:
        b = probe(n)
    except Exception as e:
        print(f"  SKIP {n}: probe raised {type(e).__name__}: {str(e)[:100]}")
        b = 0
    if b >= 2:
        viable.append((n, b))
if not viable:
    raise SystemExit("nothing fits at batch 2 or more, even checkpointed -- the "
                     "36-images-per-study geometry is the constraint, not the encoder")

# Two arms only. Three would not finish inside the session at these batch sizes.
viable = sorted(viable, key=lambda t: -t[1])[:2]
print(f"\ntraining: {viable}")

t0 = time.time()
for name, batch in viable:
    tag = name.split(".")[0].replace("_", "-")[:24]
    print("\n" + "=" * 64 + f"\n{name}  (batch {batch}, checkpointed)\n" + "=" * 64)
    !python $CODE/src/train.py --cache "$CACHE" --labels "{lab[0]}" \
        --backbone "{name}" --size 288 --only-fold 0 --epochs 10 --batch {batch} \
        --lr 1e-3 --lr-backbone 5e-5 --weight-decay 0.02 --grad-checkpoint \
        --head slot --pool focal --out /kaggle/working/enc_{tag}
    print(f"elapsed {(time.time()-t0)/60:.0f} min")

print("\n" + "=" * 64)
print("Baselines on this same cache, fold 0, slot+focal head:")
print("  DINOv2-small, batch 8 ......... 0.804")
print("  ConvNeXt-base, batch 1 ........ 0.791  (batch-1 handicap, not a verdict)")
