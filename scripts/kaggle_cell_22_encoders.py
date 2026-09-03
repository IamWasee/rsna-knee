# ============================================================
# RSNA Knee — encoder bake-off, third attempt. Measured the way we TRAIN.
#
# The last two attempts each failed for a different reason of mine, and the second
# failure produced a wrong conclusion I acted on:
#
#   1st: an encoder that ran out of memory at every batch size printed NOTHING.
#        The silence was the finding.
#   2nd: the probe measured FULL precision while training runs at HALF. So it
#        reported roughly twice the real requirement and I concluded from that
#        that a large encoder "cannot fit on this hardware". That conclusion is
#        not supported -- it came from a test that did not match the setup.
#
# What is different now:
#   - autocast float16, matching the training loop exactly
#   - --encoder-chunk: peak memory scales with SLICES PER STUDY (36), not batch
#     size, which is the real constraint. Feeding the encoder 12 at a time caps
#     the peak at 12 images regardless of batch. This is the lever that decides
#     whether a large encoder runs at all.
#   - candidates chosen for ARCHITECTURAL DIVERSITY, and one picked per family.
#     Last time I ranked by whichever fit at the largest batch and trained the top
#     two -- which selected ConvNeXt-tiny and ConvNeXt-small, the two most similar
#     models on the list, and skipped every hybrid and transformer. Ensembles are
#     built from models that DISAGREE; picking for throughput selected for
#     agreement.
#
# Why this matters more than anything else queued: every team above 0.92 runs four
# or five different encoder families and averages them. We have one. Our "blend"
# is two heads on the same backbone, which is why it bought only +0.008.
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

import torch, timm
print(f"torch {torch.__version__}, timm {timm.__version__}, "
      f"{torch.cuda.device_count()} GPU(s), "
      f"{torch.cuda.get_device_properties(0).total_memory/2**30:.0f} GB each")
SIZE, N_IMG = 288, 36
CHUNK = 12          # slices per encoder pass

# One per architecture family, deliberately. DINOv2-small (our incumbent, 0.804)
# is a plain ViT, so everything here is something else.
FAMILIES = {
    "hybrid conv+attn":  ["coatnet_rmlp_2_rw_224", "coatnet_rmlp_1_rw_224"],
    "hybrid maxvit":     ["maxvit_rmlp_small_rw_224", "maxvit_tiny_rw_224"],
    "windowed transf.":  ["swin_small_patch4_window7_224", "swin_tiny_patch4_window7_224"],
    "modern conv":       ["convnext_small.fb_in22k_ft_in1k"],
    "classic conv":      ["resnet50.a1_in1k", "seresnext50_32x4d.racm_in1k"],
    "efficient conv":    ["tf_efficientnetv2_s.in21k_ft_in1k"],
}
known = set(timm.list_models())

def probe(name):
    """Largest batch that fits, measured under autocast float16 and chunking --
    i.e. the conditions train.py actually runs in. Returns 0 and says why."""
    m = None
    for kw in ({"img_size": SIZE}, {}):
        try:
            m = timm.create_model(name, pretrained=True, num_classes=0,
                                  global_pool="", in_chans=3, **kw)
            break
        except TypeError:
            continue
        except Exception as e:
            print(f"    build failed -- {type(e).__name__}: {str(e)[:90]}")
            return 0
    if m is None:
        return 0
    m = m.cuda().train()
    try:
        m.set_grad_checkpointing(True)
    except Exception:
        pass

    fitted, last = 0, ""
    for b in (8, 6, 4, 2, 1):
        try:
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            x = torch.randn(b * N_IMG, 3, SIZE, SIZE, device="cuda")
            with torch.autocast("cuda", dtype=torch.float16):
                out = torch.cat([m(x[i:i + CHUNK])
                                 for i in range(0, x.shape[0], CHUNK)], dim=0)
            out.float().mean().backward()
            print(f"    batch {b} ({b*N_IMG} slices, {CHUNK}/pass) fits -- "
                  f"peak {torch.cuda.max_memory_allocated()/2**30:.1f} GB")
            fitted = b
        except RuntimeError as e:
            last = str(e)[:100]
            if "out of memory" not in last.lower():
                print(f"    {last}"); break
        for prm in m.parameters():
            prm.grad = None
        if fitted:
            break
    if not fitted and "out of memory" in last.lower():
        print(f"    out of memory even at batch 1 with fp16 + chunk {CHUNK}")
    del m; torch.cuda.empty_cache()
    return fitted

print("\nprobing, one family at a time -- the first that fits wins the slot:\n")
chosen = []
for fam, names in FAMILIES.items():
    print(f"{fam}:")
    for n in names:
        if n.split(".")[0] not in known and n not in known:
            print(f"  {n}: not in this timm"); continue
        print(f"  {n}")
        try:
            b = probe(n)
        except Exception as e:
            print(f"    probe raised {type(e).__name__}: {str(e)[:80]}"); b = 0
        if b:
            chosen.append((fam, n, b)); break
    print()

if not chosen:
    raise SystemExit("nothing fits even at fp16 with chunking -- that WOULD be a "
                     "hardware wall, and this is the test that establishes it")

print("=" * 66)
for fam, n, b in chosen:
    print(f"{fam:<20} {n:<38} batch {b}")
print("=" * 66)

# Train the three most different from a plain ViT: a hybrid, a windowed
# transformer, and a convolutional net. Not the three that fit best.
ORDER = ["hybrid conv+attn", "hybrid maxvit", "windowed transf.",
         "classic conv", "modern conv", "efficient conv"]
picked, seen = [], set()
for want in ORDER:
    for fam, n, b in chosen:
        if fam == want and fam not in seen:
            picked.append((n, b)); seen.add(fam)
    if len(picked) == 3:
        break
print(f"\ntraining: {picked}\n")

t0 = time.time()
for name, batch in picked:
    tag = name.split(".")[0].replace("_", "-")[:24]
    print("\n" + "=" * 66 + f"\n{name}  (batch {batch}, chunk {CHUNK})\n" + "=" * 66)
    !python $CODE/src/train.py --cache "$CACHE" --labels "{lab[0]}" \
        --backbone "{name}" --size 288 --only-fold 0 --epochs 10 --batch {batch} \
        --lr 1e-3 --lr-backbone 5e-5 --weight-decay 0.02 \
        --grad-checkpoint --encoder-chunk {CHUNK} \
        --head slot --pool focal --out /kaggle/working/enc_{tag}
    print(f"elapsed {(time.time()-t0)/60:.0f} min")

print("\n" + "=" * 66)
print("Fold 0 on this cache with the slot+focal head:")
print("  DINOv2-small (incumbent) ..... 0.804")
print("  ConvNeXt-base, batch 1 ....... 0.791   (measured at fp32, handicapped)")
print("\nA second family within ~0.02 of DINOv2 is worth keeping even if it loses:")
print("what an ensemble needs is disagreement, not a winner.")
