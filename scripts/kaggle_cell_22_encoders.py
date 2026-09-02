# ============================================================
# RSNA Knee — encoder bake-off on cache v3, fold 0
#
# Evidence for running this: a public notebook on this exact competition published
# its own encoder panel, measured on the gold studies --
#
#     coatnet_rmlp_2_rw_384  0.9025      convnext_base_384      0.8754
#     swin_base_384          0.8825      convnext_large_384     0.8752
#     convnext_336           0.8833      maxvit_384             0.8438
#     effnetv2_l_480         0.8716
#
# CoAtNet won by 0.02 over the next architecture and by 0.06 over the worst. Our
# DINOv2-small scores 0.804 held-out on this cache, so an encoder gap of that size
# is worth more than anything else on the list.
#
# What this cell does NOT do is change resolution at the same time. Their panel was
# run entirely at 384, so it compares encoders, not sizes -- and 4 slots x 9 slices
# x 384px would be 23 GB against a 20 GB limit, forcing a cut in slice coverage as
# well. Test the encoder first on the cache we already have. If CoAtNet wins here,
# a 384 cache is worth building; if it does not, we saved the rebuild.
#
# Attach: competition, cache-v3, stevenleehans labels. GPU on, Internet on
# (timm downloads the pretrained weights). ~2h.
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

# ---- which candidates can actually be built and run at 288? --------------------
# CoAtNet's -384 weights carry relative-position tables sized for 384, and Swin
# needs the window to divide the feature map. Rather than discover that at epoch 3
# of a 45-minute fold, build each candidate now, push one real batch through it,
# and find the largest batch size that fits. A candidate that fails here is skipped
# with its error printed instead of taking the GPU down with it.
import torch, timm, numpy as np
CANDIDATES = [
    ("coatnet_rmlp_2_rw_384", 5e-5),
    ("convnext_base.fb_in22k_ft_in1k", 5e-5),
    ("swinv2_base_window12to16_192to256_ms_in22k_ft_in1k", 5e-5),
]
SIZE, N_IMG = 288, 36          # 4 slots x 9 slices, all of them go through the encoder

viable = []
for name, lr in CANDIDATES:
    try:
        m = timm.create_model(name, pretrained=True, num_classes=0,
                              global_pool="", in_chans=3).cuda().train()
    except Exception as e:
        print(f"SKIP {name}: build failed -- {type(e).__name__}: {str(e)[:120]}")
        continue
    fitted = 0
    for b in (8, 6, 4, 3, 2, 1):
        try:
            torch.cuda.empty_cache()
            x = torch.randn(b * N_IMG, 3, SIZE, SIZE, device="cuda")
            y = m(x)
            y.float().mean().backward()
            fitted = b
            print(f"OK   {name}: batch {b} fits, feature map {tuple(y.shape)}")
            break
        except RuntimeError as e:
            if "out of memory" not in str(e).lower():
                print(f"SKIP {name}: {str(e)[:140]}"); break
        finally:
            del x
            for p in m.parameters():
                p.grad = None
            torch.cuda.empty_cache()
    if fitted:
        viable.append((name, lr, fitted))
    del m; torch.cuda.empty_cache()

if not viable:
    raise SystemExit("no candidate encoder could be built and run at 288px")
print("\nrunning:", [(n, b) for n, _, b in viable])

# ---- train each on fold 0 ------------------------------------------------------
# Batch size differs per encoder because memory does, so these are not perfectly
# matched runs. An encoder that only fits at batch 2 is carrying that handicap into
# its number -- read a narrow loss as inconclusive, not as a verdict.
t0 = time.time()
for name, lr, batch in viable:
    tag = name.split(".")[0].replace("_", "-")[:24]
    print("\n" + "=" * 64 + f"\n{name}   (batch {batch}, backbone lr {lr})\n" + "=" * 64)
    !python $CODE/src/train.py --cache "$CACHE" --labels "{lab[0]}" \
        --backbone "{name}" --size 288 --only-fold 0 --epochs 12 --batch {batch} \
        --lr 1e-3 --lr-backbone {lr} --weight-decay 0.02 \
        --head slot --pool focal --out /kaggle/working/enc_{tag}
    print(f"elapsed {(time.time()-t0)/60:.0f} min")

print("\n" + "=" * 64)
print("Compare 'fold 0 best OOF macro AUC' across the blocks above.")
print("DINOv2-small on this same cache, same fold, same head: 0.804")
