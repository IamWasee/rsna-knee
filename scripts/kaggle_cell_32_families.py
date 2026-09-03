# ============================================================
# RSNA Knee — six encoder families at one fold, then pick three BY BLEND
#
# The memory probe settled the question that blocked this: every family below
# fits on a T4, CoAtNet at 6 of 15 GB without chunking. The earlier "cannot fit
# on this hardware" was measured at full precision on 36 images per study where
# training uses half precision on 12 -- about six times the real requirement.
#
# WHY NOT PICK THE TOP THREE BY SCORE. Last bake-off I ranked candidates by one
# criterion and trained the top two, which selected ConvNeXt-tiny and
# ConvNeXt-small -- the two most similar models on the list. An ensemble is worth
# something only when its members DISAGREE; ranking by quality selects for
# agreement. The public frontier notebook measured seven encoders and dropped
# four as redundant, not as weak.
#
# So: train all six at fold 0, then greedy forward selection on the BLEND. Start
# with the best single model, then repeatedly add whichever remaining model most
# improves the blended macro AUC -- which is a different question from which is
# next-best alone, and is the one that decides the final ensemble.
#
# Reference: DINOv2-small on this cache, fold 0, slot+focal = 0.804. Fold 0 reads
# about +0.01 optimistic against the 5-fold pooled number, so compare within this
# run, not against 0.797.
#
# Attach: competition, cache-v3, stevenleehans labels, dinov2. GPU. ~5h.
# ============================================================
!pip install -q timm transformers

import sys, os, time, glob, itertools
import numpy as np, pandas as pd

CODE = "/kaggle/working/rsna-knee"
!rm -rf $CODE && git clone -q https://github.com/IamWasee/rsna-knee.git $CODE
sys.path.insert(0, f"{CODE}/src")
from kaggle_paths import find, describe
from config import LABELS

v3 = [f for f in find(suffix=".npy") if "cache_v3" in f]
lab = find(filename="llm_labels_v4_blend.csv")
dino = [os.path.dirname(p) for p in find(filename="config.json") if "dinov2" in p.lower()]
if not (v3 and lab and dino):
    describe(); raise SystemExit("attach cache-v3, stevenleehans labels, dinov2")
CACHE = os.path.dirname(v3[0])

# The incumbent plus one representative per family. Batch and chunk come from the
# memory probe, which measured them under autocast fp16 exactly as trained.
ARMS = [
    ("dinov2",     f"dinov2:{dino[0]}",                   4, 0,  "plain ViT (incumbent)"),
    ("coatnet",    "coatnet_rmlp_2_rw_224",               4, 6,  "hybrid conv+attention"),
    ("maxvit",     "maxvit_rmlp_small_rw_224",            4, 6,  "hybrid maxvit"),
    ("swin",       "swin_small_patch4_window7_224",       4, 0,  "windowed transformer"),
    ("convnext",   "convnext_small.fb_in22k_ft_in1k",     4, 0,  "modern convolution"),
    ("seresnext",  "seresnext50_32x4d.racm_in1k",         4, 0,  "classic convolution"),
]

# Rehearse each before committing five hours: real loop, real data, real
# precision, three steps. A hand-written probe is what produced every wrong
# memory number so far, so there is no hand-written probe here.
print("=" * 70 + "\nREHEARSAL -- real loop, 3 steps each\n" + "=" * 70)
for tag, bb, batch, chunk, fam in ARMS:
    print(f"\n{tag} ({fam})")
    !python $CODE/src/train.py --cache "$CACHE" --labels "{lab[0]}" \
        --backbone "{bb}" --size 288 --only-fold 0 --batch {batch} \
        --encoder-chunk {chunk} --grad-checkpoint --dry-run 3 \
        --head slot --pool focal --out /kaggle/working/rehearse 2>&1 | tail -6

print("\n" + "=" * 70 + "\nTRAINING\n" + "=" * 70)
t0, trained = time.time(), []
for tag, bb, batch, chunk, fam in ARMS:
    out = f"/kaggle/working/arm_{tag}"
    print("\n" + "-" * 70 + f"\n{tag}: {fam}\n" + "-" * 70)
    !python $CODE/src/train.py --cache "$CACHE" --labels "{lab[0]}" \
        --backbone "{bb}" --size 288 --only-fold 0 --epochs 10 --batch {batch} \
        --encoder-chunk {chunk} --grad-checkpoint \
        --lr 1e-3 --lr-backbone 5e-5 --weight-decay 0.02 \
        --head slot --pool focal --out $out
    if os.path.exists(f"{out}/oof.csv"):
        trained.append((tag, fam, f"{out}/oof.csv"))
    else:
        print(f"  {tag} produced no oof.csv -- excluded")
    print(f"elapsed {(time.time()-t0)/60:.0f} min")

if len(trained) < 2:
    raise SystemExit("fewer than two arms trained; nothing to select between")

# ---------------------------------------------------------------- selection
from sklearn.metrics import roc_auc_score
ID = pd.read_csv(trained[0][2]).columns[0]
base = pd.read_csv(trained[0][2])
Y = {c: pd.to_numeric(base[f"{c}__y"], errors="coerce").values for c in LABELS}

R = {}
for tag, fam, path in trained:
    df = pd.read_csv(path).set_index(ID).reindex(base[ID]).reset_index()
    # Rank per label: AUC reads only order, and two models are not calibrated to
    # each other, so averaging probabilities lets the more confident one dominate.
    R[tag] = {c: pd.Series(df[c].values).rank(pct=True).values for c in LABELS}

def macro(tags):
    out = []
    for c in LABELS:
        y = Y[c]; keep = ~np.isnan(y)
        p = np.mean([R[t][c] for t in tags], axis=0)
        t = (y[keep] > 0.5).astype(int)
        if len(set(t)) > 1:
            out.append(roc_auc_score(t, p[keep]))
    return float(np.mean(out))

print("\n" + "=" * 70 + "\nEACH FAMILY ALONE\n" + "=" * 70)
singles = sorted(((macro([t]), t, f) for t, f, _ in trained), reverse=True)
for s, t, f in singles:
    print(f"  {t:<12}{f:<26}{s:.4f}")

print("\n" + "=" * 70 + "\nHOW MUCH DO THEY DISAGREE?\n" + "=" * 70)
tags = [t for _, t, _ in singles]
print(f"{'':<12}" + "".join(f"{t[:9]:>11}" for t in tags))
for a in tags:
    row = "".join(
        f"{np.mean([np.corrcoef(R[a][c], R[b][c])[0,1] for c in LABELS]):>11.3f}"
        for b in tags)
    print(f"{a:<12}{row}")
print("Lower is better for an ensemble: 1.000 means the two make the same mistakes.")

print("\n" + "=" * 70 + "\nGREEDY SELECTION ON THE BLEND\n" + "=" * 70)
chosen, remaining = [tags[0]], [t for t in tags[1:]]
print(f"  1. {tags[0]:<12} alone {macro(chosen):.4f}")
while remaining:
    gains = sorted(((macro(chosen + [t]), t) for t in remaining), reverse=True)
    best, t = gains[0]
    print(f"  {len(chosen)+1}. + {t:<10} blend {best:.4f}  "
          f"({best - macro(chosen):+.4f})"
          + ("   <- next best ALONE is " + gains[0][1] if False else ""))
    chosen.append(t); remaining.remove(t)

print("\n" + "=" * 70)
print("Take the prefix where the gain stops paying -- typically three. Note where")
print("greedy order differs from the ranking by score alone: a model that adds")
print("more to the blend than a better model does is exactly what we are after.")
