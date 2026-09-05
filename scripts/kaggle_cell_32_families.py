# ============================================================
# RSNA Knee — five families at fold 0, gated on time, then picked BY BLEND
#
# CoAtNet is not retrained here: it already ran at fold 0 and scored 0.793
# against DINOv2's 0.804. Its oof.csv is attached instead, so all six families
# take part in the selection without paying for it twice. It cost 249 minutes --
# 11x DINOv2 per epoch -- which is why every arm below is rehearsed and gated
# before it is allowed to train.
#
# THE GATE. train.py --dry-run measures seconds per step on the real loop and
# projects the run; --max-minutes turns that projection into a refusal. The
# CoAtNet run printed "454 min/fold" and trained anyway, overdrawing the week's
# allowance by 2.7 hours. A measurement nothing acts on is not a safeguard.
#
# THE SELECTION. Greedy forward on the blend, not a ranking by score. The last
# bake-off ranked by one criterion and picked ConvNeXt-tiny and ConvNeXt-small --
# the two most similar models on the list. An ensemble pays only when its members
# disagree; the frontier notebook measured seven encoders and dropped four as
# redundant rather than weak.
#
# Attach: competition, cache-v3, stevenleehans labels, dinov2 model,
#         AND the arm-coatnet notebook output (for its oof.csv).
# GPU. Budget below is enforced, not advisory.
# ============================================================
!pip install -q timm transformers

import sys, os, time, glob, subprocess, re
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
CACHE, LABELS_CSV, DINO = os.path.dirname(v3[0]), lab[0], dino[0]

EPOCHS = 8               # matches what CoAtNet got, so the numbers compare
# 110 min at fold 0. The first run refused all five at 75, but its projections
# were inflated by cold-start steps -- CoAtNet was projected at 454 min/fold and
# ran 249. The rehearsal now discards warm-up steps, so these numbers are steady
# state and the budget is set against real ones.
BUDGET_PER_ARM = 110
ARMS = [
    ("dinov2",    f"dinov2:{DINO}",                    8, 0, "plain ViT (incumbent)"),
    ("swin",      "swin_small_patch4_window7_224",     4, 0, "windowed transformer"),
    ("convnext",  "convnext_small.fb_in22k_ft_in1k",   4, 0, "modern convolution"),
    ("seresnext", "seresnext50_32x4d.racm_in1k",       8, 0, "classic convolution"),
    ("maxvit",    "maxvit_rmlp_small_rw_224",          4, 6, "hybrid maxvit"),
]

def run(args, capture=False):
    cmd = [sys.executable, f"{CODE}/src/train.py",
           "--cache", CACHE, "--labels", LABELS_CSV, "--size", "288",
           "--only-fold", "0", "--head", "slot", "--pool", "focal"] + args
    r = subprocess.run(cmd, capture_output=capture, text=True)
    if capture:
        return r.returncode, (r.stdout or "") + (r.stderr or "")
    return r.returncode, ""

print("=" * 72 + "\nREHEARSAL -- real loop, real precision, gated\n" + "=" * 72)
viable = []
for tag, bb, batch, chunk, fam in ARMS:
    code, out = run(["--backbone", bb, "--batch", str(batch),
                     "--encoder-chunk", str(chunk), "--grad-checkpoint",
                     "--epochs", str(EPOCHS), "--dry-run", "4",
                     "--max-minutes", str(BUDGET_PER_ARM),
                     "--out", "/kaggle/working/rehearse"], capture=True)
    line = next((l.strip() for l in out.splitlines() if "s/step" in l), "")
    mem = next((l.strip() for l in out.splitlines() if "peak memory" in l), "")
    print(f"\n{tag:<11}{fam}")
    print(f"  {mem}\n  {line}")
    if code == 0:
        viable.append((tag, bb, batch, chunk, fam))
    else:
        why = next((l for l in out.splitlines() if "REFUSED" in l or "Error" in l), "")
        print(f"  SKIPPED -- {why.strip() or 'rehearsal exit ' + str(code)}")

if not viable:
    raise SystemExit(f"no arm fits {BUDGET_PER_ARM} min. Lower --epochs and retry.")
print(f"\n{len(viable)}/{len(ARMS)} arms fit the budget: {[t for t,*_ in viable]}")

print("\n" + "=" * 72 + "\nTRAINING\n" + "=" * 72)
t0 = time.time()
for tag, bb, batch, chunk, fam in viable:
    print("\n" + "-" * 72 + f"\n{tag}: {fam}\n" + "-" * 72, flush=True)
    run(["--backbone", bb, "--batch", str(batch), "--encoder-chunk", str(chunk),
         "--grad-checkpoint", "--epochs", str(EPOCHS),
         "--lr", "1e-3", "--lr-backbone", "5e-5", "--weight-decay", "0.02",
         "--out", f"/kaggle/working/arm_{tag}"])
    print(f"elapsed {(time.time()-t0)/60:.0f} min", flush=True)

# ---------------------------------------------------------------- selection
paths = {}
for tag, *_ in viable:
    p = f"/kaggle/working/arm_{tag}/oof.csv"
    if os.path.exists(p):
        paths[tag] = p
for p in find(filename="oof.csv"):
    if "coatnet" in p and "coatnet" not in paths:
        paths["coatnet"] = p           # the arm already paid for, folded back in

if len(paths) < 2:
    raise SystemExit(f"only {len(paths)} arm(s) produced oof.csv; nothing to select")
print(f"\nselecting over {len(paths)} families: {list(paths)}")

from sklearn.metrics import roc_auc_score
base = pd.read_csv(paths[list(paths)[0]])
ID = base.columns[0]
Y = {c: pd.to_numeric(base[f"{c}__y"], errors="coerce").values for c in LABELS}
R = {}
for tag, p in paths.items():
    df = pd.read_csv(p).set_index(ID).reindex(base[ID]).reset_index()
    # Rank per label: AUC reads order only, and two models are not calibrated to
    # each other, so a probability mean lets the more confident one dominate.
    R[tag] = {c: pd.Series(df[c].values).rank(pct=True).values for c in LABELS}

def macro(tags):
    out = []
    for c in LABELS:
        y = Y[c]; keep = ~np.isnan(y)
        p = np.mean([R[t][c] for t in tags], axis=0)
        tr = (y[keep] > 0.5).astype(int)
        if len(set(tr)) > 1:
            out.append(roc_auc_score(tr, p[keep]))
    return float(np.mean(out))

print("\n" + "=" * 72 + "\nEACH FAMILY ALONE\n" + "=" * 72)
singles = sorted(((macro([t]), t) for t in paths), reverse=True)
for s, t in singles:
    print(f"  {t:<12}{s:.4f}")

print("\n" + "=" * 72 + "\nAGREEMENT (1.000 = same mistakes, useless together)\n" + "=" * 72)
tags = [t for _, t in singles]
print(f"{'':<12}" + "".join(f"{t[:9]:>11}" for t in tags))
for a in tags:
    print(f"{a:<12}" + "".join(
        f"{np.mean([np.corrcoef(R[a][c], R[b][c])[0,1] for c in LABELS]):>11.3f}"
        for b in tags))

print("\n" + "=" * 72 + "\nGREEDY SELECTION ON THE BLEND\n" + "=" * 72)
chosen, rest = [tags[0]], list(tags[1:])
print(f"  1. {tags[0]:<12} alone {macro(chosen):.4f}")
while rest:
    prev = macro(chosen)
    gain, t = max((macro(chosen + [x]), x) for x in rest)
    star = "  <- beats the next-best-alone" if t != rest[0] else ""
    print(f"  {len(chosen)+1}. + {t:<10} {gain:.4f}  ({gain - prev:+.4f}){star}")
    chosen.append(t); rest.remove(t)
print("\nKeep the prefix where the gain stops paying. Where greedy order differs")
print("from the ranking above, that difference IS the reason to select this way.")
