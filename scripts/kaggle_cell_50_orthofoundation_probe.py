# ============================================================
# RSNA Knee — OrthoFoundation-L as a FROZEN encoder: does it beat our DINOv2 arms?
# GPU (2x T4), internet ON (git clone of our code and facebookresearch/dinov3).
#
# OrthoFoundation (github.com/ytrsk/OrthoFoundation, MIT) is DINOv3 ViT-L/16
# further pretrained on 1.25 M knee images (894 k MRI slices, OAI + fastMRI +
# a hospital cohort). Public weights; our copy is the snap-orthofoundation
# notebook's output, so it stays usable if the repo is taken down.
#
# 1. Every competition study, every plane: the 24 cached slices through the
#    frozen encoder -> per slice [CLS, mean patch token] (2048 numbers).
# 2. Two cheap heads on those features, our own 5 grouped folds and labels
#    (train.build_labels, pseudo50 + --sharpen-to none, as the combo arms):
#      ridge  -- mean+max over slices, all three planes, linear on soft targets
#      attn   -- per plane a gated-attention pool over the 24 slices, then one
#                linear layer on the three planes together; soft BCE
# 3. Gold-58: each fold's model predicts the 58, folds rank-averaged -- the same
#    reading as our arms' "folds rank-averaged" numbers.
#
# Reference (cell 45 / cell 44 logs): sag-combo alone 0.8985, cor-combo single
# epochs 0.8731 mean per fold, our 9-arm team 0.9003 on gold.
# Writes features (for a later submission leg) and orthofoundation_probe.json.
# ============================================================
!pip install -q timm omegaconf termcolor ftfy
import sys, os, json, time, glob, subprocess
from pathlib import Path
CODE = "/kaggle/working/rsna-knee"
!rm -rf $CODE && git clone -q https://github.com/IamWasee/rsna-knee.git $CODE
!git -C $CODE log --oneline -1
sys.path.insert(0, f"{CODE}/src")
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from kaggle_paths import find, describe
from config import ID_COL, LABELS
import train as T

t00 = time.time()
# ---- inputs ------------------------------------------------------------
caches = {}
for m in find(filename="cache_manifest.json"):
    d = os.path.dirname(m)
    man = json.load(open(m))
    for p in ("sag", "cor", "ax"):
        if (d.endswith(f"cache_{p}") and man.get("split") not in ("fastmri", "oai")
                and man.get("slots") == 1 and p not in caches):
            caches[p] = d
lab = find(filename="labels_pseudo50.csv")
wts = find(filename="OrthoFoundation-L.pth")
if sorted(caches) != ["ax", "cor", "sag"] or not lab or not wts:
    describe(); raise SystemExit(f"need cache_{{sag,cor,ax}}, labels_pseudo50.csv, OrthoFoundation-L.pth: "
                                 f"{sorted(caches)} {lab} {wts}")
print("caches", caches, "\nlabels", lab[0], "\nweights", wts[0])

args = T.make_parser().parse_args(["--cache", caches["sag"], "--labels", lab[0], "--sharpen-to", "none"])
derived, gold = T.build_labels(args)
from sklearn.model_selection import GroupKFold
folds = list(GroupKFold(5).split(derived, groups=derived["_group"]))
ids = list(derived[ID_COL]) + list(gold[ID_COL])
print(f"{len(derived)} labelled studies, {len(gold)} gold, {(time.time()-t00)/60:.1f} min")

# ---- encoder -----------------------------------------------------------
REPO = "/kaggle/working/dinov3"
!rm -rf $REPO && git clone -q https://github.com/facebookresearch/dinov3.git $REPO && git -C $REPO checkout -q 6876159
sys.path.insert(0, REPO)
from dinov3.hub.backbones import dinov3_vitl16      # not hubconf: it imports the detection/segmentation stacks
enc = dinov3_vitl16(pretrained=False)
ck = torch.load(wts[0], map_location="cpu", weights_only=False)
print("checkpoint top-level keys:", list(ck)[:8] if isinstance(ck, dict) else type(ck))
want = set(enc.state_dict())


def candidates(obj, path=""):
    """Every nested dict of tensors in the checkpoint, with its path."""
    if isinstance(obj, dict):
        if obj and all(torch.is_tensor(v) for v in obj.values()):
            yield path, obj
        for k, v in obj.items():
            if isinstance(v, dict):
                yield from candidates(v, f"{path}/{k}")


best, best_n = None, -1
for path, sd in candidates(ck):
    for pre in ("", "backbone.", "module.", "module.backbone.", "teacher.backbone.", "student.backbone.",
                "teacher.", "student.", "encoder."):
        m = {k[len(pre):]: v for k, v in sd.items() if k.startswith(pre)}
        n = len(want & set(m))
        if n > best_n:
            best, best_n, how = m, n, f"{path or '/'} prefix '{pre}'"
print(f"best match: {how}: {best_n}/{len(want)} encoder keys")
if best_n < 0.95 * len(want):
    print("unmatched examples:", sorted(want - set(best))[:10])
    print("checkpoint examples:", sorted(best)[:10])
    raise SystemExit("OrthoFoundation weights do not map onto dinov3_vitl16")
missing, unexpected = enc.load_state_dict({k: v for k, v in best.items() if k in want}, strict=False)
params = {n for n, _ in enc.named_parameters()}
print(f"loaded; missing {len(missing)} {missing[:8]}")
if set(missing) & params:
    raise SystemExit(f"{len(set(missing) & params)} encoder PARAMETERS left at random init: "
                     f"{sorted(set(missing) & params)[:10]}")
left = sorted(k for k in best if k not in want and not k.startswith(("head", "dino_head", "ibot_head")))
print(f"checkpoint keys not used: {len(left)} {left[:8]}")
if any(k.startswith(("blocks.", "patch_embed", "norm", "cls_token", "storage_tokens", "rope")) for k in left):
    raise SystemExit("backbone-looking checkpoint keys were not loaded")
# fp32 weights + fp16 autocast: LayerNorm/softmax stay fp32 and the RoPE periods are not rounded
enc = enc.float().eval().cuda()
ngpu = torch.cuda.device_count()

SIZE = 256          # 16 x 16 patches; the cache is 288 px
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).cuda()
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).cuda()
for p, d in caches.items():
    m = json.load(open(f"{d}/cache_manifest.json"))
    assert m["size"] == 288 and m["n_slices"] == 24, (p, m)


class Wrap(nn.Module):
    def __init__(self, e):
        super().__init__(); self.e = e

    def forward(self, x):
        o = self.e.forward_features(x)
        return torch.cat([o["x_norm_clstoken"], o["x_norm_patchtokens"].mean(1)], 1)


net = Wrap(enc)
net = nn.DataParallel(net) if ngpu > 1 else net

# ---- features ----------------------------------------------------------
from concurrent.futures import ThreadPoolExecutor
FEAT = Path("/kaggle/working/of_feats"); FEAT.mkdir(exist_ok=True)
feats = {}
for p in ("sag", "cor", "ax"):
    t0 = time.time()
    out = np.zeros((len(ids), 24, 2048), np.float16)
    have = np.zeros(len(ids), bool)
    load = lambda i: np.load(f"{caches[p]}/{i}.npy") if os.path.exists(f"{caches[p]}/{i}.npy") else None
    B = 8   # studies per step = 192 slices
    with ThreadPoolExecutor(8) as ex:
        for s in range(0, len(ids), B):
            vols = list(ex.map(load, ids[s:s + B]))
            ok = [k for k, v in enumerate(vols) if v is not None]
            if not ok:
                continue
            x = torch.from_numpy(np.stack([vols[k][0] for k in ok])).cuda()       # b,24,288,288
            x = x.view(-1, 1, 288, 288).float().div(255)
            x = F.interpolate(x, size=SIZE, mode="bilinear", align_corners=False).expand(-1, 3, -1, -1)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                f = net((x - MEAN) / STD).float().view(len(ok), 24, -1)
            assert torch.isfinite(f).all(), f"non-finite features, {p} batch at {s}"
            for j, k in enumerate(ok):
                out[s + k] = f[j].cpu().numpy(); have[s + k] = True
            if s % 800 == 0:
                print(f"  {p} {s}/{len(ids)}  {(time.time()-t0)/60:.1f} min", flush=True)
    np.save(FEAT / f"{p}.npy", out)
    feats[p] = out
    print(f"{p}: {have.sum()}/{len(ids)} studies, {(time.time()-t0)/60:.1f} min", flush=True)
pd.Series(ids).to_csv(FEAT / "ids.csv", index=False, header=[ID_COL])

nD = len(derived)
Y = derived[LABELS].values.astype(np.float32)
G = gold[LABELS].values.astype(np.float32)


def gold_auc(pred_by_fold):
    r = np.mean([pd.DataFrame(p).rank(pct=True).values for p in pred_by_fold], 0)
    return T.macro_auc(G, r)

res = {}
# ---- head 1: ridge on pooled features ----------------------------------
from sklearn.linear_model import Ridge
pool = np.concatenate([np.concatenate([feats[p].astype(np.float32).mean(1), feats[p].astype(np.float32).max(1)], 1)
                       for p in ("sag", "cor", "ax")], 1)
mu, sd = pool[:nD].mean(0), pool[:nD].std(0) + 1e-6
pool = (pool - mu) / sd
for alpha in (300.0, 3000.0, 30000.0):
    oof = np.zeros_like(Y); gp = []
    for ti, vi in folds:
        m = Ridge(alpha=alpha).fit(pool[ti], Y[ti])
        oof[vi] = m.predict(pool[vi]); gp.append(m.predict(pool[nD:]))
    o, _ = T.macro_auc((Y > 0.5).astype(int), oof)
    g, gpl = gold_auc(gp)
    res[f"ridge_a{int(alpha)}"] = {"oof": o, "gold": g, "gold_per_label": gpl}
    print(f"ridge alpha {alpha:>7.0f}: OOF {o:.4f}  gold {g:.4f}", flush=True)

# ---- head 2: gated attention over slices, three planes -----------------
X = {p: torch.from_numpy(feats[p]) for p in ("sag", "cor", "ax")}


class Head(nn.Module):
    def __init__(self, d=2048, h=384):
        super().__init__()
        self.proj = nn.ModuleDict({p: nn.Sequential(nn.LayerNorm(d), nn.Linear(d, h), nn.GELU(), nn.Dropout(0.2))
                                   for p in X})
        self.att = nn.ModuleDict({p: nn.Linear(h, 12) for p in X})     # per-finding slice attention
        self.out = nn.Linear(3 * h, 12)

    def forward(self, xs):
        zs = []
        for p in ("sag", "cor", "ax"):
            h = self.proj[p](xs[p].float())                  # b,24,h
            a = self.att[p](h).softmax(1)                    # b,24,12
            zs.append(torch.einsum("bsh,bsk->bkh", h, a))    # b,12,h
        z = torch.cat(zs, -1)                                # b,12,3h
        return (z * self.out.weight).sum(-1) + self.out.bias


def run_head(ti, vi, seed, epochs=25):
    torch.manual_seed(seed)
    m = Head().cuda()
    opt = torch.optim.AdamW(m.parameters(), lr=3e-4, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-4, total_steps=epochs * (len(ti) // 32 + 1))
    yt = torch.from_numpy(Y)
    for ep in range(epochs):
        m.train()
        for b in np.array_split(np.random.permutation(ti), len(ti) // 32 + 1):
            xs = {p: X[p][b].cuda() for p in X}
            if True:  # slice dropout: hide a random 25% of slices
                keep = (torch.rand(len(b), 24, 1, device="cuda") > 0.25)
                xs = {p: v * keep for p, v in xs.items()}
            loss = F.binary_cross_entropy_with_logits(m(xs), yt[b].cuda())
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    m.eval()

    def pred(idx):
        with torch.no_grad():
            return torch.cat([m({p: X[p][c].cuda() for p in X}).sigmoid().cpu()
                              for c in np.array_split(idx, max(1, len(idx) // 256))]).numpy()
    return pred(vi), pred(np.arange(nD, len(ids)))


oof = np.zeros_like(Y); gp = []
for k, (ti, vi) in enumerate(folds):
    o1, g1 = run_head(ti, vi, 42 + k)
    oof[vi] = o1; gp.append(g1)
o, _ = T.macro_auc((Y > 0.5).astype(int), oof)
g, gpl = gold_auc(gp)
res["attn"] = {"oof": o, "gold": g, "gold_per_label": gpl}
print(f"attention head: OOF {o:.4f}  gold {g:.4f}", flush=True)

print("\n" + "=" * 66)
print("OrthoFoundation-L, frozen, on gold-58 (folds rank-averaged):")
for k, v in res.items():
    print(f"  {k:<14} gold {v['gold']:.4f}   OOF {v['oof']:.4f}")
print("ours for reference: sag-combo arm 0.8985 | 9-arm team 0.9003")
bestk = max(res, key=lambda k: res[k]["gold"])
print(f"\nper finding, {bestk}:")
for c, a in res[bestk]["gold_per_label"].items():
    print(f"  {c:<18} {a:.3f}")
json.dump(res, open("/kaggle/working/orthofoundation_probe.json", "w"), indent=1, default=float)
print(f"\ntotal {(time.time()-t00)/60:.0f} min")
