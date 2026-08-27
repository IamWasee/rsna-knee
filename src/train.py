"""Train the study-level classifier on report-derived labels.

Labels come from report extraction, so they are soft probabilities, not ground
truth. Two consequences shape this file:

  - The 58 gold studies are held out of every fold and used as the only honest
    validation signal. Validating on report-derived labels would measure agreement
    with the extractor, not with radiologists.
  - Soft targets are kept as-is rather than thresholded. A 0.6 from the extractor
    genuinely means "probably", and BCE against 0.6 says exactly that. Rounding it
    to 1.0 would train the model to be confident about something the extractor was
    not.

    python src/train.py --cache cache --labels report_labels.csv --folds 5
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import GOLD_PREVALENCE, ID_COL, LABELS  # noqa: E402
from dataset import KneeStudies  # noqa: E402
from model import KneeModel, build_loss  # noqa: E402
from paths import data_root  # noqa: E402


def macro_auc(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, dict]:
    from sklearn.metrics import roc_auc_score

    per = {}
    for i, c in enumerate(LABELS):
        t = y_true[:, i]
        per[c] = roc_auc_score(t, y_pred[:, i]) if len(set(t)) > 1 else float("nan")
    valid = [v for v in per.values() if v == v]
    return (float(np.mean(valid)) if valid else float("nan")), per


def sharpen(df: pd.DataFrame, k: float = 8.0) -> pd.DataFrame:
    """Spread report-derived labels across [0,1] without changing their ranking.

    The ensemble averages a coarse source (three discrete values) with a continuous
    one, which compresses nearly every target toward 0.5 -- measured means 0.43-0.65
    with std ~0.15. BCE against a target of 0.55 carries almost no gradient, so the
    network is told "maybe" about everything and learns slowly. Observed: loss
    plateaus near ln(2) because that IS the entropy floor of such targets.

    Each label is mapped through its own rank percentile and a logistic centred so
    that the top `gold prevalence` fraction lands above 0.5. This is monotone per
    label, so it leaves label-vs-gold AUC exactly unchanged -- nothing is claimed
    that the extractor did not already say -- while restoring usable gradient.

    Prevalence comes from the 58 gold studies, which look enriched relative to the
    corpus. That only shifts where the curve is centred, not the ordering, so it
    cannot hurt AUC; it makes the positive rate plausible rather than arbitrary.
    """
    out = df.copy()
    for c in LABELS:
        r = df[c].rank(pct=True)
        p = GOLD_PREVALENCE[c]
        out[c] = (1 / (1 + np.exp(-k * (r - (1 - p))))).clip(0.02, 0.98)
    return out


def build_labels(args) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Training frame (report-derived) and gold frame (58 radiologist-labelled)."""
    train = pd.read_csv(data_root() / "train.csv")
    gold = train[train[LABELS].notna().all(axis=1)][[ID_COL] + LABELS].copy()

    derived = pd.read_csv(args.labels)
    derived = derived[~derived[ID_COL].isin(gold[ID_COL])]

    # A study missing from the cache silently trains on a zero-filled volume, which
    # looks like a hard example rather than a bug. Check coverage up front.
    cache = Path(args.cache)
    have = {p.stem for p in cache.glob("*.npy")}
    for name, frame in (("derived", derived), ("gold", gold)):
        missing = (~frame[ID_COL].isin(have)).sum()
        if missing:
            pct = missing / len(frame)
            print(f"WARNING: {missing}/{len(frame)} {name} studies missing from cache "
                  f"({pct:.1%}) -- these train on zeros")
            if pct > 0.05:
                raise SystemExit(
                    f"{pct:.0%} of {name} studies are uncached. Finish preprocess.py "
                    "before training; a partial cache silently degrades every result."
                )

    if args.sharpen:
        before = derived[LABELS].std().mean()
        derived = sharpen(derived, args.sharpen_k)
        after = derived[LABELS].std().mean()
        print(f"sharpened labels: mean std {before:.3f} -> {after:.3f}")

    print(f"{len(derived)} studies with report-derived labels, {len(gold)} gold held out")
    return derived, gold


@torch.no_grad()
def evaluate(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    P, Y = [], []
    for x, y in loader:
        with torch.autocast("cuda", dtype=torch.float16, enabled=device == "cuda"):
            logits = model(x.to(device, non_blocking=True))
        P.append(torch.sigmoid(logits.float()).cpu().numpy())
        Y.append(y.numpy())
    return np.concatenate(P), np.concatenate(Y)


def train_fold(args, tr: pd.DataFrame, va: pd.DataFrame, gold: pd.DataFrame,
               fold: int, device: str) -> float:
    ds_kw = dict(cache=args.cache, n_series=args.n_series,
                 n_slices=args.n_slices, size=args.size)
    dl_kw = dict(num_workers=args.workers, pin_memory=(device == "cuda"))

    tr_dl = DataLoader(KneeStudies(tr, train=True, **ds_kw),
                       batch_size=args.batch, shuffle=True, drop_last=True, **dl_kw)
    va_dl = DataLoader(KneeStudies(va, train=False, **ds_kw),
                       batch_size=args.batch, **dl_kw)
    gold_dl = DataLoader(KneeStudies(gold, train=False, **ds_kw),
                         batch_size=args.batch, **dl_kw)

    model = KneeModel(args.backbone, len(LABELS), pretrained=args.pretrained).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=args.epochs * len(tr_dl), pct_start=0.1)
    scaler = torch.amp.GradScaler(enabled=(device == "cuda"))
    criterion = build_loss()

    best = -1.0
    for epoch in range(args.epochs):
        model.train()
        t0, total = time.time(), 0.0
        for i, (x, y) in enumerate(tr_dl):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=(device == "cuda")):
                loss = criterion(model(x), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            total += loss.item()
            if (i + 1) % 50 == 0:
                print(f"  fold{fold} ep{epoch} {i+1}/{len(tr_dl)} "
                      f"loss {total/(i+1):.4f}", flush=True)

        # Two scores, and only one of them is trustworthy.
        vp, vy = evaluate(model, va_dl, device)
        val_auc, _ = macro_auc((vy > 0.5).astype(int), vp)
        gp, gy = evaluate(model, gold_dl, device)
        gold_auc, per = macro_auc(gy.astype(int), gp)

        print(f"  fold{fold} ep{epoch}  loss {total/len(tr_dl):.4f}  "
              f"derived-val {val_auc:.3f}  GOLD {gold_auc:.3f}  ({time.time()-t0:.0f}s)")

        if gold_auc > best:
            best = gold_auc
            print("    per-label GOLD: " + "  ".join(
                f"{c.split()[0][:4]} {per[c]:.2f}" for c in LABELS))
            torch.save({"model": model.state_dict(), "fold": fold,
                        "gold_auc": gold_auc, "per_label": per, "args": vars(args)},
                       Path(args.out) / f"fold{fold}.pt")
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--labels", type=Path, required=True, help="report-derived labels CSV")
    ap.add_argument("--out", type=Path, default=Path("weights"))
    ap.add_argument("--backbone", default="resnet34")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--only-fold", type=int, help="train a single fold (9h runtime limits)")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--n-series", type=int, default=3)
    ap.add_argument("--n-slices", type=int, default=12)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--no-pretrained", dest="pretrained", action="store_false")
    ap.add_argument("--no-sharpen", dest="sharpen", action="store_false",
                    help="train on raw ensemble scores (compressed toward 0.5)")
    ap.add_argument("--sharpen-k", type=float, default=8.0,
                    help="logistic steepness; higher is closer to hard 0/1 labels")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}  backbone: {args.backbone}")

    derived, gold = build_labels(args)

    # Plain KFold: the labels are soft, so there is nothing discrete to stratify on,
    # and studies are independent (one exam each, no patient-level leakage to guard).
    from sklearn.model_selection import KFold
    kf = KFold(args.folds, shuffle=True, random_state=args.seed)

    scores = []
    for fold, (ti, vi) in enumerate(kf.split(derived)):
        if args.only_fold is not None and fold != args.only_fold:
            continue
        s = train_fold(args, derived.iloc[ti], derived.iloc[vi], gold, fold, device)
        scores.append(s)
        print(f"fold {fold} best GOLD macro AUC: {s:.3f}\n")

    if scores:
        print(f"mean best GOLD macro AUC: {np.mean(scores):.3f}")
        print("\nGOLD is 58 studies. It is the only radiologist-labelled signal available,")
        print("and it is far too small to separate close models -- use it to catch")
        print("failures, and the leaderboard to rank what survives.")


if __name__ == "__main__":
    main()
