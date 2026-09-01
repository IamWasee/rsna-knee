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


def _cache_manifest(cache) -> dict:
    """Whatever built this cache, recorded so inference can prove it matches."""
    import json
    p = Path(cache) / "cache_manifest.json"
    return json.loads(p.read_text()) if p.exists() else {}


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


ID_ALIASES = ["StudyInstanceUID", "study_instance_uid", "studyinstanceuid",
              "study_id", "StudyID", "study", "id"]


def normalise_label_table(path: Path) -> pd.DataFrame:
    """Load a label table from any source and fail loudly if it will not train.

    Published tables do not share a convention: the id column may be named
    differently, the twelve targets may carry suffixes, extra confidence columns may
    ride along, and values may be hard 0/1 or soft. Any of those silently becomes a
    KeyError deep in the loop or, worse, a NaN loss that trains to nothing.
    """
    df = pd.read_csv(path)

    id_col = next((c for c in df.columns if c.strip() in ID_ALIASES), None)
    if id_col is None:
        raise SystemExit(f"no study-id column in {path.name}. Columns: {list(df.columns)}")
    if id_col != ID_COL:
        print(f"  id column '{id_col}' -> '{ID_COL}'")
        df = df.rename(columns={id_col: ID_COL})

    # Tolerate case and whitespace differences in the target names, nothing more.
    lookup = {c.strip().lower(): c for c in df.columns}
    rename, missing = {}, []
    for t in LABELS:
        hit = lookup.get(t.lower())
        if hit is None:
            missing.append(t)
        elif hit != t:
            rename[hit] = t
    if missing:
        raise SystemExit(
            f"{path.name} is missing {len(missing)} of the 12 targets: {missing}\n"
            f"columns present: {list(df.columns)}"
        )
    df = df.rename(columns=rename)

    # Keep per-label confidence if the table carries it. "No mention of synovitis"
    # and "moderate synovitis is present" both become a number, and only one of them
    # deserves full weight in the loss; the confidence is what separates them.
    conf = {}
    for t in LABELS:
        for cand in (f"{t}__conf", f"{t}_conf", f"{t} conf"):
            hit = lookup.get(cand.lower())
            if hit is not None:
                conf[hit] = f"{t}__conf"
                break
    if conf:
        df = df.rename(columns=conf)
        print(f"  {len(conf)}/12 confidence columns found -- used as sample weights")
        df = df[[ID_COL] + LABELS + [c for c in conf.values() if c in df.columns]]
    else:
        df = df[[ID_COL] + LABELS]

    for c in [c for c in df.columns if c != ID_COL]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in df.columns:
        if c.endswith("__conf"):
            df[c] = df[c].fillna(1.0).clip(0.0, 1.0)
    n_nan = int(df[LABELS].isna().sum().sum())
    if n_nan:
        # A NaN target makes the loss NaN and the run trains to nothing while
        # reporting a plausible-looking curve. Fill with the label's own mean.
        print(f"  {n_nan} non-numeric/missing values -> filled with per-label mean")
        df[LABELS] = df[LABELS].fillna(df[LABELS].mean())

    lo, hi = float(df[LABELS].min().min()), float(df[LABELS].max().max())
    if lo < -0.01 or hi > 1.01:
        print(f"  values span [{lo:.2f}, {hi:.2f}] -> rescaling each label to [0,1]")
        rng = df[LABELS].max() - df[LABELS].min()
        df[LABELS] = (df[LABELS] - df[LABELS].min()) / rng.replace(0, 1)
    hard = bool(((df[LABELS] == 0) | (df[LABELS] == 1)).all().all())
    print(f"  {len(df)} rows, values in [{lo:.2f}, {hi:.2f}]"
          f"{', hard 0/1' if hard else ', soft'}")
    return df


def report_oof(oof: pd.DataFrame, path: Path) -> None:
    """Per-label AUC over the pooled out-of-fold predictions, plus the file itself.

    Saved because the metric is a macro average: choosing ensemble weights, blending
    sources or tuning anything per label needs the predictions, not a summary. The
    targets here are the report-derived labels, so this measures agreement with the
    teacher -- read it for which labels are learnable from these images, and the
    annotated 58 for whether the teacher was right.
    """
    train = pd.read_csv(data_root() / "train.csv")
    y = train.set_index(ID_COL).reindex(oof[ID_COL])
    oof.to_csv(path, index=False)

    print(f"\n{'label':<18} {'OOF AUC':>8} {'pos rate':>9}")
    print("-" * 38)
    aucs = []
    for c in LABELS:
        t = (y[c] > 0.5).astype(int).values if c in y else None
        if t is None or len(set(t)) < 2:
            print(f"{c:<18} {'--':>8}"); continue
        from sklearn.metrics import roc_auc_score
        a = roc_auc_score(t, oof[c].values)
        aucs.append(a)
        print(f"{c:<18} {a:>8.3f} {t.mean():>9.3f}")
    print("-" * 38)
    print(f"{'MACRO':<18} {np.mean(aucs):>8.3f}   ({len(oof)} studies)")
    print(f"\nwrote {path} -- per-label ensemble weights and blending need this file.")


def build_labels(args) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Training frame (report-derived) and gold frame (58 radiologist-labelled)."""
    train = pd.read_csv(data_root() / "train.csv")
    gold = train[train[LABELS].notna().all(axis=1)][[ID_COL] + LABELS].copy()

    derived = normalise_label_table(Path(args.labels))
    derived = derived[~derived[ID_COL].isin(gold[ID_COL])]

    # Group duplicate report texts. train.csv has 4,407 studies but fewer unique
    # reports; identical text is the same patient or a repeated study, and plain
    # KFold would put the pair either side of the split. There is no patient id in
    # this dataset, so the report is the only grouping key available.
    text = train.set_index(ID_COL)["Report"].astype(str)
    derived["_group"] = derived[ID_COL].map(
        lambda i: hash(text.get(i, i)) if i in text.index else hash(i))
    n_dup = len(derived) - derived["_group"].nunique()
    if n_dup:
        print(f"  {n_dup} studies share a report with another study -- grouped so "
              f"they cannot straddle a fold boundary")

    overlap = derived[ID_COL].isin(train[ID_COL]).mean()
    if overlap < 0.9:
        raise SystemExit(
            f"only {overlap:.0%} of the label table's study ids appear in train.csv -- "
            "wrong table, or ids in a different format."
        )

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
    for batch in loader:
        x, y = batch[0], batch[1]
        with torch.autocast("cuda", dtype=torch.float16, enabled=device == "cuda"):
            logits = model(x.to(device, non_blocking=True))
        P.append(torch.sigmoid(logits.float()).cpu().numpy())
        Y.append(y.numpy())
    return np.concatenate(P), np.concatenate(Y)


def train_fold(args, tr: pd.DataFrame, va: pd.DataFrame, gold: pd.DataFrame,
               fold: int, device: str) -> float:
    ds_kw = dict(cache=args.cache, slots=args.slots,
                 n_slices=args.n_slices, size=args.size)
    dl_kw = dict(num_workers=args.workers, pin_memory=(device == "cuda"))

    tr_dl = DataLoader(KneeStudies(tr, train=True, **ds_kw),
                       batch_size=args.batch, shuffle=True, drop_last=True, **dl_kw)
    va_dl = DataLoader(KneeStudies(va, train=False, **ds_kw),
                       batch_size=args.batch, **dl_kw)
    gold_dl = DataLoader(KneeStudies(gold, train=False, **ds_kw),
                         batch_size=args.batch, **dl_kw)

    model = KneeModel(args.backbone, LABELS, pretrained=args.pretrained,
                      head=args.head, pool=args.pool, n_slot=args.slots,
                      groups_per_slot=args.n_slices // 3,
                      unfreeze_last=args.unfreeze_last).to(device)

    # A pretrained encoder driven at the head's rate forgets what it knew before it
    # learns the task, so the two get separate rates. For a CNN trained from an
    # ImageNet init the gap matters less, but keeping one code path avoids a second
    # definition that has to be kept in step by hand.
    opt = torch.optim.AdamW(model.param_groups(args.lr, args.lr_backbone),
                            weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=[g["lr"] for g in opt.param_groups],
        total_steps=args.epochs * len(tr_dl), pct_start=0.1)
    scaler = torch.amp.GradScaler(enabled=(device == "cuda"))
    criterion = build_loss()

    best, best_oof = -1.0, None
    for epoch in range(args.epochs):
        model.train()
        t0, total = time.time(), 0.0
        for i, (x, y, w) in enumerate(tr_dl):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            w = w.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=(device == "cuda")):
                # Per-sample, per-label weights: a finding the report states plainly
                # counts more than one it never mentioned.
                loss = (criterion(model(x), y) * w).sum() / w.sum().clamp(min=1e-6)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            total += loss.item()
            if (i + 1) % 50 == 0:
                print(f"  fold{fold} ep{epoch} {i+1}/{len(tr_dl)} "
                      f"loss {total/(i+1):.4f}", flush=True)

        # Selection runs on this fold's own held-out studies. Selecting on the 58
        # annotated studies instead meant 60 choices against one small set -- the
        # resulting number is a fitted value, not a held-out one, and it is why the
        # reported score moved further than the leaderboard did.
        vp, vy = evaluate(model, va_dl, device)
        val_auc, val_per = macro_auc((vy > 0.5).astype(int), vp)
        gp, gy = evaluate(model, gold_dl, device)
        gold_auc, gold_per = macro_auc(gy.astype(int), gp)

        print(f"  fold{fold} ep{epoch}  loss {total/len(tr_dl):.4f}  "
              f"OOF {val_auc:.3f}  gold {gold_auc:.3f}  ({time.time()-t0:.0f}s)")

        if val_auc > best:
            best = val_auc
            best_oof = pd.DataFrame(vp, columns=LABELS)
            best_oof.insert(0, ID_COL, va[ID_COL].values)
            best_oof["fold"] = fold
            print("    per-label OOF: " + "  ".join(
                f"{c.split()[0][:4]} {val_per[c]:.2f}" for c in LABELS))
            torch.save({"model": model.state_dict(), "fold": fold,
                        "oof_auc": val_auc, "gold_auc": gold_auc,
                        "per_label": val_per, "gold_per_label": gold_per,
                        "args": vars(args), "cache_manifest": _cache_manifest(args.cache)},
                       Path(args.out) / f"fold{fold}.pt")
    return best, best_oof


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
    ap.add_argument("--lr", type=float, default=3e-4, help="head learning rate")
    ap.add_argument("--lr-backbone", type=float, default=3e-4,
                    help="encoder learning rate; for a pretrained ViT the public "
                         "baseline uses 8e-6 against a 1e-3 head, a 125x gap")
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--unfreeze-last", type=int, default=6,
                    help="trainable transformer blocks from the output end (ViT only)")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--slots", type=int, default=4)
    ap.add_argument("--n-slices", type=int, default=9)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--head", default="slot", choices=["slot", "shared"],
                    help="slot: one attention query per diagnosis over sequence types; "
                         "shared: a single attention for all twelve labels")
    ap.add_argument("--pool", default="focal", choices=["focal", "gap"],
                    help="focal keeps the upper tail of each channel alongside the mean, "
                         "so a small lesion is not averaged away")
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
    from sklearn.model_selection import GroupKFold
    kf = GroupKFold(args.folds)

    scores, oof = [], []
    for fold, (ti, vi) in enumerate(kf.split(derived, groups=derived["_group"])):
        if args.only_fold is not None and fold != args.only_fold:
            continue
        s, fold_oof = train_fold(args, derived.iloc[ti], derived.iloc[vi], gold,
                                 fold, device)
        scores.append(s)
        oof.append(fold_oof)
        print(f"fold {fold} best OOF macro AUC: {s:.3f}\n")

    if oof:
        report_oof(pd.concat(oof, ignore_index=True), Path(args.out) / "oof.csv")

    if scores:
        print(f"mean best OOF macro AUC: {np.mean(scores):.3f}")
        print("\nOOF is measured on each fold's own held-out studies against the")
        print("report-derived labels, so it is honest but only as good as those labels.")
        print("The 58 annotated studies are reported alongside as a second opinion and")
        print("are no longer used to choose checkpoints -- 60 selections against 58")
        print("studies fits them rather than measuring them.")


if __name__ == "__main__":
    main()
