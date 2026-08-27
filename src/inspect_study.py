"""Look at one knee: the slices, the model's 12 calls, the truth, the report.

Aggregate AUC says how good a model is; it never says what it is doing. This shows
a single study the way a person would read it -- the images, what the model
predicted, what the radiologist said, and which slices the model actually weighted.

Runs on the 58 gold studies by default, because those are the only ones with real
radiologist labels to be right or wrong against.

    python src/inspect_study.py --cache cache --weights weights/ --n 3
    python src/inspect_study.py --cache cache --weights weights/ --worst
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import ID_COL, LABELS  # noqa: E402
from dataset import KneeStudies  # noqa: E402
from model import KneeModel  # noqa: E402
from paths import data_root  # noqa: E402

BAR = 28


def bar(p: float) -> str:
    return "#" * int(round(p * BAR)) + "." * (BAR - int(round(p * BAR)))


def verdict(pred: float, truth: float) -> str:
    """Plain-language call, judged at 0.5."""
    said = pred > 0.5
    real = truth > 0.5
    if said and real:
        return "correct    (found it)"
    if not said and not real:
        return "correct    (ruled out)"
    if said and not real:
        return "WRONG      (false alarm)"
    return "WRONG      (missed it)"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--n", type=int, default=3, help="how many studies to show")
    ap.add_argument("--worst", action="store_true",
                    help="show the studies the model got most wrong")
    ap.add_argument("--figdir", default="/kaggle/working/figs",
                    help="where to save the attention images; empty string disables")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train = pd.read_csv(data_root() / "train.csv")
    gold = train[train[LABELS].notna().all(axis=1)].reset_index(drop=True)

    ckpts = sorted(args.weights.glob("*.pt")) if args.weights.is_dir() else [args.weights]
    models, cfg = [], {}
    for c in ckpts:
        ck = torch.load(c, map_location=device, weights_only=False)
        cfg = ck["args"]
        m = KneeModel(cfg["backbone"], len(LABELS), pretrained=False).to(device)
        m.load_state_dict(ck["model"])
        m.eval()
        models.append(m)
    print(f"loaded {len(models)} checkpoint(s), backbone {cfg['backbone']}\n")

    ds = KneeStudies(gold, cache=args.cache, train=False,
                     n_series=cfg.get("n_series", 3), n_slices=cfg.get("n_slices", 12),
                     size=cfg.get("size", 224))

    # Predict on all 58 so "worst" is meaningful.
    preds, attns = [], []
    with torch.no_grad():
        for i in range(len(ds)):
            x = ds[i][0].unsqueeze(0).to(device)
            ps, ats = [], []
            for m in models:
                logit, a = m(x, return_attention=True)
                ps.append(torch.sigmoid(logit.float()))
                ats.append(a.float())
            preds.append(torch.cat(ps).mean(0).cpu().numpy())
            attns.append(torch.cat(ats).mean(0).cpu().numpy())
    preds = np.stack(preds)

    err = np.abs(preds - gold[LABELS].values).mean(axis=1)
    order = np.argsort(-err) if args.worst else np.arange(len(gold))

    for idx in order[: args.n]:
        row = gold.iloc[idx]
        print("=" * 74)
        print(f"STUDY {row[ID_COL][-24:]}     mean error {err[idx]:.3f}")
        print("=" * 74)

        n_right = 0
        print(f"\n{'finding':<18} {'model says':<30} {'truth':>6}  verdict")
        print("-" * 74)
        for j, c in enumerate(LABELS):
            p, t = preds[idx, j], row[c]
            v = verdict(p, t)
            n_right += v.startswith("correct")
            print(f"{c:<18} {bar(p)} {p:.2f}  {t:>5.0f}  {v}")
        print("-" * 74)
        print(f"{n_right}/12 correct on this study")

        # Which slices drove it -- attention is over series*slices, in cache order.
        a = attns[idx]
        n_sl = cfg.get("n_slices", 12)
        top = np.argsort(-a)[:5]
        print("\nslices the model weighted most (series, slice):")
        for k in top:
            print(f"   series {k // n_sl}  slice {k % n_sl:>2}   weight {a[k]:.3f}")

        print("\n--- what the radiologist wrote ---")
        print(str(row["Report"])[:900])
        print()

        if args.figdir:
            out = _plot(args.cache, row[ID_COL], a, n_sl, Path(args.figdir), int(idx))
            if out:
                print(f"[image saved: {out}]\n")


def _plot(cache: Path, study_id: str, attn: np.ndarray, n_slices: int,
          figdir: Path, tag: int) -> Path | None:
    """Save to PNG rather than show().

    This script runs as a subprocess under `!python`, and plt.show() in a
    subprocess renders to nothing -- Jupyter only displays figures created in its
    own kernel. Writing files lets the calling cell display them.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = Path(cache) / f"{study_id}.npy"
    if not path.exists():
        return None
    vol = np.load(path)
    top = np.argsort(-attn)[:6]
    fig, ax = plt.subplots(1, len(top), figsize=(3 * len(top), 3.4))
    for i, k in enumerate(top):
        s, sl = k // n_slices, k % n_slices
        ax[i].imshow(vol[s, sl], cmap="gray")
        ax[i].set_title(f"series {s} slice {sl}\nweight {attn[k]:.3f}", fontsize=9)
        ax[i].axis("off")
    fig.suptitle(f"study {study_id[-16:]} -- the six slices the model weighted most",
                 fontsize=11)
    plt.tight_layout()
    figdir.mkdir(parents=True, exist_ok=True)
    out = figdir / f"study_{tag:03d}.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return out


if __name__ == "__main__":
    main()
