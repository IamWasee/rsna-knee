"""Generate submission.csv. Runs inside the offline Kaggle submission notebook.

Constraints this file exists to respect:
  - no internet, so backbone weights come from the checkpoint, never a download
  - 9 hours total, including preprocessing the hidden test set from raw DICOM
  - a malformed CSV fails after all of that, so submission.py validates strictly

    python src/infer.py --cache test_cache --weights weights/ --out submission.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import ID_COL, LABELS  # noqa: E402
from dataset import KneeStudies  # noqa: E402
from model import KneeModel  # noqa: E402
from paths import data_root  # noqa: E402
from submission import validate, write  # noqa: E402


def load_models(weights: Path, device: str) -> tuple[list, dict]:
    ckpts = sorted(weights.glob("*.pt")) if weights.is_dir() else [weights]
    if not ckpts:
        raise FileNotFoundError(f"no .pt checkpoints in {weights}")

    models, cfg, manifest = [], {}, {}
    for p in ckpts:
        ck = torch.load(p, map_location=device, weights_only=False)
        cfg = ck["args"]
        manifest = ck.get("cache_manifest", manifest)
        # pretrained=False: the weights are in the checkpoint, and there is no
        # internet here to fetch them from anyway.
        m = KneeModel(cfg["backbone"], LABELS, pretrained=False,
                      head=cfg.get("head", "shared"), pool=cfg.get("pool", "gap"),
                      n_slot=cfg.get("slots", 4),
                      groups_per_slot=cfg.get("n_slices", 9) // 3,
                      unfreeze_last=cfg.get("unfreeze_last", 6)).to(device)
        m.load_state_dict(ck["model"])
        m.eval()
        models.append(m)
        print(f"  {p.name}  oof={ck.get('oof_auc', float('nan')):.3f} "
              f"gold={ck.get('gold_auc', float('nan')):.3f}")
    return models, cfg, manifest


def check_parity(train_manifest: dict, cache: Path) -> None:
    """Refuse to infer from a cache built differently from the training one.

    A silent mismatch here previously served the model the top-left corner of each
    study plus blank slices and still produced a plausible submission -- it scored
    0.675 with nothing in the log to say why. Every field that changes the pixels is
    compared, not just the ones that change the tensor shape.
    """
    import json

    p = cache / "cache_manifest.json"
    if not train_manifest:
        print("  checkpoint predates cache manifests -- parity NOT verified")
        return
    if not p.exists():
        raise SystemExit(f"no cache_manifest.json in {cache}; cannot verify parity")

    test = json.loads(p.read_text())
    # laterality defaults to False for manifests written before it existed: a cache
    # with no such key was built without it. Left it out of this list and a model
    # trained on unnormalised knees would be served mirrored ones at test time --
    # every pixel moved, the tensor shape unchanged, and nothing in the log.
    fields = ["slots", "size", "crop_mm", "n_anchors", "group", "n_slices", "band",
              "laterality"]
    bad = [(f, train_manifest.get(f, False if f == "laterality" else None),
            test.get(f, False if f == "laterality" else None))
           for f in fields
           if train_manifest.get(f, False if f == "laterality" else None)
           != test.get(f, False if f == "laterality" else None)]
    if bad:
        lines = "\n".join(f"    {f}: trained on {a!r}, test cache has {b!r}"
                          for f, a, b in bad)
        hint = ("\n  laterality differs: pass --no-laterality to preprocess.py to match"
                " a cache built before it existed, or retrain on a normalised cache."
                if any(f == "laterality" for f, _, _ in bad) else "")
        raise SystemExit("cache/checkpoint preprocessing mismatch:\n" + lines + hint)
    print(f"  preprocessing parity verified ({len(fields)} fields match)")


def tta_views(x: torch.Tensor, n: int) -> list[torch.Tensor]:
    """The identity plus small rigid jitters.

    No flips, for the same reason training has none: horizontal mirrors medial into
    lateral, corrupting four of the twelve targets, and vertical moves the knee off
    the orientation it is always acquired in. What is left are transforms the label
    genuinely does not depend on.
    """
    import torch.nn.functional as F

    views = [x]
    for k in range(1, n):
        ang = (-1) ** k * (3.0 + 2.0 * (k // 2)) * np.pi / 180
        sc = 1.0 + 0.03 * (k % 2)
        b = x.shape[0] * x.shape[1]
        flat = x.flatten(0, 1)
        cos, sin = float(np.cos(ang)) / sc, float(np.sin(ang)) / sc
        theta = torch.tensor([[cos, -sin, 0.0], [sin, cos, 0.0]],
                             device=x.device, dtype=torch.float32)
        theta = theta.unsqueeze(0).expand(b, -1, -1)
        grid = F.affine_grid(theta, flat.shape, align_corners=False)
        warped = F.grid_sample(flat, grid, mode="bilinear",
                               padding_mode="border", align_corners=False)
        views.append(warped.view_as(x))
    return views


@torch.no_grad()
def predict(models: list, loader: DataLoader, device: str,
            tta: int = 1) -> tuple[np.ndarray, list]:
    preds, ids = [], []
    for i, (x, sid) in enumerate(loader):
        x = x.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16, enabled=(device == "cuda")):
            # Keep each fold separate; they are combined by rank after every study
            # has been scored, which cannot be done batch by batch. TTA views are
            # averaged within a fold, since they are the same model on the same
            # study and their scales are directly comparable.
            p = torch.stack([
                torch.stack([torch.sigmoid(m(v).float())
                             for v in tta_views(x, tta)]).mean(0)
                for m in models])
        preds.append(p.cpu().numpy())
        ids.extend(sid)
        if (i + 1) % 25 == 0 or i + 1 == len(loader):
            print(f"  {i+1}/{len(loader)}", flush=True)
    return np.concatenate(preds), ids


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, required=True, help="preprocessed test volumes")
    ap.add_argument("--weights", type=Path, required=True, help=".pt file or directory")
    ap.add_argument("--out", type=Path, default=Path("submission.csv"))
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--tta", type=int, default=1,
                    help="views per model: 1 disables TTA, 3-5 is typical. Validate "
                         "out of fold before trusting it -- averaging jittered views "
                         "can as easily blur a correct ranking as sharpen it.")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    root = data_root()

    test = pd.read_csv(root / "test.csv")
    print(f"{len(test)} test studies")

    models, cfg, train_manifest = load_models(args.weights, device)
    check_parity(train_manifest, args.cache)

    # Shape comes from the cache manifest first -- that records what the pixels
    # actually are -- then the checkpoint args, and it is an error if neither has it.
    # A silent .get(key, default) here is what served a model trained on 3 series a
    # tensor built for 4: the default was returned, _load padded the difference, and
    # the run scored 0.675 with nothing in the log to say why.
    def _shape(key, alt=None):
        for src in (train_manifest, cfg):
            for k in (key, alt):
                if k and k in src:
                    return src[k]
        raise SystemExit(
            f"checkpoint and cache manifest both lack '{key}'. This checkpoint "
            "predates shape recording and cannot be served safely -- retrain, or "
            "pass the original preprocessing settings explicitly."
        )

    ds = KneeStudies(test, cache=args.cache, train=False,
                     slots=_shape("slots", "n_series"),
                     n_slices=_shape("n_slices"),
                     size=_shape("size"))
    loader = DataLoader(ds, batch_size=args.batch, num_workers=args.workers)

    per_model, ids = predict(models, loader, device, tta=args.tta)
    preds = rank_average(per_model)
    print(f"rank-averaged {per_model.shape[0]} fold(s) over {per_model.shape[1]} studies")

    sub = pd.DataFrame(preds, columns=LABELS)
    sub.insert(0, ID_COL, ids)

    # Every test study must appear, in the order the sample file expects.
    sample = pd.read_csv(root / "sample_submission.csv")
    sub = sample[[ID_COL]].merge(sub, on=ID_COL, how="left")
    missing = sub[LABELS].isna().any(axis=1).sum()
    if missing:
        print(f"WARNING: {missing} studies had no prediction -- filling 0.5")
        sub[LABELS] = sub[LABELS].fillna(0.5)

    validate(sub, expected_ids=sample[ID_COL])
    write(sub, args.out)


if __name__ == "__main__":
    main()
