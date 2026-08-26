"""Build a compact image cache from the DICOM archive. Run once, reuse forever.

The imaging is roughly 0.9-1.8 TB across 24,371 series. Kaggle's writable working
directory holds ~19.5 GB, so caching everything is impossible and re-reading DICOM
every epoch is far too slow. This selects a few informative series per study, takes
a handful of slices from each, downsizes to uint8, and writes one .npy per study.

At the defaults (3 series x 12 slices x 224px) that is ~1.8 MB per study and about
8 GB total -- small enough to save as a Kaggle Dataset and attach to every later
training run, so this cost is paid once rather than per experiment.

    python src/preprocess.py --out /kaggle/working/cache --workers 4
    python src/preprocess.py --out cache --split test      # for the submission notebook
"""

from __future__ import annotations

import argparse
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import ID_COL, SERIES_COL  # noqa: E402
from paths import data_root  # noqa: E402

# Which series to keep, in priority order. Each slot is (plane, fluid_sensitive).
#
# Rationale, from how knees are read rather than from anything measured here:
#   sagittal fluid-sensitive  -- cruciates, meniscal tears, bone marrow oedema
#   coronal fluid-sensitive   -- collateral ligaments, meniscal body, compartment OA
#   axial fluid-sensitive     -- patellofemoral cartilage, effusion, synovitis
# Fluid-sensitive sequences are chosen first because 7 of the 12 labels are
# oedema-, fluid-, or inflammation-related.
SLOTS = [
    ("Sagittal", 1), ("Coronal", 1), ("Axial", 1),
    ("Sagittal", 0), ("Coronal", 0), ("Axial", 0),
]


def pick_series(series: pd.DataFrame, n_series: int) -> list[str]:
    """Choose up to n_series, spreading across planes before doubling up on one."""
    chosen: list[str] = []
    for plane, fluid in SLOTS:
        if len(chosen) >= n_series:
            break
        match = series[(series["Anatomical_Plane"] == plane)
                       & (series["Fluid_Sensitive"] == fluid)]
        if not match.empty:
            chosen.append(match.iloc[0][SERIES_COL])
    # Fall back to whatever exists if the study lacks the usual planes.
    for sid in series[SERIES_COL]:
        if len(chosen) >= n_series:
            break
        if sid not in chosen:
            chosen.append(sid)
    return chosen[:n_series]


def load_slices(series_dir: Path, n_slices: int, size: int) -> np.ndarray:
    """Evenly-spaced slices from the middle of a series, as uint8.

    Slices are ordered by InstanceNumber where present -- directory order is not
    anatomical order. The outer slices of a knee series are mostly soft tissue and
    air, so sampling is biased toward the middle 80%.
    """
    import pydicom

    files = sorted(series_dir.glob("*.dcm"))
    if not files:
        return np.zeros((n_slices, size, size), dtype=np.uint8)

    # Sort by InstanceNumber, reading headers only (cheap -- no pixel decode).
    def order(f: Path) -> float:
        try:
            ds = pydicom.dcmread(str(f), stop_before_pixels=True)
            return float(getattr(ds, "InstanceNumber", 0) or 0)
        except Exception:
            return 0.0

    files = sorted(files, key=order)

    lo, hi = int(len(files) * 0.1), int(np.ceil(len(files) * 0.9))
    core = files[lo:hi] or files
    idx = np.linspace(0, len(core) - 1, n_slices).round().astype(int)

    out = np.zeros((n_slices, size, size), dtype=np.uint8)
    for k, i in enumerate(idx):
        try:
            arr = pydicom.dcmread(str(core[i])).pixel_array.astype(np.float32)
        except Exception:
            continue
        out[k] = normalise(arr, size)
    return out


def normalise(arr: np.ndarray, size: int) -> np.ndarray:
    """Percentile-clip, scale to 0-255, resize.

    MRI intensity has no absolute meaning -- it varies by scanner, sequence, and
    institution, and this data comes from ~20 sites. Per-slice percentile
    normalisation is what makes them comparable; a fixed window would not.
    """
    import cv2

    lo, hi = np.percentile(arr, [1, 99])
    if hi <= lo:
        hi = lo + 1.0
    arr = np.clip((arr - lo) / (hi - lo), 0, 1)
    return (cv2.resize(arr, (size, size), interpolation=cv2.INTER_AREA) * 255).astype(np.uint8)


def process_study(args: tuple) -> tuple[str, bool, str]:
    study_id, series_ids, root, split, out_dir, n_slices, size = args
    dest = Path(out_dir) / f"{study_id}.npy"
    if dest.exists():
        return study_id, True, "cached"
    try:
        vols = [
            load_slices(Path(root) / f"{split}_series" / study_id / sid, n_slices, size)
            for sid in series_ids
        ]
        np.save(dest, np.stack(vols))          # (n_series, n_slices, size, size) uint8
        return study_id, True, ""
    except Exception:
        return study_id, False, traceback.format_exc(limit=2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--split", default="train", choices=["train", "test"])
    ap.add_argument("--n-series", type=int, default=3)
    ap.add_argument("--n-slices", type=int, default=12)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, help="process only N studies (for a smoke test)")
    args = ap.parse_args()

    root = data_root()
    args.out.mkdir(parents=True, exist_ok=True)

    series = pd.read_csv(root / f"{args.split}_series.csv")
    groups = series.groupby(ID_COL)
    studies = list(groups.groups)[: args.limit]

    per_study_mb = args.n_series * args.n_slices * args.size ** 2 / 1e6
    print(f"{len(studies)} studies -> {args.out}")
    print(f"{per_study_mb:.1f} MB each, ~{per_study_mb * len(studies) / 1024:.1f} GB total")

    jobs = [
        (s, pick_series(groups.get_group(s), args.n_series), str(root),
         args.split, str(args.out), args.n_slices, args.size)
        for s in studies
    ]

    done = failed = 0
    with ProcessPoolExecutor(args.workers) as pool:
        futures = [pool.submit(process_study, j) for j in jobs]
        for f in as_completed(futures):
            sid, ok, msg = f.result()
            done += 1
            if not ok:
                failed += 1
                if failed <= 3:
                    print(f"\nFAILED {sid}:\n{msg}")
            if done % 200 == 0:
                print(f"  {done}/{len(jobs)}  ({failed} failed)", flush=True)

    print(f"\ndone: {done - failed} written, {failed} failed")
    if failed:
        print("Failures become zero-filled volumes at train time rather than crashing;")
        print("check the count is small before trusting the run.")


if __name__ == "__main__":
    main()
