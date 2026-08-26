"""Discover the dataset layout. Makes no assumptions -- run this FIRST.

Everything downstream (dataset.py, label extraction, training) should be written
against what this prints, not against a guess.

    python src/explore.py            # on Kaggle, or with RSNA_KNEE_DATA set
    python src/explore.py --dicom    # also crack open one DICOM's headers
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import LABELS, ID_COL  # noqa: E402
from paths import data_root  # noqa: E402


def show_tree(root: Path, max_entries: int = 25) -> None:
    print(f"\n=== TOP LEVEL: {root} ===")
    entries = sorted(root.iterdir())
    for e in entries[:max_entries]:
        kind = "dir " if e.is_dir() else "file"
        size = "" if e.is_dir() else f"  {e.stat().st_size/1e6:.1f} MB"
        print(f"  [{kind}] {e.name}{size}")
    if len(entries) > max_entries:
        print(f"  ... and {len(entries)-max_entries} more")


def probe_image_dirs(root: Path) -> None:
    """Walk one branch deep into each image directory to learn the nesting."""
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        children = sorted(d.iterdir())
        print(f"\n=== {d.name}/  ({len(children)} entries) ===")
        for c in children[:3]:
            print(f"  {c.name}  ({'dir' if c.is_dir() else 'file'})")
            if c.is_dir():
                grand = sorted(c.iterdir())
                print(f"    -> {len(grand)} entries, e.g. {[g.name for g in grand[:3]]}")
                if grand and grand[0].is_dir():
                    ggrand = sorted(grand[0].iterdir())
                    print(f"       -> {len(ggrand)} files, e.g. {[g.name for g in ggrand[:3]]}")


def summarize_csvs(root: Path) -> None:
    import pandas as pd

    for csv in sorted(root.rglob("*.csv")):
        print(f"\n=== CSV: {csv.relative_to(root)} ===")
        df = pd.read_csv(csv)
        print(f"  shape: {df.shape}")
        print(f"  columns: {list(df.columns)}")
        print(df.head(3).to_string(max_colwidth=60))

        label_cols = [c for c in LABELS if c in df.columns]
        if label_cols:
            print("\n  label prevalence (positive rate):")
            for c in label_cols:
                pos = df[c].mean()
                print(f"    {c:<18} {pos:.4f}  (n_pos={int(df[c].sum())})")

        # Free-text report columns are the multimodal half of this competition.
        for c in df.columns:
            if df[c].dtype == object:
                lens = df[c].astype(str).str.len()
                if lens.mean() > 100:
                    print(f"\n  >>> '{c}' looks like report text (mean len {lens.mean():.0f})")
                    print("  --- sample ---")
                    print("  " + df[c].iloc[0][:800].replace("\n", "\n  "))


def probe_dicom(root: Path) -> None:
    import pydicom

    dcm = next(root.rglob("*.dcm"), None)
    if dcm is None:
        print("\n(no .dcm files found -- images may be in another format)")
        return
    print(f"\n=== DICOM: {dcm.relative_to(root)} ===")
    ds = pydicom.dcmread(str(dcm))
    for tag in [
        "Modality", "SeriesDescription", "SequenceName", "ScanningSequence",
        "MRAcquisitionType", "Rows", "Columns", "PixelSpacing", "SliceThickness",
        "StudyInstanceUID", "SeriesInstanceUID", "InstanceNumber",
        "PhotometricInterpretation", "BitsStored", "RescaleSlope", "RescaleIntercept",
    ]:
        if hasattr(ds, tag):
            print(f"  {tag:<26} {getattr(ds, tag)}")
    try:
        arr = ds.pixel_array
        print(f"  pixel_array               shape={arr.shape} dtype={arr.dtype} "
              f"min={arr.min()} max={arr.max()}")
    except Exception as e:  # pragma: no cover
        print(f"  pixel_array               FAILED: {e}")


def series_stats(root: Path, n_studies: int = 20) -> None:
    """How many series per study, how many slices per series, what descriptions?"""
    import pydicom

    dcms = root.rglob("*.dcm")
    per_series = Counter()
    descs = Counter()
    seen = 0
    for f in dcms:
        per_series[f.parent] += 1
        if per_series[f.parent] == 1:
            seen += 1
            try:
                ds = pydicom.dcmread(str(f), stop_before_pixels=True)
                descs[getattr(ds, "SeriesDescription", "?")] += 1
            except Exception:
                pass
        if seen > n_studies * 6:
            break
    if per_series:
        counts = sorted(per_series.values())
        print(f"\n=== SERIES STATS (first ~{len(per_series)} series) ===")
        print(f"  slices/series: min={counts[0]} median={counts[len(counts)//2]} max={counts[-1]}")
        print("  most common SeriesDescription:")
        for d, n in descs.most_common(12):
            print(f"    {n:>4}  {d}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dicom", action="store_true", help="inspect a DICOM header + pixels")
    ap.add_argument("--series", action="store_true", help="slices-per-series / sequence stats (slow)")
    args = ap.parse_args()

    root = data_root()
    show_tree(root)
    probe_image_dirs(root)
    summarize_csvs(root)
    if args.dicom:
        probe_dicom(root)
    if args.series:
        series_stats(root)


if __name__ == "__main__":
    main()
