"""Build a compact image cache from the DICOM archive. Run once, reuse forever.

Three things here are not obvious and all three were measured by other competitors
before being adopted:

1. Slice order comes from geometry, never filenames. The filenames are SOP Instance
   UIDs -- unique, not ordered -- and sorted filename order matches true anatomical
   order about 5% of the time on this corpus. Slices are sorted by projecting
   ImagePositionPatient onto the slice normal from ImageOrientationPatient.

2. The crop is in millimetres, not pixels. Field of view spans 71 distinct values
   with a median of 160 mm, so resizing whole frames to a fixed pixel size leaves
   every study at a different physical scale. A 1-3 mm meniscal tear can disappear
   before the first convolution. Cropping to a fixed physical extent first fixes it.

3. Slices come in adjacent triplets, not evenly spread. Three anchors across the
   joint, three physically adjacent slices at each, so every triplet is a real
   ~10 mm depth neighbourhood rather than a slab of unrelated views. Measured at
   +0.018 macro AUC over evenly-spaced sampling.

    python src/preprocess.py --out /kaggle/working/cache --workers 4
    python src/preprocess.py --out cache --split test
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

# Sequence slots, in priority order: (plane, fluid_sensitive).
# Fluid_Sensitive and Fat_Suppression are identical columns in this dataset, so
# fluid_sensitive=1 is the fat-suppressed fluid-sensitive sequence and 0 is the
# anatomical (T1/PD) one. Seven of the twelve labels are fluid, oedema or
# inflammation findings, so fluid-sensitive planes come first.
SLOTS = [
    ("Sagittal", 1), ("Coronal", 1), ("Axial", 1), ("Sagittal", 0),
]

N_ANCHORS = 3      # anchor positions across the joint
GROUP = 3          # physically adjacent slices per anchor -> N_ANCHORS*GROUP total
CROP_MM = 140.0    # physical field kept before resizing
LAT_MIN_OFFSET_MM = 20.0   # inside this the side is not readable from geometry


def pick_series(series: pd.DataFrame, n_slots: int) -> list[str]:
    """One series per slot, falling back to whatever the study actually has."""
    chosen: list[str] = []
    for plane, fluid in SLOTS[:n_slots]:
        m = series[(series["Anatomical_Plane"] == plane)
                   & (series["Fluid_Sensitive"] == fluid)]
        chosen.append(m.iloc[0][SERIES_COL] if not m.empty else None)

    spare = [s for s in series[SERIES_COL] if s not in chosen]
    for i, c in enumerate(chosen):
        if c is None and spare:
            chosen[i] = spare.pop(0)
    return [c for c in chosen if c is not None][:n_slots]


def sort_by_position(files: list[Path], pydicom) -> list[Path]:
    """Order slices anatomically, by geometry.

    Projecting the slice origin onto the normal of the imaging plane gives a true
    physical coordinate along the stack axis. Falls back through SliceLocation and
    InstanceNumber; filename order is the last resort and is close to random here.
    """
    keyed = []
    for f in files:
        k = None
        try:
            ds = pydicom.dcmread(str(f), stop_before_pixels=True)
            iop = getattr(ds, "ImageOrientationPatient", None)
            ipp = getattr(ds, "ImagePositionPatient", None)
            if iop is not None and ipp is not None and len(iop) == 6:
                normal = np.cross(np.array(iop[:3], float), np.array(iop[3:], float))
                k = float(np.dot(np.array(ipp, float), normal))
            elif getattr(ds, "SliceLocation", None) is not None:
                k = float(ds.SliceLocation)
            elif getattr(ds, "InstanceNumber", None) is not None:
                k = float(ds.InstanceNumber)
        except Exception:
            pass
        keyed.append((k if k is not None else float("inf"), f.name, f))
    keyed.sort(key=lambda t: (t[0], t[1]))
    return [t[2] for t in keyed]


def image_centre_x(ds) -> float | None:
    """x of the image centre in patient coordinates.

    ImagePositionPatient is the centre of the FIRST voxel, i.e. a corner. Using it
    directly reads the edge of the field of view, which for a knee near the midline
    can sit on the wrong side of zero. Stepping to the middle of the image first
    gives the side the anatomy is actually on. DICOM +x is the patient's left.
    """
    iop = getattr(ds, "ImageOrientationPatient", None)
    ipp = getattr(ds, "ImagePositionPatient", None)
    if iop is None or ipp is None or len(iop) != 6:
        return None
    try:
        row = np.array(iop[:3], float)
        col = np.array(iop[3:], float)
        pos = np.array(ipp, float)
        ps = getattr(ds, "PixelSpacing", [1.0, 1.0])
        centre = pos + row * (float(ds.Columns) / 2) * float(ps[1]) \
                     + col * (float(ds.Rows) / 2) * float(ps[0])
        return float(centre[0])
    except Exception:
        return None


def series_side(ds) -> str | None:
    """'L' or 'R' for this series, from the tag when present, else from geometry.

    The Laterality tag is populated on only about half of these studies, so geometry
    is the fallback rather than the exception.
    """
    tag = str(getattr(ds, "Laterality", "") or "").strip().upper()
    if tag in ("L", "R"):
        return tag
    x = image_centre_x(ds)
    if x is None or abs(x) < LAT_MIN_OFFSET_MM:
        return None            # too near the midline to call
    return "L" if x > 0 else "R"


def scanner_fingerprint(ds) -> str:
    """Identify the scanner, for grouping folds by site.

    Random K-fold over studies lets a model memorise the scanner instead of the
    knee: a competitor measured a 0.053 macro AUC drop moving to scanner-grouped
    folds, and a 0.136 gap at pixel level. Grouping on this makes the held-out score
    mean what it claims to.
    """
    parts = [str(getattr(ds, t, "") or "") for t in
             ("Manufacturer", "ManufacturerModelName", "SoftwareVersions",
              "ImagingFrequency", "ReceiveCoilName")]
    return "|".join(p.strip() for p in parts)


def pixel_spacing(ds) -> float:
    ps = getattr(ds, "PixelSpacing", None)
    if ps is not None and len(ps) >= 1:
        try:
            return float(ps[0])
        except (TypeError, ValueError):
            pass
    return 0.0


def crop_resize(arr: np.ndarray, spacing: float, size: int, crop_mm: float) -> np.ndarray:
    """Percentile-normalise, crop to a fixed physical extent, resize.

    MRI intensity has no absolute meaning -- it varies by scanner, sequence and
    site, and this data comes from ~20 institutions -- so normalisation is per
    slice by percentile rather than a fixed window.
    """
    import cv2

    lo, hi = np.percentile(arr, [1, 99])
    if hi <= lo:
        hi = lo + 1.0
    img = np.clip((arr.astype(np.float32) - lo) / (hi - lo), 0, 1)

    if spacing > 0:
        want = int(round(crop_mm / spacing))
        h, w = img.shape
        # Only crop when the frame is genuinely larger; a silent no-op otherwise.
        if want < min(h, w):
            cy, cx = h // 2, w // 2
            half = want // 2
            img = img[max(0, cy - half):cy + half, max(0, cx - half):cx + half]

    return (cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA) * 255).astype(np.uint8)


def load_slot(series_dir: Path, size: int, crop_mm: float,
              n_anchors: int = N_ANCHORS, plane: str = "",
              normalise_side: bool = True) -> tuple[np.ndarray, str | None, str]:
    """n_anchors anchors across the joint, GROUP adjacent slices at each.

    Returns the volume, the detected side, and the scanner fingerprint.

    Laterality is normalised so every study is presented as a left knee. Which
    operation achieves that depends on the plane, and getting this wrong is worse
    than not doing it:

      coronal, axial   -- medial/lateral runs across the IMAGE, so mirror the pixels
      sagittal         -- medial/lateral is the STACK axis, so reverse the slice
                          order; mirroring the pixels would flip anterior/posterior
                          instead and put the patella at the back of the knee
    """
    import pydicom

    n_out = n_anchors * GROUP
    out = np.zeros((n_out, size, size), dtype=np.uint8)

    files = list(series_dir.glob("*.dcm"))
    if not files:
        return out, None, ""
    files = sort_by_position(files, pydicom)

    side, fp = None, ""
    try:
        head = pydicom.dcmread(str(files[0]), stop_before_pixels=True)
        side, fp = series_side(head), scanner_fingerprint(head)
    except Exception:
        pass

    flip_lr = normalise_side and side == "R" and plane in ("Coronal", "Axial")
    if normalise_side and side == "R" and plane == "Sagittal":
        files = files[::-1]

    # Anchors spread across the central 70%: the outer slices of a knee series are
    # mostly soft tissue and air, but the menisci sit peripherally, so the window
    # stays wide rather than hugging the middle.
    lo, hi = int(len(files) * 0.15), int(np.ceil(len(files) * 0.85))
    core = files[lo:hi] or files
    lo_a, hi_a = GROUP // 2, len(core) - 1 - GROUP // 2
    if n_anchors == 1:
        # np.linspace(a, b, 1) returns [a], not the midpoint -- with one anchor that
        # samples the far edge of the stack, which on a sagittal knee is soft tissue
        # with no joint in it. One anchor means the centre.
        anchors = np.array([(lo_a + hi_a) / 2])
    else:
        anchors = np.linspace(lo_a, hi_a, n_anchors)
    anchors = np.clip(anchors.round().astype(int), 0, max(0, len(core) - 1))

    k = 0
    for a in anchors:
        for d in range(-(GROUP // 2), GROUP // 2 + 1):
            i = int(np.clip(a + d, 0, len(core) - 1))
            try:
                ds = pydicom.dcmread(str(core[i]))
                img = crop_resize(ds.pixel_array, pixel_spacing(ds), size, crop_mm)
                out[k] = img[:, ::-1] if flip_lr else img
            except Exception:
                pass
            k += 1
    return out, side, fp


def process_study(args: tuple) -> tuple[str, bool, str]:
    (study_id, series_ids, root, split, out_dir, size, crop_mm, n_slots, n_anchors,
     planes, normalise_side) = args
    dest = Path(out_dir) / f"{study_id}.npy"
    if dest.exists():
        return study_id, True, "cached"
    try:
        n_out = n_anchors * GROUP
        vol = np.zeros((n_slots, n_out, size, size), dtype=np.uint8)
        sides, fps = [], []
        for i, sid in enumerate(series_ids[:n_slots]):
            v, side, fp = load_slot(Path(root) / f"{split}_series" / study_id / sid,
                                    size, crop_mm, n_anchors,
                                    plane=planes.get(sid, ""),
                                    normalise_side=normalise_side)
            vol[i] = v
            if side:
                sides.append(side)
            if fp:
                fps.append(fp)
        np.save(dest, vol)
        # Majority side across the study's series; a study is one knee.
        side = max(set(sides), key=sides.count) if sides else ""
        return study_id, True, f"{side}\t{fps[0] if fps else ''}"
    except Exception:
        return study_id, False, traceback.format_exc(limit=2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--split", default="train", choices=["train", "test"])
    ap.add_argument("--slots", type=int, default=4)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--crop-mm", type=float, default=CROP_MM)
    ap.add_argument("--anchors", type=int, default=N_ANCHORS,
                    help="anchor groups per slot; slices per slot is anchors x 3. "
                         "One group at 336px matches the public solutions and costs "
                         "less storage than three at 256px.")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--no-laterality", action="store_true",
                    help="skip presenting every knee as a left one")
    ap.add_argument("--limit", type=int, help="process only N studies (smoke test)")
    args = ap.parse_args()

    root = data_root()
    args.out.mkdir(parents=True, exist_ok=True)

    series = pd.read_csv(root / f"{args.split}_series.csv")
    groups = series.groupby(ID_COL)
    studies = list(groups.groups)[: args.limit]

    n_out = args.anchors * GROUP
    mb = args.slots * n_out * args.size ** 2 / 1e6
    print(f"{len(studies)} studies -> {args.out}")
    print(f"{args.slots} slots x {n_out} slices ({args.anchors} anchors x {GROUP}) "
          f"x {args.size}px, {args.crop_mm:.0f}mm crop")
    print(f"{mb:.1f} MB each, ~{mb * len(studies) / 1024:.1f} GB total")

    # Plane per series id, so load_slot knows whether a right knee needs its pixels
    # mirrored (coronal/axial) or its slice order reversed (sagittal).
    planes = dict(zip(series[SERIES_COL], series["Anatomical_Plane"]))

    jobs = [(s, pick_series(groups.get_group(s), args.slots), str(root), args.split,
             str(args.out), args.size, args.crop_mm, args.slots, args.anchors,
             planes, not args.no_laterality)
            for s in studies]

    done = failed = 0
    meta = {}
    with ProcessPoolExecutor(args.workers) as pool:
        futures = [pool.submit(process_study, j) for j in jobs]
        for f in as_completed(futures):
            sid, ok, msg = f.result()
            done += 1
            if not ok:
                failed += 1
                if failed <= 3:
                    print(f"\nFAILED {sid}:\n{msg}")
            elif msg and msg != "cached" and "\t" in msg:
                side, fp = msg.split("\t", 1)
                meta[sid] = (side, fp)
            if done % 200 == 0:
                print(f"  {done}/{len(jobs)}  ({failed} failed)", flush=True)

    # Side and scanner per study. The scanner is what folds must be grouped on:
    # random splits let a model memorise the site instead of the knee, worth an
    # inflated 0.053 macro AUC by one competitor's measurement.
    out_csv = args.out / "study_meta.csv"

    # Merge rather than overwrite. Studies already on disk return "cached" and
    # contribute no row, so a full run that follows a --limit smoke test would
    # otherwise drop the smoke test's studies from the file entirely -- and a study
    # missing here is a study whose fold cannot be grouped by scanner.
    if out_csv.exists():
        prev = pd.read_csv(out_csv).fillna("")
        for r in prev.itertuples(index=False):
            meta.setdefault(str(r.StudyInstanceUID), (str(r.side), str(r.scanner)))

    if meta:
        with out_csv.open("w") as fh:
            fh.write("StudyInstanceUID,side,scanner\n")
            for sid, (side, fp) in sorted(meta.items()):
                fh.write(f'{sid},{side},"{fp}"\n')
        sides = [v[0] for v in meta.values() if v[0]]
        scanners = {v[1] for v in meta.values() if v[1]}
        print(f"laterality: {len(sides)}/{len(meta)} resolved "
              f"({sides.count('R')} right normalised to left)")
        print(f"scanners: {len(scanners)} distinct fingerprints")
        print(f"wrote {out_csv}")
        gap = len(studies) - len(meta)
        if gap > 0:
            print(f"WARNING: {gap} of {len(studies)} studies have no metadata row. "
                  f"Those studies cannot be grouped by scanner. Delete {out_csv} and "
                  f"the .npy files for a clean rebuild if this number is large.")

    # Record how this cache was built. The checkpoint stores a copy, and inference
    # compares the two: a cache built with different constants is the failure that
    # silently served corner crops and blank slices to a model expecting neither.
    import json
    manifest = {"slots": args.slots, "size": args.size, "crop_mm": args.crop_mm,
                "n_anchors": args.anchors, "group": GROUP,
                "n_slices": args.anchors * GROUP, "band": [0.15, 0.85],
                "laterality": not args.no_laterality,
                "slot_scheme": [list(x) for x in SLOTS[:args.slots]], "split": args.split}
    (args.out / "cache_manifest.json").write_text(json.dumps(manifest, indent=2))
    print("manifest:", json.dumps(manifest))

    print(f"\ndone: {done - failed} written, {failed} failed")


if __name__ == "__main__":
    main()
