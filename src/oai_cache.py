"""Build our one-sequence plane caches from downloaded OAI baseline series.

Each OAI series arrives as one <series>.tar.gz of extension-less DICOM slices,
under <root>/.../00m/<kit>/<subject>/<date>/. Three series per knee are used:

    SAG_IW_TSE_{LEFT,RIGHT}   sagittal intermediate-weighted TSE, fat-suppressed
    COR_MPR_{LEFT,RIGHT}      coronal reformat of the 3D DESS (water excitation)
    AX_MPR_{LEFT,RIGHT}       axial reformat of the 3D DESS

The plane comes from the series description, not the geometry: the MPR
reformats carry an ImageOrientationPatient that reads as axial for the coronal
series. Pixels then go through preprocess.load_slot unchanged, as in
fastmri_cache.py, so an OAI knee is cropped, sampled and side-normalised
exactly as a competition study.

A knee is oai_<subject>_<L|R>. Re-running skips knees already written, so it
can follow a download that is still in progress.

    python src/oai_cache.py --root /Volumes/USB-189982/oai --out data/oai_cache [--workers 6]

Writes <out>/cache_{sag,cor,ax}/<knee>.npy, (1, n_slices, size, size) uint8,
one cache_manifest.json per plane (split "oai"), and oai_index.csv.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import tarfile
import tempfile
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from preprocess import CROP_MM, GROUP, load_slot  # noqa: E402

PLANES = {"SAG_IW_TSE": ("Sagittal", "sag", 0), "COR_MPR": ("Coronal", "cor", 1),
          "AX_MPR": ("Axial", "ax", 2)}
DESC = re.compile(r"^(SAG_IW_TSE|COR_MPR|AX_MPR)_(LEFT|RIGHT)$")


def convert(job: tuple) -> dict:
    """One series archive -> one cache file. Never raises: a bad archive is a row."""
    tgz, out, size, crop_mm, anchors, band = job
    import pydicom

    row = {"archive": tgz.name, "knee": "", "plane": "", "status": ""}
    tmp = Path(tempfile.mkdtemp(prefix="oai_"))
    try:
        with tarfile.open(tgz, "r:gz") as tar:
            for m in tar.getmembers():
                # flat, extension-less slice files; anything else is skipped,
                # and no member name is used as a path
                if m.isfile() and re.fullmatch(r"[0-9A-Za-z_.-]+", Path(m.name).name):
                    data = tar.extractfile(m).read()
                    (tmp / f"{Path(m.name).name}.dcm").write_bytes(data)
        files = sorted(tmp.glob("*.dcm"))
        if not files:
            row["status"] = "empty"
            return row
        head = pydicom.dcmread(str(files[0]), stop_before_pixels=True)
        mt = DESC.match(str(getattr(head, "SeriesDescription", "")).strip().upper())
        if not mt:
            row["status"] = "not a used series"
            return row
        plane, short, _ = PLANES[mt.group(1)]
        subject = str(tgz.parent.parent.name)
        if not subject.isdigit():
            row["status"] = "unexpected path"
            return row
        knee = f"oai_{subject}_{mt.group(2)[0]}"
        row.update(knee=knee, plane=short)
        dest = out / f"cache_{short}" / f"{knee}.npy"
        if dest.exists():
            row["status"] = "cached"
            return row
        # The side comes from the description, not the header: load_slot's own
        # series_side() reads the Laterality tag or the image offset, and OAI's centred
        # knee coil leaves both empty, so right knees went unmirrored (critic, 2026-10-11).
        from preprocess import series_side
        hdr = series_side(head)
        vol, side, _ = load_slot(tmp, size, crop_mm, anchors, plane=plane, normalise_side=True,
                                 band=band, side_override=mt.group(2)[0])
        if not vol.any():
            row["status"] = "blank"
            return row
        row["status"] = f"ok header-side {hdr or 'none'}"
        np.save(dest, vol[None])
        return row
    except Exception as e:
        row["status"] = f"ERROR {type(e).__name__}"
        return row
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--size", type=int, default=288)
    ap.add_argument("--crop-mm", type=float, default=CROP_MM)
    ap.add_argument("--anchors", type=int, default=8)
    ap.add_argument("--band", type=float, nargs=2, default=[0.15, 0.85])
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    for _, short, _ in PLANES.values():
        (args.out / f"cache_{short}").mkdir(parents=True, exist_ok=True)
    # "._*" are macOS metadata files the exFAT drive grows beside every archive
    tgzs = sorted(p for p in args.root.rglob("*.tar.gz") if not p.name.startswith("._"))
    jobs = [(p, args.out, args.size, args.crop_mm, args.anchors, tuple(args.band))
            for p in tgzs]
    print(f"{len(jobs)} series archives under {args.root}", flush=True)
    rows = []
    with Pool(args.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(convert, jobs, chunksize=4), 1):
            rows.append(r)
            if i % 100 == 0:
                print(f"  {i}/{len(jobs)}", flush=True)

    index = args.out / "oai_index.csv"
    old = index.read_text().splitlines()[1:] if index.exists() else []
    seen = {l.split(",")[0] for l in old}
    with index.open("w") as f:
        f.write("archive,knee,plane,status\n")
        for l in old:
            f.write(l + "\n")
        for r in rows:
            if r["archive"] not in seen:
                f.write(",".join(r[c] for c in ("archive", "knee", "plane", "status")) + "\n")
    for plane, short, slot in PLANES.values():
        manifest = {"slots": 1, "size": args.size, "crop_mm": args.crop_mm,
                    "n_anchors": args.anchors, "group": GROUP,
                    "n_slices": args.anchors * GROUP, "band": list(args.band),
                    "laterality": True, "slot_scheme": [[plane, 1]],
                    "only_slot": slot, "split": "oai"}
        (args.out / f"cache_{short}" / "cache_manifest.json").write_text(
            json.dumps(manifest, indent=2))
    from collections import Counter
    print(Counter(r["status"].split(" ")[0] for r in rows))
    print({s: len(list((args.out / f"cache_{s}").glob("*.npy"))) for s in ("sag", "cor", "ax")},
          "knees per plane in the cache")


if __name__ == "__main__":
    main()
