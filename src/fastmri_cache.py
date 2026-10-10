"""Build our one-sequence plane caches from the NYU fastMRI clinical knee DICOMs.

fastMRI ships no series table, so each series' plane comes from its geometry
(ImageOrientationPatient) and its fluid sensitivity from its description and
ScanOptions -- fat-suppressed, which is how the competition's Fluid_Sensitive
column is defined. Pixels then go through preprocess.load_slot unchanged, with
the competition caches' constants, so a fastMRI volume is processed exactly as a
competition one: same physical crop, same slice band and triplets, same
laterality normalisation.

Unlike preprocess.pick_series, a missing plane is left missing rather than
filled with some other series: a coronal image in the sagittal slot trains the
sagittal model on the wrong anatomy.

The archive is read as a stream, member by member, and an exam is processed once
the stream has moved past it, so the disk holds a few exams at a time rather
than 134 GB. --shard/--shards splits the exams across notebooks by a hash of the
study UID; each notebook reads the whole stream and keeps its share.

    python src/fastmri_cache.py --src <archive URL or path> --out /kaggle/working/fm \\
        [--shard 0 --shards 4] [--limit 20]

Writes <out>/cache_{sag,cor,ax}/<id>.npy, (1, n_slices, size, size) uint8, one
cache_manifest.json per plane in preprocess.py's format, and fastmri_index.csv
(id, which planes were found, side, scanner, the chosen series descriptions).
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import shutil
import sys
import tarfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from preprocess import CROP_MM, GROUP, load_slot  # noqa: E402

PLANES = [("Sagittal", "sag", 0), ("Coronal", "cor", 1), ("Axial", "ax", 2)]
FAT_SAT = re.compile(r"(?<![A-Z])(FS|FAT\s*-?\s*SAT\w*|FATSAT|STIR|SPAIR|SPIR|TIRM|"
                     r"FS\s*PD|PD\s*FS|T2\s*FS|FS\s*T2|PDFS|T2FS|IDEAL\s*W|DIXON\s*W)"
                     r"(?![A-Z])")
# "SAG PD FSE NON FAT SAT" names the suppression it does NOT use; seen in the
# first fastMRI exams read, where it had been taken as fat-suppressed.
NOT_FAT_SAT = re.compile(r"NON[\s_-]*(FAT|FS)|NO[\s_-]+(FAT|FS)(?![A-Z])|W/?O[\s_-]*(FAT|FS)|"
                         r"WITHOUT[\s_-]*(FAT|FS)")
SKIP = re.compile(r"LOC|SCOUT|SURVEY|CAL|ASSET|PLANE\s*LOC|3\s*PL", re.I)
MIN_SLICES = 10          # below this a series is a localiser, not a stack
FLUSH_AFTER = 400        # members from other studies before an open study is complete


def plane_of(iop) -> str | None:
    """Sagittal / Coronal / Axial from the slice normal's dominant axis."""
    try:
        row = np.array(iop[:3], float)
        col = np.array(iop[3:6], float)
    except (TypeError, ValueError, IndexError):
        return None
    n = np.abs(np.cross(row, col))
    if n.max() < 0.5:          # strongly oblique: none of the three
        return None
    return ("Sagittal", "Coronal", "Axial")[int(n.argmax())]


def is_fat_sat(ds) -> bool:
    desc = " ".join(str(getattr(ds, t, "") or "") for t in
                    ("SeriesDescription", "ProtocolName", "SequenceName")).upper()
    if NOT_FAT_SAT.search(desc):
        return False
    opts = getattr(ds, "ScanOptions", "") or ""
    opts = [opts] if isinstance(opts, str) else list(opts)
    return bool(FAT_SAT.search(desc)) or any(str(o).upper() in ("FS", "SFS", "FSA")
                                              for o in opts)


def study_key(uid: str) -> str:
    return "fm_" + hashlib.sha1(uid.encode()).hexdigest()[:20]


def in_shard(uid: str, shard: int, shards: int) -> bool:
    return int(hashlib.md5(uid.encode()).hexdigest(), 16) % shards == shard


def open_stream(src: str):
    if re.match(r"https?://", src):
        import urllib.request
        return urllib.request.urlopen(src, timeout=120)
    return open(src, "rb")


def process_exam(exam_dir: Path, out: Path, args) -> dict:
    """Choose one fat-suppressed series per plane and write its cache slice stack."""
    import pydicom

    series = []
    for sdir in sorted(p for p in exam_dir.iterdir() if p.is_dir()):
        files = sorted(sdir.glob("*.dcm"))
        if len(files) < MIN_SLICES:
            continue
        try:
            ds = pydicom.dcmread(str(files[0]), stop_before_pixels=True)
        except Exception:
            continue
        desc = str(getattr(ds, "SeriesDescription", "") or "")
        if SKIP.search(desc):
            continue
        p = plane_of(getattr(ds, "ImageOrientationPatient", None))
        if p is None:
            continue
        series.append({"dir": sdir, "plane": p, "fs": is_fat_sat(ds), "n": len(files),
                       "desc": desc})

    row = {"id": exam_dir.name}
    sides, fps = [], []
    for plane, short, _ in PLANES:
        cands = [s for s in series if s["plane"] == plane and s["fs"]]
        if not cands:
            row[short] = ""
            continue
        best = max(cands, key=lambda s: s["n"])
        vol, side, fp = load_slot(best["dir"], args.size, args.crop_mm, args.anchors,
                                  plane=plane, normalise_side=True, band=tuple(args.band))
        if not vol.any():
            row[short] = ""
            continue
        np.save(out / f"cache_{short}" / f"{exam_dir.name}.npy", vol[None])
        row[short] = best["desc"].replace(",", " ")
        if side:
            sides.append(side)
        if fp:
            fps.append(fp)
    row["side"] = max(set(sides), key=sides.count) if sides else ""
    row["scanner"] = (fps[0] if fps else "").replace(",", " ")
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="archive URL or local .tar/.tar.gz path")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--work", type=Path, default=Path("/tmp/fm_work"))
    ap.add_argument("--size", type=int, default=288)
    ap.add_argument("--crop-mm", type=float, default=CROP_MM)
    ap.add_argument("--anchors", type=int, default=8)
    ap.add_argument("--band", type=float, nargs=2, default=[0.15, 0.85])
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0, help="stop after N exams (smoke test)")
    args = ap.parse_args()
    import pydicom

    for _, short, _ in PLANES:
        (args.out / f"cache_{short}").mkdir(parents=True, exist_ok=True)
    shutil.rmtree(args.work, ignore_errors=True)
    args.work.mkdir(parents=True)

    open_exams: dict[str, int] = {}      # study key -> member count when last seen
    rows, seen, members, t0 = [], set(), 0, time.time()
    index = args.out / "fastmri_index.csv"

    def flush(key: str) -> None:
        d = args.work / key
        try:
            rows.append(process_exam(d, args.out, args))
        except Exception as e:                       # one bad exam must not stop the run
            rows.append({"id": key, "sag": "", "cor": "", "ax": "", "side": "",
                         "scanner": f"ERROR {type(e).__name__}"})
        shutil.rmtree(d, ignore_errors=True)
        if len(rows) % 50 == 0:
            el = (time.time() - t0) / 60
            print(f"  {len(rows)} exams, {members} members read, {el:.0f} min", flush=True)

    with open_stream(args.src) as fh, tarfile.open(fileobj=fh, mode="r|*") as tar:
        for m in tar:
            if not m.isfile():
                continue
            members += 1
            data = tar.extractfile(m).read()
            try:
                ds = pydicom.dcmread(io.BytesIO(data), stop_before_pixels=True, force=True)
                uid = str(ds.StudyInstanceUID)
                ser = str(ds.SeriesInstanceUID)
            except Exception:
                continue
            key = study_key(uid)
            if key in seen or not in_shard(uid, args.shard, args.shards):
                continue
            sdir = args.work / key / hashlib.sha1(ser.encode()).hexdigest()[:12]
            sdir.mkdir(parents=True, exist_ok=True)
            (sdir / f"{hashlib.sha1(m.name.encode()).hexdigest()[:16]}.dcm").write_bytes(data)
            open_exams[key] = members
            # An exam is complete once the stream has moved well past it.
            for k in [k for k, last in open_exams.items() if members - last > FLUSH_AFTER]:
                del open_exams[k]
                seen.add(k)
                flush(k)
            if args.limit and len(rows) >= args.limit:
                break
        for k in list(open_exams):
            seen.add(k)
            flush(k)

    with index.open("w") as f:
        f.write("id,sag,cor,ax,side,scanner\n")
        for r in rows:
            f.write(",".join(str(r.get(c, "")) for c in
                             ("id", "sag", "cor", "ax", "side", "scanner")) + "\n")
    for plane, short, slot in PLANES:
        manifest = {"slots": 1, "size": args.size, "crop_mm": args.crop_mm,
                    "n_anchors": args.anchors, "group": GROUP,
                    "n_slices": args.anchors * GROUP, "band": list(args.band),
                    "laterality": True, "slot_scheme": [[plane, 1]],
                    "only_slot": slot, "split": "fastmri"}
        (args.out / f"cache_{short}" / "cache_manifest.json").write_text(
            json.dumps(manifest, indent=2))
    got = {s: sum(1 for r in rows if r.get(s)) for _, s, _ in PLANES}
    print(f"\n{len(rows)} exams in shard {args.shard}/{args.shards} from {members} members, "
          f"{(time.time()-t0)/60:.0f} min; planes found: {got}")


if __name__ == "__main__":
    main()
