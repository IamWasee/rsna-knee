"""Upload the OAI knees converted since the last batch as a new PRIVATE Kaggle dataset.

Kaggle re-uploads every file on a dataset version, so each batch is its own
dataset (abdullahwasee/oai-knee-b<N>) holding only knees no earlier batch has.
Every batch carries the full oai_labels.csv and split-"oai" manifests, so the
harness (cell 44, EXTRA OAI:N) can attach any number of batches together.
OAI's data use agreement forbids redistribution: datasets are created private
and must stay private.

    python src/oai_cache.py --root /Volumes/USB-189982/oai --out data/oai_cache
    python scripts/oai_upload_batch.py [--min-new 50] [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

OWNER = "abdullahwasee"
CACHE = Path("data/oai_cache")
STAGE = Path("data/oai_upload")
LEDGER = STAGE / "uploaded.json"           # batch -> knee files it holds


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-new", type=int, default=50)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    ledger = json.loads(LEDGER.read_text()) if LEDGER.exists() else {}
    done = {f for files in ledger.values() for f in files}
    new = sorted(f"{p.parent.name}/{p.name}" for p in CACHE.glob("cache_*/*.npy")
                 if f"{p.parent.name}/{p.name}" not in done)
    if len(new) < args.min_new:
        raise SystemExit(f"only {len(new)} new knee files (< {args.min_new}); nothing uploaded")
    n = len(ledger) + 1
    slug = f"oai-knee-b{n}"
    d = STAGE / slug
    shutil.rmtree(d, ignore_errors=True)
    for rel in new:
        (d / rel).parent.mkdir(parents=True, exist_ok=True)
        os.link(CACHE / rel, d / rel)            # hard link: no second copy on disk
    for sub in {r.split("/")[0] for r in new}:
        shutil.copy(CACHE / sub / "cache_manifest.json", d / sub / "cache_manifest.json")
    shutil.copy("data/oai_labels.csv", d / "oai_labels.csv")
    (d / "dataset-metadata.json").write_text(json.dumps({
        "title": slug, "id": f"{OWNER}/{slug}", "licenses": [{"name": "other"}]}))
    per = {s: sum(r.startswith(s) for r in new) for s in ("cache_sag", "cache_cor", "cache_ax")}
    print(f"{slug}: {len(new)} new knee files {per}")
    if args.dry_run:
        return
    # no --public flag: Kaggle creates datasets private by default
    r = subprocess.run(["kaggle", "datasets", "create", "-p", str(d), "--dir-mode", "zip"],
                       capture_output=True, text=True)
    print(r.stdout + r.stderr)
    if r.returncode or "error" in (r.stdout + r.stderr).lower():
        raise SystemExit("upload failed; ledger not updated")
    ledger[slug] = new
    LEDGER.write_text(json.dumps(ledger))
    print(f"recorded {slug}; https://www.kaggle.com/datasets/{OWNER}/{slug}")


if __name__ == "__main__":
    main()
