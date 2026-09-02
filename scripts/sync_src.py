"""Publish src/ as a Kaggle dataset so a submission notebook needs no internet.

Code competitions require internet OFF, which rules out the `git clone` every
other cell in this repo starts with. The submission notebook attaches this
dataset instead. Run it after any change to src/ that a submission depends on --
a stale dataset is a submission running last week's inference code.

    python scripts/sync_src.py -m "restore rank_average"
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

SLUG = "abdullahwasee/rsna-knee-src"
SRC = Path(__file__).resolve().parent.parent / "src"


def fingerprint(d: Path) -> str:
    h = hashlib.sha256()
    for f in sorted(d.glob("*.py")):
        h.update(f.name.encode())
        h.update(f.read_bytes())
    return h.hexdigest()[:12]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--message", default="sync src/")
    args = ap.parse_args()

    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        for f in SRC.glob("*.py"):
            shutil.copy(f, d / f.name)
        fp = fingerprint(d)
        # Stamped into the dataset so a notebook can print which src/ it is running
        # and be compared against the local tree, rather than assumed current.
        (d / "SRC_VERSION.txt").write_text(
            f"{fp}\n{subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], cwd=SRC.parent, capture_output=True, text=True).stdout.strip()}\n")
        (d / "dataset-metadata.json").write_text(json.dumps(
            {"title": "rsna-knee-src", "id": SLUG,
             "licenses": [{"name": "CC0-1.0"}]}, indent=2))
        r = subprocess.run(
            ["kaggle", "datasets", "version", "-p", str(d), "-m",
             f"{args.message} [{fp}]", "--dir-mode", "zip"],
            capture_output=True, text=True)
        sys.stdout.write(r.stdout)
        sys.stderr.write(r.stderr)
        if r.returncode:
            raise SystemExit(r.returncode)
        print(f"\nsrc fingerprint {fp} -> https://www.kaggle.com/datasets/{SLUG}")


if __name__ == "__main__":
    main()
