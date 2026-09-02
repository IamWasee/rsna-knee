"""Push one of the scripts/kaggle_cell_*.py drivers to Kaggle as a notebook.

The drivers are kept as .py because that is what is reviewable in a diff, but
Kaggle wants a .ipynb. Each driver becomes a SINGLE cell on purpose: a
`raise SystemExit` in a guard then aborts everything after it, where split cells
would sail past the guard and run the expensive part anyway.

    python scripts/push_kernel.py scripts/kaggle_cell_19_cache_v3.py cache-v3
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

OWNER = "abdullahwasee"
COMP = "rsna-knee-abnormality-detection"
# The image the 0.814 DINOv2 run was trained in. Pinned so a Kaggle-side image
# bump cannot change results between two runs we intend to compare.
IMAGE = ("gcr.io/kaggle-private-byod/python@sha256:"
         "37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461")


def notebook(source: str) -> dict:
    return {
        "cells": [{"cell_type": "code", "execution_count": None,
                   "metadata": {}, "outputs": [], "source": source.splitlines(True)}],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                    "name": "python3"},
                     "language_info": {"name": "python"}},
        "nbformat": 4, "nbformat_minor": 5,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cell", type=Path)
    ap.add_argument("slug")
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--dataset", action="append", default=[])
    ap.add_argument("--model", action="append", default=[])
    ap.add_argument("--kernel", action="append", default=[],
                    help="another kernel whose OUTPUT this one reads")
    ap.add_argument("--no-competition", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not args.cell.exists():
        raise SystemExit(f"no such driver: {args.cell}")

    meta = {
        "id": f"{OWNER}/{args.slug}",
        "title": args.slug,
        "code_file": f"{args.slug}.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": args.gpu,
        "enable_tpu": False,
        "enable_internet": True,
        "keywords": ["gpu"] if args.gpu else [],
        "dataset_sources": args.dataset,
        "kernel_sources": [k if "/" in k else f"{OWNER}/{k}" for k in args.kernel],
        "competition_sources": [] if args.no_competition else [COMP],
        "model_sources": args.model,
        "docker_image": IMAGE,
        "machine_shape": "NvidiaTeslaT4" if args.gpu else "None",
    }

    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        (d / f"{args.slug}.ipynb").write_text(json.dumps(notebook(args.cell.read_text())))
        (d / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
        print(json.dumps(meta, indent=2))
        if args.dry_run:
            print("\n(dry run -- nothing pushed)")
            return
        r = subprocess.run(["kaggle", "kernels", "push", "-p", str(d)],
                           capture_output=True, text=True)
        sys.stdout.write(r.stdout)
        sys.stderr.write(r.stderr)
        if r.returncode:
            raise SystemExit(r.returncode)
        print(f"\nhttps://www.kaggle.com/code/{OWNER}/{args.slug}")


if __name__ == "__main__":
    main()
