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
    ap.add_argument("--no-internet", action="store_true",
                    help="required for a submission notebook; also rules out git clone")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not args.cell.exists():
        raise SystemExit(f"no such driver: {args.cell}")

    # Refuse the one mistake this project keeps making. A recursive glob rooted at
    # /kaggle/input descends 24,371 DICOM series -- measured at 400+ seconds per
    # call -- and it has been written into five separate cells now, the last of
    # which ran for an hour before anyone noticed. kaggle_paths.find() prunes the
    # competition tree and is the only correct way to locate an attached file.
    # Comments are stripped first: this file and the cells both describe the
    # anti-pattern in prose, and a guard that fires on its own documentation is a
    # guard people switch off.
    body = "\n".join(l for l in args.cell.read_text().splitlines()
                     if not l.lstrip().startswith("#"))
    import re as _re
    bad = _re.search(r"glob\([^)]*/kaggle/input[^)]*\*\*", body) or (
        "recursive=True" in body and "/kaggle/input" in body)
    # Only shell-magic lines and subprocess calls actually reach the network. An
    # earlier version matched the bare text and rejected a cell whose own error
    # message told the reader not to add a git clone.
    if args.no_internet:
        net = [l.strip() for l in body.splitlines()
               if (l.lstrip().startswith(("!", "%")) or "subprocess" in l)
               and ("git clone" in l or "pip install" in l or "wget" in l
                    or "curl" in l)]
        if net:
            raise SystemExit(
                f"{args.cell.name} reaches the network, but --no-internet is set:\n"
                + "\n".join(f"    {l}" for l in net[:5])
                + "\nA submission notebook has no network. Attach "
                  "abdullahwasee/rsna-knee-src and import from it."
            )
    if bad:
        raise SystemExit(
            f"{args.cell.name} recursively globs /kaggle/input.\n"
            "That walks the whole DICOM archive (400+ s per call).\n"
            "Use: from kaggle_paths import find, competition_file, describe"
        )

    meta = {
        "id": f"{OWNER}/{args.slug}",
        "title": args.slug,
        "code_file": f"{args.slug}.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": args.gpu,
        "enable_tpu": False,
        "enable_internet": not args.no_internet,
        "keywords": ["gpu"] if args.gpu else [],
        "dataset_sources": args.dataset,
        "kernel_sources": [k if "/" in k else f"{OWNER}/{k}" for k in args.kernel],
        "competition_sources": [] if args.no_competition else [COMP],
        "model_sources": args.model,
        "docker_image": IMAGE,
        "machine_shape": "NvidiaTeslaT4" if args.gpu else "None",
    }

    # Syntax-check the driver with the shell magics removed. A cell that does not
    # parse costs a queue slot and however long Kaggle takes to reach it, and the
    # `!cmd \` line-continuations mean a naive strip leaves dangling arguments.
    lines, py = body.splitlines(), []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.lstrip().startswith("!"):
            while line.rstrip().endswith("\\") and i + 1 < len(lines):
                i += 1
                line = lines[i]
        else:
            py.append(line)
        i += 1
    try:
        import ast as _ast
        _ast.parse("\n".join(py))
    except SyntaxError as e:
        raise SystemExit(f"{args.cell.name} does not parse: line {e.lineno}: {e.msg}")

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
