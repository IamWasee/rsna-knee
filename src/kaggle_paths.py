"""Locate attached Kaggle inputs without walking the DICOM archive.

Every notebook cell needs this and each one re-derived it, so the same mistake --
`glob("/kaggle/input/**/*", recursive=True)` -- was written three separate times.
That glob descends the competition's hundreds of thousands of DICOM files and has
measured 400+ seconds to find a handful of paths.

Two rules make it fast and correct:
  - prune the competition directory from the traversal, not from the results
  - follow symlinks, since Kaggle mounts notebook outputs as links and os.walk
    ignores those by default
"""

from __future__ import annotations

import os

INPUT = "/kaggle/input"
COMP = "rsna-knee-abnormality-detection"


def find(filename: str = "", suffix: str = "", include_comp: bool = False,
         limit: int | None = None) -> list[str]:
    """Paths of attached files matching an exact name or an extension."""
    hits: list[str] = []
    for root, dirs, files in os.walk(INPUT, followlinks=True):
        if not include_comp:
            dirs[:] = [d for d in dirs if COMP not in os.path.join(root, d)]
        for f in files:
            if (filename and f == filename) or (suffix and f.endswith(suffix)):
                hits.append(os.path.join(root, f))
                if limit and len(hits) >= limit:
                    return sorted(hits)
    return sorted(hits)


def competition_file(name: str) -> str:
    """A file inside the mounted competition, whatever level Kaggle mounted it at."""
    for root, _, files in os.walk(INPUT, followlinks=True):
        if COMP in root and name in files:
            return os.path.join(root, name)
    raise FileNotFoundError(f"{name} not found -- is the competition attached?")


def describe() -> None:
    """Print the attached inputs, for when something expected is missing."""
    print("attached inputs:")
    for root, dirs, files in os.walk(INPUT, followlinks=True):
        if COMP in root:
            dirs[:] = []
            print("  " + os.path.relpath(root, INPUT) + "/  (competition data)")
            continue
        depth = root.count(os.sep) - INPUT.count(os.sep)
        if depth > 4:
            dirs[:] = []
            continue
        print("  " * (depth + 1) + os.path.basename(root) + "/")
        for f in files[:6]:
            print("  " * (depth + 2) + f)
