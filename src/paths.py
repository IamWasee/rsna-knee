"""Resolve the competition data root across Kaggle, Colab, and local runs.

The dataset never lives in this repo -- it is far too large. This module finds
it wherever the current environment happens to have mounted it.
"""

import os
from pathlib import Path

COMP = "rsna-knee-abnormality-detection"

CANDIDATES = [
    Path(f"/kaggle/input/{COMP}"),          # Kaggle notebook (code competition runtime)
    Path(f"/content/{COMP}"),               # Colab, unzipped locally
    Path(f"/content/drive/MyDrive/{COMP}"), # Colab via Drive
    Path(__file__).resolve().parents[1] / "data" / COMP,  # local (unlikely: needs ~100s of GB)
]


def data_root() -> Path:
    """Return the first existing data root, or raise with actionable guidance."""
    env = os.environ.get("RSNA_KNEE_DATA")
    if env:
        p = Path(env)
        if p.exists():
            return p
        raise FileNotFoundError(f"RSNA_KNEE_DATA={env} does not exist")

    for p in CANDIDATES:
        if p.exists():
            return p

    raise FileNotFoundError(
        "Competition data not found. Searched:\n  "
        + "\n  ".join(str(p) for p in CANDIDATES)
        + "\n\nRun this on Kaggle (data is pre-mounted) or set RSNA_KNEE_DATA=<path>."
    )


def in_kaggle() -> bool:
    return Path("/kaggle/input").exists()
