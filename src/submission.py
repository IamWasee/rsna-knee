"""Build and validate submission.csv.

A code competition rejects a malformed file after burning 9 hours of runtime,
so validate() is deliberately strict and runs on every write.

    python src/submission.py --constant 0.5 -o submission.csv
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import ID_COL, LABELS, SUBMISSION_COLUMNS  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


def load_sample() -> pd.DataFrame:
    """Sample submission from the mounted data root, falling back to the repo copy."""
    try:
        from paths import data_root
        p = data_root() / "sample_submission.csv"
        if p.exists():
            return pd.read_csv(p)
    except FileNotFoundError:
        pass
    return pd.read_csv(REPO / "data" / "sample_submission.csv")


def validate(df: pd.DataFrame, expected_ids=None) -> None:
    if list(df.columns) != SUBMISSION_COLUMNS:
        missing = set(SUBMISSION_COLUMNS) - set(df.columns)
        extra = set(df.columns) - set(SUBMISSION_COLUMNS)
        raise ValueError(
            f"column mismatch.\n  expected: {SUBMISSION_COLUMNS}\n  got:      {list(df.columns)}"
            + (f"\n  missing: {sorted(missing)}" if missing else "")
            + (f"\n  extra:   {sorted(extra)}" if extra else "")
            + ("\n  (order matters too)" if not missing and not extra else "")
        )

    if df[ID_COL].duplicated().any():
        dupes = df.loc[df[ID_COL].duplicated(), ID_COL].tolist()[:5]
        raise ValueError(f"duplicate {ID_COL}s, e.g. {dupes}")

    vals = df[LABELS]
    if vals.isna().any().any():
        bad = vals.columns[vals.isna().any()].tolist()
        raise ValueError(f"NaN predictions in columns: {bad}")
    if not vals.map(lambda v: isinstance(v, (int, float))).all().all():
        raise ValueError("non-numeric prediction values")
    if (vals < 0).any().any() or (vals > 1).any().any():
        raise ValueError("predictions outside [0, 1]")

    if expected_ids is not None:
        got, want = set(df[ID_COL]), set(expected_ids)
        if got != want:
            raise ValueError(
                f"id set mismatch: {len(want - got)} missing, {len(got - want)} unexpected"
            )


def build(preds: dict | pd.DataFrame, study_ids=None) -> pd.DataFrame:
    """preds: {study_id: {label: score}} or a DataFrame indexed by study id."""
    if isinstance(preds, dict):
        df = pd.DataFrame.from_dict(preds, orient="index")
    else:
        df = preds.copy()
    df = df.reindex(columns=LABELS)
    df.index.name = ID_COL
    df = df.reset_index()
    validate(df, expected_ids=study_ids)
    return df


def write(df: pd.DataFrame, out: Path = Path("submission.csv")) -> Path:
    validate(df)
    df.to_csv(out, index=False)
    print(f"wrote {out}  ({len(df)} rows x {len(df.columns)} cols)")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--constant", type=float, default=0.5,
                    help="fill every prediction with this value (pipeline smoke test)")
    ap.add_argument("-o", "--out", type=Path, default=Path("submission.csv"))
    args = ap.parse_args()

    sample = load_sample()
    df = sample.copy()
    df[LABELS] = args.constant
    df = df[SUBMISSION_COLUMNS]
    write(df, args.out)


if __name__ == "__main__":
    main()
