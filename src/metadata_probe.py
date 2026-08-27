"""Can the 12 findings be predicted from metadata alone, with no pixels?

Acquisition protocol can leak diagnosis: a radiologist who suspects an injury
orders extra sequences, so series count and sequence mix encode clinical
suspicion. If a model trained only on that reaches image-model accuracy, the
signal is not anatomy.

This is a diagnostic, not a submission. It answers one question -- how much of the
achievable score needs no imaging at all -- and it runs in minutes.

    python src/metadata_probe.py --labels report_labels.csv
    python src/metadata_probe.py --labels report_labels.csv --deep   # + DICOM headers
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import ID_COL, LABELS, SERIES_COL  # noqa: E402
from paths import data_root  # noqa: E402


def series_features(series: pd.DataFrame) -> pd.DataFrame:
    """Everything derivable from train_series.csv -- no pixels, no headers."""
    g = series.groupby(ID_COL)
    f = pd.DataFrame(index=g.size().index)
    f["n_series"] = g.size()
    for plane in ("Sagittal", "Coronal", "Axial"):
        f[f"n_{plane.lower()}"] = series[series["Anatomical_Plane"] == plane] \
            .groupby(ID_COL).size().reindex(f.index).fillna(0)
    f["n_fluid"] = series[series["Fluid_Sensitive"] == 1].groupby(ID_COL).size() \
        .reindex(f.index).fillna(0)
    f["frac_fluid"] = f["n_fluid"] / f["n_series"]
    f["n_fatsat"] = series[series["Fat_Suppression"] == 1].groupby(ID_COL).size() \
        .reindex(f.index).fillna(0)
    # The exact plane/sequence recipe as one categorical -- protocols are recipes.
    f["protocol"] = g.apply(
        lambda d: hash(tuple(sorted(zip(d["Anatomical_Plane"], d["Fluid_Sensitive"]))))
        % 100000, include_groups=False)
    return f


def header_features(series: pd.DataFrame, root: Path, split: str,
                    limit: int | None = None) -> pd.DataFrame:
    """One DICOM header per series. Scanner, timing, geometry -- still no pixels."""
    import pydicom

    rows = {}
    studies = list(series.groupby(ID_COL).groups)[:limit]
    for n, sid in enumerate(studies):
        sub = series[series[ID_COL] == sid]
        vals: dict[str, list] = {}
        n_inst = []
        for _, r in sub.iterrows():
            d = root / f"{split}_series" / sid / r[SERIES_COL]
            files = sorted(d.glob("*.dcm"))
            n_inst.append(len(files))
            if not files:
                continue
            try:
                ds = pydicom.dcmread(str(files[0]), stop_before_pixels=True)
            except Exception:
                continue
            for tag in ("Manufacturer", "MagneticFieldStrength", "SliceThickness",
                        "Rows", "Columns", "RepetitionTime", "EchoTime",
                        "SpacingBetweenSlices", "PixelBandwidth", "FlipAngle"):
                v = getattr(ds, tag, None)
                if v is not None:
                    vals.setdefault(tag, []).append(v)
        rec = {"total_slices": int(np.sum(n_inst)) if n_inst else 0,
               "max_slices": int(np.max(n_inst)) if n_inst else 0,
               "min_slices": int(np.min(n_inst)) if n_inst else 0}
        for tag, vs in vals.items():
            try:
                rec[tag] = float(np.mean([float(v) for v in vs]))
            except (TypeError, ValueError):
                rec[tag] = float(hash(str(Counter(vs).most_common(1)[0][0])) % 1000)
        rows[sid] = rec
        if n % 200 == 0:
            print(f"  headers {n}/{len(studies)}", flush=True)
    return pd.DataFrame.from_dict(rows, orient="index")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", type=Path, required=True, help="report_labels.csv")
    ap.add_argument("--deep", action="store_true", help="also read DICOM headers")
    ap.add_argument("--limit", type=int, help="cap studies when reading headers")
    args = ap.parse_args()

    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score

    root = data_root()
    train = pd.read_csv(root / "train.csv")
    series = pd.read_csv(root / "train_series.csv")

    gold = train[train[LABELS].notna().all(axis=1)].set_index(ID_COL)
    derived = pd.read_csv(args.labels).set_index(ID_COL)
    derived = derived[~derived.index.isin(gold.index)]

    X = series_features(series)
    print(f"series features: {list(X.columns)}")
    if args.deep:
        H = header_features(series, root, "train", args.limit)
        X = X.join(H, how="left")
        print(f"+ header features: {list(H.columns)}")

    Xd = X.reindex(derived.index).astype(float)
    Xg = X.reindex(gold.index).astype(float)

    print(f"\ntraining on {len(Xd)} studies, testing on {len(Xg)} gold\n")
    print(f"{'label':<18} {'AUC':>6}")
    print("-" * 26)
    aucs = []
    for c in LABELS:
        y = (derived[c] > 0.5).astype(int)
        if y.nunique() < 2:
            print(f"{c:<18} {'--':>6}")
            continue
        m = HistGradientBoostingClassifier(max_iter=250, learning_rate=0.06,
                                           max_depth=4, random_state=0)
        m.fit(Xd, y)
        p = m.predict_proba(Xg)[:, 1]
        t = gold[c].astype(int)
        a = roc_auc_score(t, p) if t.nunique() > 1 else float("nan")
        aucs.append(a)
        print(f"{c:<18} {a:>6.3f}")
    print("-" * 26)
    print(f"{'MACRO AUC':<18} {np.mean(aucs):>6.3f}   (metadata only, no pixels)")
    print("\nRead against the image model's GOLD score. Close to it means most of what")
    print("the image model learned was protocol, not anatomy. Near 0.5 means the")
    print("metadata carries nothing and the shortcut, if it exists, is elsewhere.")


if __name__ == "__main__":
    main()
