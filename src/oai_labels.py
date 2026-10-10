"""Soft targets for OAI baseline knees from their MOAKS semi-quantitative reads.

MOAKS grades nine of our twelve findings, by a musculoskeletal radiologist on the
image itself rather than from a report. MCL, Contusion and Fracture are left
blank -- MOAKS has no MCL grade, and its bone marrow lesions in an osteoarthritis
cohort are degenerative, not the traumatic bruise the competition label means.
train.py --extra-labels gives a blank cell zero weight.

Grades become probabilities on a coarse ladder, roughly what a report-reading
labeller would say about the same knee: a grade a radiologist would not mention
stays low, one they would call moderate or worse goes high. A knee read by
several MOAKS projects gets the mean of its reads.

    python src/oai_labels.py --moaks <kMRI_SQ_MOAKS_BICL00.txt> --out data/oai_labels.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import ID_COL, LABELS  # noqa: E402

P = "V00M"
MEN = {"Medial Meniscus": ("MTMA", "MTMB", "MTMP", "RTM"),
       "Lateral Meniscus": ("MTLA", "MTLB", "MTLP", "RTL")}
# cartilage (CM*) and osteophyte (OS*) subregions per compartment; in MOAKS the
# femoral "anterior" subregions FMA/FLA are the trochlea, i.e. patellofemoral
OA = {"Medial OA": (("CMFMC", "CMFMP", "CMTMA", "CMTMC", "CMTMP"),
                    ("OSFMC", "OSFMP", "OSTM")),
      "Lateral OA": (("CMFLC", "CMFLP", "CMTLA", "CMTLC", "CMTLP"),
                     ("OSFLC", "OSFLP", "OSTL")),
      "PF OA": (("CMPM", "CMPL", "CMFMA", "CMFLA"),
                ("OSPS", "OSPI", "OSPM", "OSPL", "OSFMA", "OSFLA"))}
LADDER = {"ACL": ("ACLTR", {0: .03, 1: .6, 2: .97}),
          "Effusion": ("EFFWK", {0: .05, 1: .4, 2: .85, 3: .95}),
          "Synovitis": ("SYIC", {0: .05, 1: .35, 2: .8, 3: .95}),
          "Baker's": ("POPCYS", {0: .05, 1: .9})}


def code(s: pd.Series) -> pd.Series:
    """'2.2: 10-75% area, ...' -> 2.2; '.: Missing ...' -> NaN."""
    return pd.to_numeric(s.astype(str).str.split(":").str[0].str.strip(), errors="coerce")


def meniscus(d: pd.DataFrame, cols) -> pd.Series:
    tear = [code(d[P + c]) for c in cols[:3]]
    morph = pd.concat(tear, axis=1)
    root = code(d[P + "M" + cols[3]])
    # 0 normal, 1 intrasubstance signal (not a tear), 2-8 tear or maceration
    worst = morph.max(axis=1)
    p = pd.Series(np.nan, index=d.index)
    p[worst == 0] = .05
    p[worst == 1] = .15
    p[worst >= 2] = .92
    if root is not None:
        p[root == 1] = .95
    return p


def oa(d: pd.DataFrame, cart, ost) -> pd.Series:
    c = pd.concat([code(d[P + x]) for x in cart], axis=1)
    area, full = np.floor(c).max(axis=1), ((c * 10).round() % 10).max(axis=1)
    o = pd.concat([code(d[P + x]) for x in ost], axis=1).max(axis=1)   # often unread
    o = o.fillna(0)
    p = pd.Series(.05, index=d.index)
    p[area >= 1] = .2
    p[(area >= 2) | (o >= 1)] = .5
    p[((area >= 2) & (full >= 1)) | (o >= 2) | (area >= 3)] = .85
    p[area.isna()] = np.nan
    return p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--moaks", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    d = pd.read_csv(args.moaks, sep="|", encoding="latin-1", dtype=str)
    side = d["SIDE"].astype(str).str.strip().str[0].map({"1": "R", "2": "L"})
    out = pd.DataFrame({ID_COL: "oai_" + d["ID"].astype(str).str.strip() + "_" + side})
    for lab, cols in MEN.items():
        out[lab] = meniscus(d, cols)
    for lab, (cart, ost) in OA.items():
        out[lab] = oa(d, cart, ost)
    for lab, (col, ladder) in LADDER.items():
        out[lab] = code(d[P + col]).map(ladder)
    for lab in ("MCL", "Contusion", "Fracture"):
        out[lab] = np.nan
    out = out[side.notna()]
    out = out.groupby(ID_COL)[LABELS].mean().reset_index()    # several reads -> mean
    out.to_csv(args.out, index=False)
    print(f"{len(out)} knees -> {args.out}")
    print("share graded / mean target per finding:")
    print(pd.DataFrame({"graded": out[LABELS].notna().mean().round(2),
                        "mean": out[LABELS].mean().round(3)}).to_string())


if __name__ == "__main__":
    main()
