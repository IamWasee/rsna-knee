"""Labels from Claude reading the reports in-session, graded by the host's thresholds.

No API key: Claude Code subagents read batches of reports and record, for each of
the twelve findings, WHAT the report says about it -- not a yes/no. A fixed
mapping below then turns those facts into a score using the host's published
criteria (research.md section 1.2). Splitting reading from grading means the
grading can be audited line by line and is committed before any gold scoring.

    python src/claude_labels.py prepare data/audit/train.csv <outdir> [--gold|--rest] [--per 60]
    # subagents: read SPEC below + a batch file, write <batch>.out.json
    python src/claude_labels.py build data/audit/train.csv <outdir> <labels.csv>

Each record is {"id": ..., "<Label>": [status, grade, certainty, acuity, kind]}
for all twelve labels; status "n" and "s" need nothing after them.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import ID_COL, LABELS  # noqa: E402

SPEC = """\
You are reading knee MRI radiology reports (English, Spanish, Turkish, Croatian,
Greek, German, Bulgarian, Dutch, French). For each report, record what it SAYS
about each of twelve findings. Do not judge what is clinically significant --
record the report's own words as categories. You have no answers to check
against; there is nothing to optimise, only to read carefully.

The twelve findings:
  ACL               anterior cruciate ligament tear or injury
  MCL               medial collateral ligament tear, sprain or injury
  Medial Meniscus   tear or signal abnormality of the medial meniscus
  Lateral Meniscus  tear or signal abnormality of the lateral meniscus
  Medial OA         cartilage loss / osteoarthritis, medial femorotibial compartment
  Lateral OA        cartilage loss / osteoarthritis, lateral femorotibial compartment
  PF OA             cartilage loss / osteoarthritis, patellofemoral (patella, trochlea)
  Effusion          joint effusion (fluid in the joint / suprapatellar recess)
  Synovitis         synovitis, synovial thickening or hypertrophy
  Baker's           Baker's / popliteal cyst
  Contusion         post-traumatic bone bruise / contusion. Marrow oedema from
                    arthritis, stress or a fracture's own margin is NOT a contusion.
  Fracture          fracture of any kind

For each finding give a list:
  status     "n" = explicitly normal/absent ("ACL intact", "no effusion")
             "s" = not mentioned at all
             "p" = present
  For "p" only, four more fields:
  grade      0 = present but size/grade not stated
             1 = trace / minimal / tiny
             2 = small / mild / low-grade / grade 1 or 2 sprain / interstitial or
                 intrasubstance signal / partial-thickness (<50%) cartilage loss /
                 "early" or "incipient" OA / meniscal signal NOT reaching the surface
             3 = moderate
             4 = large / severe / marked / high-grade / complete / full-thickness /
                 grade 3-4 / meniscal tear reaching the surface, displaced or complex
  certainty  "d" definite  "p" probable ("likely", "compatible with", "in favor of")
             "h" hedged ("possible", "suspected", "cannot exclude", "r/o", "may")
  acuity     "a" acute/recent  "c" chronic/old/healed/remote  "u" not stated
  kind       Fracture only: "l" clear fracture line or cortical break,
             "i" impaction / osteochondral / subchondral, "x" stress or
             insufficiency, "v" avulsion, "u" not stated. Others: "-".

If a finding is described more than once, use the most specific description.
If the report contradicts itself, prefer the Impression/Conclusion.

Output: a JSON list, one object per report, keys "id" plus all twelve findings,
e.g. {"id": "1.2.3", "ACL": ["p", 4, "d", "a", "-"], "MCL": ["n"], ...,
"Fracture": ["s"]}. No commentary.
"""

# ---- grading: the host's criteria, fixed before any gold scoring -----------
# Scores are ordinal: >0.5 means the host's criterion is met, and within each
# side a more severe or more certain finding ranks higher. Silence ranks above
# an explicit negative, below any mention.
ABSENT, SILENT = 0.02, 0.20
BY_GRADE = {  # grade: 4, 3, 0 (unstated), 2, 1
    # moderate or large amount
    "Effusion": (0.95, 0.85, 0.60, 0.40, 0.32),
    "Baker's": (0.95, 0.85, 0.60, 0.40, 0.32),
    # high-grade partial or complete; low-grade and signal change negative
    "ACL": (0.95, 0.80, 0.75, 0.40, 0.32),
    "MCL": (0.95, 0.80, 0.75, 0.40, 0.32),
    # moderate or large area of high-grade (>50%) cartilage loss
    "Medial OA": (0.95, 0.75, 0.65, 0.40, 0.32),
    "Lateral OA": (0.95, 0.75, 0.65, 0.40, 0.32),
    "PF OA": (0.95, 0.75, 0.65, 0.40, 0.32),
    # signal reaching the surface or morphologic change; intrasubstance negative
    "Medial Meniscus": (0.95, 0.90, 0.85, 0.35, 0.30),
    "Lateral Meniscus": (0.95, 0.90, 0.85, 0.35, 0.30),
    # no published threshold: any definite mention counts, size only orders it
    "Synovitis": (0.90, 0.85, 0.80, 0.65, 0.55),
    "Contusion": (0.90, 0.90, 0.85, 0.80, 0.70),
}
FRACTURE_KIND = {"l": 0.95, "u": 0.85, "v": 0.80, "i": 0.70, "x": 0.45}
GRADE_SLOT = {4: 0, 3: 1, 0: 2, 2: 3, 1: 4}


def grade(label: str, rec: list) -> float:
    if not rec or rec[0] == "s":
        return SILENT
    if rec[0] == "n":
        return ABSENT
    g, cert, acu, kind = (list(rec[1:]) + [0, "d", "u", "-"])[:4]
    try:
        g = int(g)
    except (TypeError, ValueError):
        g = 0
    if label == "Fracture":
        v = FRACTURE_KIND.get(kind, 0.85)
    else:
        v = BY_GRADE[label][GRADE_SLOT.get(g, 2)]
    # acute only: "chronic or remote stress changes are negative" (fracture, MCL)
    if label in ("Fracture", "MCL") and acu == "c":
        v = min(v, 0.30)
    # on the fence is negative
    if cert == "h":
        v = min(v, 0.45)
    elif cert == "p":
        v *= 0.95
    return round(v, 4)


def prepare(train_csv: str, outdir: str, which: str = "--gold", per: int = 60) -> None:
    t = pd.read_csv(train_csv)
    gold = t[LABELS].notna().all(axis=1)
    rows = t[gold] if which == "--gold" else t[~gold]
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    recs = [{"id": r[ID_COL], "report": str(r["Report"])} for _, r in rows.iterrows()]
    n = 0
    for i in range(0, len(recs), per):
        n += 1
        (out / f"batch_{n:03d}.json").write_text(
            json.dumps(recs[i:i + per], ensure_ascii=False), encoding="utf-8")
    (out / "SPEC.txt").write_text(SPEC, encoding="utf-8")
    print(f"{len(recs)} reports -> {n} batches of <= {per} in {out}")


def build(train_csv: str, outdir: str, labels_csv: str) -> None:
    t = pd.read_csv(train_csv)
    got, bad = {}, []
    for f in sorted(Path(outdir).glob("batch_*.out.json")):
        for r in json.loads(f.read_text(encoding="utf-8")):
            if not isinstance(r, dict) or "id" not in r:
                bad.append(f.name)
                continue
            got[r["id"]] = {c: grade(c, r.get(c, ["s"])) for c in LABELS}
    missing = sorted(set(t[ID_COL]) - set(got))
    df = pd.DataFrame([{ID_COL: k, **v} for k, v in got.items()])
    df.to_csv(labels_csv, index=False)
    print(f"{len(got)} studies graded -> {labels_csv}; "
          f"{len(missing)} of train.csv not read; {len(bad)} malformed records")


if __name__ == "__main__":
    cmd, *a = sys.argv[1:]
    if cmd == "prepare":
        prepare(a[0], a[1], *(a[2:3] or ["--gold"]),
                **({"per": int(a[a.index("--per") + 1])} if "--per" in a else {}))
    elif cmd == "build":
        build(*a[:3])
    else:
        raise SystemExit(f"unknown command {cmd}")
