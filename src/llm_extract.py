"""Extract the 12 labels from multilingual radiology reports using Claude.

Only 58 of 4,407 studies ship with ground truth. This turns the other 4,349
reports into training labels. It runs once, offline, and the result gets saved as
a Kaggle Dataset -- the submission notebook never calls an API.

The hard part is not finding terms, it's negation: reports are dominated by normal
findings ("Normal ACL, PCL, MCL", "Няма МР данни за ставен излив"), across nine
languages. That is why this is an LLM job and not a regex.

Workflow -- validate before you spend:

    python src/llm_extract.py --data train.csv --validate    # 58 reports, ~$0.50, prints AUC
    python src/llm_extract.py --data train.csv --estimate    # cost of the full run
    python src/llm_extract.py --data train.csv --all         # 4,349 reports via Batches API
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import ID_COL, LABELS  # noqa: E402

MODEL = "claude-opus-5"

# JSON keys must be identifier-safe; LABELS are not (spaces, apostrophe).
KEY = {
    "ACL": "acl", "MCL": "mcl",
    "Medial Meniscus": "medial_meniscus", "Lateral Meniscus": "lateral_meniscus",
    "Medial OA": "medial_oa", "Lateral OA": "lateral_oa", "PF OA": "pf_oa",
    "Effusion": "effusion", "Synovitis": "synovitis", "Baker's": "bakers",
    "Contusion": "contusion", "Fracture": "fracture",
}
UNKEY = {v: k for k, v in KEY.items()}

DEFINITIONS = """\
acl               Anterior cruciate ligament tear, rupture, or partial tear.
mcl               Medial collateral ligament tear or sprain (any grade).
medial_meniscus   Medial meniscus tear, including degenerative tearing.
lateral_meniscus  Lateral meniscus tear, including degenerative tearing.
medial_oa         Medial compartment osteoarthritis: cartilage thinning or loss,
                  osteophytes, subchondral change in the medial femorotibial joint.
lateral_oa        Lateral compartment osteoarthritis, same criteria.
pf_oa             Patellofemoral osteoarthritis, including chondromalacia patellae
                  and retropatellar or trochlear cartilage loss.
effusion          Joint effusion or intra-articular fluid.
synovitis         Synovitis, synovial thickening, or synovial proliferation.
bakers            Baker's cyst / popliteal cyst.
contusion         Bone contusion, bone bruise, or bone marrow oedema from trauma.
fracture          Fracture of any bone, including stress, avulsion, and insufficiency
                  fractures."""

SYSTEM = f"""\
You read knee MRI radiology reports and score 12 findings. Reports come from about
20 institutions worldwide and appear in many languages -- English, Spanish, Turkish,
Croatian, Greek, German, Bulgarian, Dutch, French and others. Read whichever
language the report is in. Do not translate first; read it directly.

The 12 findings:

{DEFINITIONS}

For each finding, return a probability from 0.0 to 1.0 that it is present in this
patient's knee, according to the report.

Rules that matter most, in order:

1. NEGATION IS THE WHOLE TASK. These reports mostly describe normal structures.
   "Normal ACL, PCL, MCL and LCL" means acl=low AND mcl=low. "Няма МР данни за
   ставен излив" (no MR evidence of joint effusion) means effusion=low.
   "Menisco interno y externo sin alteraciones" means both menisci low. A finding
   being *mentioned* is not evidence it is *present* -- often the opposite, since
   radiologists explicitly list the structures they checked and found normal.

2. Report what the knee has, not what the report says loudly. A finding stated in
   the body but omitted from the conclusion is still present.

3. Use the full range. You are scored on ranking, not on binary accuracy, so
   graded confidence beats rounding to 0 and 1:
     0.95  explicitly stated as present
     0.75  strongly implied, or described in different words
     0.5   genuinely ambiguous, or the report does not cover this structure
     0.25  probably absent
     0.05  explicitly stated as absent or normal

4. Silence is not absence. If a report never mentions the patellofemoral joint at
   all, pf_oa is about 0.5, not 0.05. Reserve low scores for findings the
   radiologist actually looked for and ruled out.

5. Laterality matters. Medial and lateral are separate labels for meniscus and for
   osteoarthritis. Do not let one bleed into the other. "Medial compartment
   degeneration" means medial_oa high, lateral_oa low-to-unknown.

6. Grade partial findings as present. A partial ACL tear, a grade 2 chondromalacia,
   a small effusion, a degenerative meniscal tear -- all count as present.

7. [DATE], [NAME], [ID] and similar are de-identification placeholders. Ignore them.
"""

SCHEMA = {
    "type": "object",
    "properties": {k: {"type": "number", "minimum": 0, "maximum": 1} for k in KEY.values()},
    "required": list(KEY.values()),
    "additionalProperties": False,
}


def _params(report: str, effort: str) -> dict:
    """Request params shared by the sync and batch paths."""
    return {
        "model": MODEL,
        "max_tokens": 4000,
        "system": [{"type": "text", "text": SYSTEM, "cache_control": {"type": "ephemeral"}}],
        "messages": [{"role": "user", "content": f"<report>\n{report}\n</report>"}],
        "thinking": {"type": "adaptive"},
        "output_config": {"format": {"type": "json_schema", "schema": SCHEMA}, "effort": effort},
    }


def _parse(message) -> dict[str, float]:
    """Pull the schema-guaranteed JSON out of a response, keyed by competition label."""
    text = next(b.text for b in message.content if b.type == "text")
    raw = json.loads(text)
    return {UNKEY[k]: float(v) for k, v in raw.items() if k in UNKEY}


# --------------------------------------------------------------------------
# Sync path -- for validating against the 58 gold studies
# --------------------------------------------------------------------------
class Fatal(RuntimeError):
    """An error that retrying cannot fix -- no credit, bad key, no access."""


def _fatal_if_hopeless(e: Exception) -> None:
    """Billing and auth failures repeat identically for every report. Stop at the first."""
    import anthropic

    if isinstance(e, (anthropic.AuthenticationError, anthropic.PermissionDeniedError)):
        raise Fatal(f"credentials rejected: {e}") from e
    if isinstance(e, anthropic.BadRequestError) and "credit balance" in str(e):
        raise Fatal(
            "Anthropic account is out of credits.\n"
            "  Add credits at console.anthropic.com -> Plans & Billing, then re-run.\n"
            "  Nothing was charged and no progress was lost."
        ) from e


def extract_one(client, report: str, effort: str = "medium") -> dict[str, float]:
    msg = client.messages.create(**_params(report, effort))
    if msg.stop_reason == "refusal":
        cat = msg.stop_details.category if msg.stop_details else None
        raise RuntimeError(f"refused ({cat})")
    return _parse(msg)


def validate(client, train: pd.DataFrame, effort: str, out: Path | None) -> None:
    """Score the extractor on the 58 labeled studies before spending on the rest."""
    cache: dict[str, dict] = {}
    if out and out.exists():
        cache = {json.loads(l)["id"]: json.loads(l)["pred"] for l in out.open()}
        print(f"resuming: {len(cache)} cached")

    labeled = train[train[LABELS].notna().all(axis=1)]
    fails = 0
    fh = out.open("a") if out else None
    for i, (_, row) in enumerate(labeled.iterrows(), 1):
        sid = row[ID_COL]
        if sid in cache:
            continue
        try:
            pred = extract_one(client, row["Report"], effort)
        except Exception as e:
            _fatal_if_hopeless(e)
            fails += 1
            print(f"  [{i}/{len(labeled)}] failed: {str(e)[:120]}")
            if fails >= 5 and not cache:
                raise Fatal(f"{fails} failures, nothing succeeded -- stopping") from e
            continue
        cache[sid] = pred
        if fh:
            fh.write(json.dumps({"id": sid, "pred": pred}) + "\n")
            fh.flush()
        print(f"  [{i}/{len(labeled)}] ok", end="\r")
    if fh:
        fh.close()

    if not cache:
        print("\nNo extractions succeeded -- nothing to score.")
        return
    print(f"\nextracted {len(cache)}/{len(labeled)}")
    preds = pd.DataFrame([cache.get(s, {}) for s in labeled[ID_COL]], index=labeled.index)
    _report(labeled, preds)


def _report(labeled: pd.DataFrame, preds: pd.DataFrame) -> None:
    from report_labels import macro_auc

    preds = preds.reindex(columns=LABELS).fillna(0.5)
    score, per_label = macro_auc(labeled[LABELS], preds)
    print(f"\n{'label':<18} {'AUC':>6} {'n_pos':>6}")
    print("-" * 32)
    for c in LABELS:
        print(f"{c:<18} {per_label[c]:>6.3f} {int(labeled[c].sum()):>6}")
    print("-" * 32)
    print(f"{'MACRO AUC':<18} {score:>6.3f}  (n={len(labeled)})")
    print("\n58 studies is a smoke test. Treat anything under ~0.85 as a broken prompt,")
    print("and do not tune the prompt to chase the last 0.03 -- that is noise.")


# --------------------------------------------------------------------------
# Batch path -- the 4,349 unlabeled reports
# --------------------------------------------------------------------------
def run_all(client, train: pd.DataFrame, effort: str, out: Path) -> None:
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request

    todo = train[train[LABELS].isna().all(axis=1)]
    print(f"submitting {len(todo)} reports to the Batches API (50% off, up to 24h)")

    batch = client.messages.batches.create(
        requests=[
            Request(
                custom_id=f"s{i}",
                params=MessageCreateParamsNonStreaming(**_params(row["Report"], effort)),
            )
            for i, (_, row) in enumerate(todo.iterrows())
        ]
    )
    print(f"batch id: {batch.id}  -- save this, results live 29 days")
    Path("batch_id.txt").write_text(batch.id)

    while True:
        b = client.messages.batches.retrieve(batch.id)
        if b.processing_status == "ended":
            break
        print(f"  {b.processing_status}: {b.request_counts.processing} processing, "
              f"{b.request_counts.succeeded} done", end="\r")
        time.sleep(60)

    print(f"\ndone: {b.request_counts.succeeded} ok, {b.request_counts.errored} errored")
    collect(client, batch.id, todo, out)


def collect(client, batch_id: str, todo: pd.DataFrame, out: Path) -> None:
    """Results arrive in ANY order -- key by custom_id, never by position."""
    ids = list(todo[ID_COL])
    rows, failed = [], 0
    for result in client.messages.batches.results(batch_id):
        idx = int(result.custom_id[1:])
        if result.result.type != "succeeded":
            failed += 1
            continue
        try:
            pred = _parse(result.result.message)
        except Exception:
            failed += 1
            continue
        rows.append({ID_COL: ids[idx], **pred})

    df = pd.DataFrame(rows).reindex(columns=[ID_COL] + LABELS)
    df.to_csv(out, index=False)
    print(f"wrote {out}: {len(df)} studies, {failed} failed")
    print("\npositive rate at threshold 0.5 (sanity-check against GOLD_PREVALENCE):")
    for c in LABELS:
        print(f"  {c:<18} {(df[c] > 0.5).mean():.3f}")


def estimate(client, train: pd.DataFrame, effort: str) -> None:
    """Price the full run before committing to it."""
    todo = train[train[LABELS].isna().all(axis=1)]
    sample = todo["Report"].iloc[0]
    try:
        n = client.messages.count_tokens(
            model=MODEL,
            system=SYSTEM,
            messages=[{"role": "user", "content": f"<report>\n{sample}\n</report>"}],
        ).input_tokens
    except Exception as e:
        _fatal_if_hopeless(e)
        raise

    sys_tokens = client.messages.count_tokens(
        model=MODEL, system=SYSTEM, messages=[{"role": "user", "content": "x"}]
    ).input_tokens
    per_report = n - sys_tokens

    out_tokens = {"low": 200, "medium": 500, "high": 1200}.get(effort, 500)
    N = len(todo)
    # System prompt is cached: full price once, ~0.1x after.
    in_cost = (sys_tokens * 1.25 + sys_tokens * 0.1 * (N - 1) + per_report * N) / 1e6 * 5.0
    out_cost = out_tokens * N / 1e6 * 25.0
    total = (in_cost + out_cost) * 0.5  # Batches API is 50% off

    print(f"reports:            {N}")
    print(f"system prompt:      {sys_tokens} tokens (cached after first call)")
    print(f"avg report:         ~{per_report} tokens")
    print(f"assumed output:     ~{out_tokens} tokens at effort={effort}")
    print(f"\nestimated cost:     ${total:.2f}  (Opus 5, batch pricing, caching on)")
    print(f"  without batching: ${(in_cost + out_cost):.2f}")
    print("\nOutput tokens dominate. Dropping effort to 'low' roughly halves this;")
    print("validate at both and compare AUC on the 58 before deciding.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True, help="path to train.csv")
    ap.add_argument("--effort", default="medium", choices=["low", "medium", "high"])
    ap.add_argument("--out", type=Path, default=Path("report_labels.csv"))
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--validate", action="store_true", help="58 gold studies only")
    g.add_argument("--estimate", action="store_true", help="price the full run")
    g.add_argument("--all", action="store_true", help="all 4,349 via Batches API")
    g.add_argument("--collect", metavar="BATCH_ID", help="fetch a submitted batch")
    args = ap.parse_args()

    import anthropic
    client = anthropic.Anthropic()
    train = pd.read_csv(args.data)

    try:
        if args.estimate:
            estimate(client, train, args.effort)
        elif args.validate:
            validate(client, train, args.effort, Path("validate_cache.jsonl"))
        elif args.all:
            run_all(client, train, args.effort, args.out)
        else:
            todo = train[train[LABELS].isna().all(axis=1)]
            collect(client, args.collect, todo, args.out)
    except Fatal as e:
        # A stack trace adds nothing here -- the message is the whole story.
        print(f"\nSTOPPED: {e}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
