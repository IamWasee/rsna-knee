"""Extract the 12 labels from reports using an open-weights model on a local GPU.

Free alternative to llm_extract.py: runs entirely inside a Kaggle notebook on the
provided GPU, so there is no API, no cost, and nothing external to reproduce when
the code is open-sourced.

Same contract as llm_extract.py -- reads train.csv, writes report_labels.csv with
a probability per label per study, and validates against the 58 gold studies first.

    python src/local_extract.py --data train.csv --validate   # 58 studies, ~2 min
    python src/local_extract.py --data train.csv --all        # 4,349 studies

Model choice matters more than anything else here. The task is negation detection
across nine languages, so multilingual instruction-following is what to optimise
for, not size. Try a couple and compare on the 58.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import ID_COL, LABELS  # noqa: E402
from llm_extract import DEFINITIONS, KEY, UNKEY  # noqa: E402

DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"

PROMPT = f"""\
You read knee MRI radiology reports and score 12 findings. Reports come from many
countries and appear in many languages -- English, Spanish, Turkish, Croatian,
Greek, German, Bulgarian, Dutch, French. Read the report in whatever language it
is written in.

The 12 findings:

{DEFINITIONS}

CRITICAL: these reports mostly describe NORMAL structures. "Normal ACL, PCL, MCL"
means acl and mcl are ABSENT. "No evidence of joint effusion" means effusion is
ABSENT. A finding being mentioned is not evidence it is present -- radiologists
list the structures they checked and found normal.

Score each finding 0.0 to 1.0:
  0.95 explicitly present    0.75 implied or described in other words
  0.50 not mentioned at all  0.25 probably absent
  0.05 explicitly normal or ruled out

If a structure is never mentioned, use 0.5 -- not 0.05. Only score low when the
radiologist actually looked and found nothing.

Medial and lateral are separate labels. Do not let one bleed into the other.

Reply with ONLY a JSON object, no other text:
{{{", ".join(f'"{k}": 0.5' for k in KEY.values())}}}"""

NEUTRAL = {label: 0.5 for label in LABELS}


def load_model(model_id: str, four_bit: bool):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    kwargs = {"dtype": torch.float16, "device_map": "auto"}
    if four_bit:
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
        )
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    model.eval()
    return tok, model


def parse_scores(text: str) -> dict[str, float]:
    """Open models wrap JSON in prose and markdown. Take the last {...} and be forgiving."""
    blocks = re.findall(r"\{[^{}]*\}", text, flags=re.DOTALL)
    for block in reversed(blocks):
        try:
            raw = json.loads(block)
        except json.JSONDecodeError:
            continue
        out = {}
        for k, v in raw.items():
            key = str(k).strip().lower().replace(" ", "_").replace("'", "")
            if key in UNKEY:
                try:
                    out[UNKEY[key]] = min(1.0, max(0.0, float(v)))
                except (TypeError, ValueError):
                    continue
        if out:
            return {**NEUTRAL, **out}
    return dict(NEUTRAL)


def generate(tok, model, reports: list[str], batch_size: int, max_new: int = 160,
             checkpoint=None):
    """Batched greedy decoding. Greedy because we want the same answer every run."""
    import torch

    outs = []
    for i in range(0, len(reports), batch_size):
        chunk = reports[i:i + batch_size]
        prompts = [
            tok.apply_chat_template(
                [{"role": "system", "content": PROMPT},
                 {"role": "user", "content": f"<report>\n{r}\n</report>"}],
                tokenize=False, add_generation_prompt=True,
            )
            for r in chunk
        ]
        enc = tok(prompts, return_tensors="pt", padding=True,
                  truncation=True, max_length=3072).to(model.device)
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
        for j in range(len(chunk)):
            outs.append(tok.decode(gen[j][enc["input_ids"].shape[1]:],
                                   skip_special_tokens=True))
        # One line per flush, not per batch. A \r-per-batch progress line is fine
        # in a terminal but Jupyter appends rather than overwrites, and thousands
        # of them freeze the browser tab.
        if (i // batch_size) % 20 == 19:
            n = min(i + batch_size, len(reports))
            print(f"  {n}/{len(reports)}  ({n / len(reports):.0%})", flush=True)
            if checkpoint:
                _flush(checkpoint, outs)
    print()
    return outs


def _flush(checkpoint, outs: list[str]) -> None:
    args, todo, done = checkpoint
    rows = dict(done)
    for sid, text in zip(todo[ID_COL].iloc[:len(outs)], outs):
        rows[sid] = parse_scores(text)
    df = pd.DataFrame.from_dict(rows, orient="index").reindex(columns=LABELS)
    df.index.name = ID_COL
    df.reset_index().to_csv(args.out, index=False)


def run(args) -> None:
    from report_labels import macro_auc

    train = pd.read_csv(args.data)
    labeled = train[train[LABELS].notna().all(axis=1)]
    todo = labeled if args.validate else train[train[LABELS].isna().all(axis=1)]

    # Resume. The full run is 4,349 reports -- several hours against a 9-hour
    # session cap -- so losing everything to a timeout is a real failure mode,
    # not a hypothetical one.
    done: dict[str, dict] = {}
    if args.out.exists():
        prev = pd.read_csv(args.out).set_index(ID_COL)
        done = {i: r.to_dict() for i, r in prev.iterrows()}
        todo = todo[~todo[ID_COL].isin(done)]
        print(f"resuming: {len(done)} already done, {len(todo)} remaining")

    print(f"{len(todo)} reports, model={args.model}, batch={args.batch}")

    if todo.empty:
        print("nothing left to do -- rescoring the cached predictions")
        preds = pd.DataFrame(columns=LABELS)
    else:
        tok, model = load_model(args.model, args.four_bit)
        texts = generate(tok, model, list(todo["Report"].astype(str)), args.batch,
                         max_new=args.max_new, checkpoint=(args, todo, done))
        preds = pd.DataFrame([parse_scores(t) for t in texts], index=todo.index)

        # A high neutral rate means the model is not answering, not that it is unsure.
        neutral = (preds[LABELS] == 0.5).all(axis=1).mean()
        print(f"unparseable responses: {neutral:.1%}")
        if neutral > 0.3:
            print("  >>> most responses failed to parse. Check a raw output:")
            print("  " + texts[0][:400].replace("\n", "\n  "))

    rows = dict(done)
    for sid, row in zip(todo[ID_COL], preds.to_dict("records")):
        rows[sid] = row
    out = pd.DataFrame.from_dict(rows, orient="index").reindex(columns=LABELS)
    out.index.name = ID_COL
    out.reset_index().to_csv(args.out, index=False)
    print(f"wrote {args.out}: {len(out)} studies")

    if args.validate:
        preds = out.reindex(labeled[ID_COL]).reset_index(drop=True)
        preds.index = labeled.index
        score, per_label = macro_auc(labeled[LABELS], preds[LABELS])
        print(f"\n{'label':<18} {'AUC':>6} {'n_pos':>6}")
        print("-" * 32)
        for c in LABELS:
            print(f"{c:<18} {per_label[c]:>6.3f} {int(labeled[c].sum()):>6}")
        print("-" * 32)
        print(f"{'MACRO AUC':<18} {score:>6.3f}  (n={len(labeled)})")
        print("\nCompare against the keyword baseline, then ensemble.py -- the two")
        print("sources fail on different labels, so combining beats either alone.")



def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--four-bit", action="store_true", help="4-bit quantisation, for 16GB GPUs")
    ap.add_argument("--out", type=Path, default=Path("report_labels.csv"))
    ap.add_argument("--max-new", type=int, default=160,
                    help="output is a 12-number JSON; 160 is ample and faster than 200")
    ap.add_argument("--validate", action="store_true", help="58 gold studies only")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
