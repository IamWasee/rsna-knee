"""Gold-58 AUC of every arm in a Kaggle training run, read from its log.

train.py prints "fold{f} ep{e}  loss ...  OOF x  gold y" every epoch and keeps
the checkpoint with the best OOF. The number that matters is gold at that kept
epoch: AUC on the 58 radiologist-labelled studies, never used for selection,
invariant to --sharpen-to, and the only labels we hold from the same source as
the leaderboard. On 2026-09-25 it predicted the 0.882 -> 0.909 jump (+0.025
predicted, +0.027 actual) where both OOF yardsticks under-called it.

    kaggle kernels output abdullahwasee/<slug> -p logs/<slug> --file-pattern '\\.log$'
    python scripts/gold_from_logs.py logs/<slug>/<slug>.log

A multi-arm run (planes-fixed, planes-5fold) prints its arms in the order they
trained -- ax, cor, sag for the plane cells. The log rounds to 3 decimals; the
exact value is gold_auc inside each fold*.pt.
"""
import json
import re
import sys

import numpy as np

RX = re.compile(r"fold(\d) ep(\d+)\s+loss [\d.]+\s+OOF ([\d.]+)\s+gold ([\d.]+)")


def arms(path):
    d = json.load(open(path))
    txt = "".join(e.get("data", "") for e in d if e["stream_name"] == "stdout")
    rows, seen = [], set()
    for m in RX.finditer(txt):
        if m.group(0) in seen:        # subprocess launches double-log every line
            continue
        seen.add(m.group(0))
        rows.append(tuple(map(float, m.groups())))
    out, cur, last = [], [], -1
    for r in rows:                    # a new arm starts when the fold index drops
        if r[0] < last:
            out.append(cur)
            cur = []
        cur.append(r)
        last = r[0]
    if cur:
        out.append(cur)
    res = []
    for a in out:
        best, tied = {}, {}
        for f, e, o, g in a:
            if f not in best or o > best[f][0]:
                best[f] = (o, g, int(e))
                tied[f] = {round(g, 3)}
            elif o == best[f][0]:
                tied[f].add(round(g, 3))
        # The log rounds OOF to 3 decimals, so two epochs can tie here while
        # train.py, comparing unrounded values, kept either one. When the tied
        # epochs disagree on gold, this log cannot say which was kept: on
        # ax-24ep-source fold 1 the log read 0.843 and the checkpoint held 0.854.
        amb = {f: sorted(v) for f, v in tied.items() if len(v) > 1}
        res.append((best, amb))
    return res


if __name__ == "__main__":
    for i, (b, amb) in enumerate(arms(sys.argv[1])):
        g = [b[f][1] for f in sorted(b)]
        print(f"arm {i}: {len(b)} folds  gold per fold {g}  mean {np.mean(g):.4f}  "
              f"(kept epochs {[b[f][2] for f in sorted(b)]})")
        for f, v in sorted(amb.items()):
            print(f"  fold {int(f)}: OOF tied at 3 decimals across epochs with gold {v} -- "
                  f"read gold_auc from fold{int(f)}.pt for the exact value")
