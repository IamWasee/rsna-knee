"""Watch Kaggle for new public notebooks and weights, and snapshot weights before they vanish.

Run every 30 minutes by launchd (scripts/install_watch.sh); no Claude involved.
Each run:
  1. lists this competition's public notebooks (newest and most voted) and
     flags new ones whose title claims a score >= FLAG_SCORE or that are popular
  2. lists knee weight datasets; a new or updated one used by a flagged notebook,
     or with enough votes, is SNAPSHOTTED: a CPU notebook (cell 49) copies it into
     our own private notebook output, which stays attachable if the author
     withdraws it (heliosli's compact-v1 vanished 2026-10-09 and took a 0.96
     recipe with it)
  3. records the medal cutoffs
  4. appends to data/audit/watch/report.md; a macOS notification for big news

State lives in data/audit/watch/state.json (git-ignored).
"""

from __future__ import annotations

import csv
import io
import json
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "audit" / "watch"
STATE = OUT / "state.json"
REPORT = OUT / "report.md"
COMP = "rsna-knee-abnormality-detection"
KAGGLE = str(Path.home() / ".local" / "bin" / "kaggle")
FLAG_SCORE = 0.950
POPULAR = 40                 # votes that make a notebook worth a look
SNAP_VOTES = 5               # dataset votes that make it worth keeping
MAX_SNAPS_PER_RUN = 3
SNAP_MAX_BYTES = 18e9        # notebook output limit is 20 GB
# What our submissions stand on, snapshotted before anything else: the 0.954 pipeline's
# weights and the d4-blend / fast-2xT4 ensemble's.
PRIORITY = ["nartaa/rsna-knee-publication-swa-weights-20261007",
            "goodpjw2008/rsna-knee-2-5d-convnext-reader",
            "dreaddevelopment/raptor-knee-maxspan", "dreaddevelopment/raptor-knee-native384",
            "dreaddevelopment/raptor-knee-native384dense", "mattiaangeli/knee-mri-fold-weights",
            "mattiaangeli/rsna-knee-coat-resgated-ep10-top3",
            "mattiaangeli/rsna-knee-coatnet-d4-depthzone-swa3-b2",
            "mattiaangeli/rsna-knee-coatnet-global96-top3", "antoinegg1/rsna-knee-e11-diverse-heads-v20",
            "antoinegg1/rsna-knee-e9-radimagenet-heads-v15",
            "prvsiyan/rsna-knee-v52-radimagenet-heads-20260812", "pilkwang/rsna-knee-weights",
            "marwanmath/resnet-50-radimagenet-marwan", "dreaddevelopment/raptor-knee-widedense"]
SEARCHES = ["rsna knee", "knee weights", "raptor knee", "knee coatnet", "knee dinov",
            "knee mri checkpoint", "orthofoundation"]


def kaggle(*args, timeout=180) -> str:
    try:
        r = subprocess.run([KAGGLE, *args], capture_output=True, text=True, timeout=timeout)
        return r.stdout
    except subprocess.TimeoutExpired:
        return ""


def rows(text: str) -> list[dict]:
    try:
        return list(csv.DictReader(io.StringIO(text)))
    except csv.Error:
        return []


def claimed(title: str) -> float:
    s = [float("0." + m) for m in re.findall(r"0[.\-_ ]?(9[3-9]\d)", title)]
    return max(s) if s else 0.0


def notify(msg: str) -> None:
    subprocess.run(["osascript", "-e", f'display notification "{msg[:200]}" with title "Kaggle watch"'],
                   capture_output=True, timeout=20)


def snapshot(ref: str, state: dict, log: list) -> None:
    slug = "snap-" + re.sub(r"[^a-z0-9]+", "-", ref.split("/")[1].lower())[:40].strip("-")
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "push_kernel.py"),
                        str(ROOT / "scripts" / "kaggle_cell_49_snapshot.py"), slug,
                        "--no-competition", "--dataset", ref, "--sub", "URL=-", "--sub", "NAME=-"],
                       capture_output=True, text=True, timeout=300, cwd=ROOT)
    ok = "successfully pushed" in r.stdout
    state["snapped"][ref] = {"slug": slug, "at": time.time(), "ok": ok}
    log.append(f"- SNAPSHOT {'launched' if ok else 'FAILED'}: `{ref}` -> abdullahwasee/{slug}")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    state = json.loads(STATE.read_text()) if STATE.exists() else {}
    for k in ("kernels", "datasets", "snapped", "cutoffs"):
        state.setdefault(k, {})
    first = not state["kernels"]
    log, big = [], []

    # 1. notebooks
    seen_now = {}
    for sort in ("dateCreated", "voteCount", "dateRun"):
        for r in rows(kaggle("kernels", "list", "--competition", COMP, "--sort-by", sort,
                             "--page-size", "100", "--csv")):
            seen_now[r["ref"]] = r
    flagged_kernels = []
    for ref, r in seen_now.items():
        votes = int(r.get("totalVotes") or 0)
        score = claimed(r.get("title", ""))
        old = state["kernels"].get(ref)
        if old is None and not first:
            line = f"- NEW notebook `{ref}`: \"{r.get('title')}\" ({votes} votes)"
            if score >= FLAG_SCORE or votes >= POPULAR:
                line += f"  **<- FLAG (claims {score:.3f})**" if score else "  **<- popular**"
                big.append(f"{ref} {score or ''}")
                flagged_kernels.append(ref)
            log.append(line)
        elif old and votes >= POPULAR and old.get("votes", 0) < POPULAR:
            log.append(f"- notebook `{ref}` became popular ({votes} votes)")
            flagged_kernels.append(ref)
        state["kernels"][ref] = {"votes": votes, "title": r.get("title"), "score": score}

    # weights those flagged notebooks depend on
    wanted = set()
    for ref in flagged_kernels:
        tmp = OUT / "meta"
        tmp.mkdir(exist_ok=True)
        kaggle("kernels", "pull", ref, "-p", str(tmp), "-m")
        mf = tmp / "kernel-metadata.json"
        if mf.exists():
            for d in json.loads(mf.read_text()).get("dataset_sources", []):
                if d:
                    wanted.add(d)
            mf.unlink()

    # 2. datasets
    for q in SEARCHES:
        for r in rows(kaggle("datasets", "list", "-s", q, "--sort-by", "updated", "--csv")):
            ref = r["ref"]
            upd, votes = r.get("lastUpdated", ""), int(r.get("voteCount") or 0)
            old = state["datasets"].get(ref)
            if old is None and not first:
                log.append(f"- NEW dataset `{ref}` ({r.get('size')} bytes, {votes} votes): {r.get('title')}")
            elif old and old.get("updated") != upd:
                log.append(f"- dataset UPDATED `{ref}` ({r.get('size')} bytes): {r.get('title')}")
                if "retired" in r.get("title", "").lower() or int(r.get("size") or 0) < 10_000 < int(old.get("size") or 0):
                    big.append(f"{ref} looks WITHDRAWN")
            state["datasets"][ref] = {"updated": upd, "votes": votes, "size": r.get("size"),
                                      "title": r.get("title")}
            if votes >= SNAP_VOTES and int(r.get("size") or 0) > 10_000_000:
                wanted.add(ref)

    # 3. snapshot what matters, a couple per run, never twice for one version
    n = 0
    wanted |= set(PRIORITY)
    for ref in [r for r in PRIORITY] + sorted(wanted - set(PRIORITY)):
        meta = state["datasets"].get(ref, {})
        if not meta:      # a priority dataset no search returned: ask for it directly
            r = rows(kaggle("datasets", "list", "-s", ref.split("/")[1], "--csv"))
            r = [x for x in r if x["ref"] == ref]
            if r:
                meta = {"updated": r[0].get("lastUpdated"), "size": r[0].get("size")}
                state["datasets"][ref] = meta
        size = int(meta.get("size") or 0)
        prev = state["snapped"].get(ref)
        if prev and prev.get("ok") and prev.get("updated") == meta.get("updated"):
            continue
        if size > SNAP_MAX_BYTES or n >= MAX_SNAPS_PER_RUN:
            continue
        snapshot(ref, state, log)
        state["snapped"][ref]["updated"] = meta.get("updated")
        n += 1

    # 4. medal cutoffs
    lb = OUT / "lb"
    lb.mkdir(exist_ok=True)
    kaggle("competitions", "leaderboard", COMP, "-d", "-p", str(lb), timeout=300)
    try:
        import zipfile
        z = sorted(lb.glob("*.zip"))[-1]
        with zipfile.ZipFile(z) as zf:
            text = zf.read(zf.namelist()[0]).decode()
        scores = sorted((float(r["Score"]) for r in csv.DictReader(io.StringIO(text)) if r.get("Score")),
                        reverse=True)
        t = len(scores)
        cut = {"teams": t, "gold": scores[10 + int(0.002 * t) - 1], "silver": scores[int(0.05 * t) - 1],
               "bronze": scores[int(0.10 * t) - 1], "top1": scores[0]}
        prev = state["cutoffs"].get("last")
        if prev and (cut["silver"], cut["bronze"]) != (prev["silver"], prev["bronze"]):
            log.append(f"- MEDAL LINES moved: silver {prev['silver']}->{cut['silver']}, "
                       f"bronze {prev['bronze']}->{cut['bronze']} ({t} teams)")
        state["cutoffs"]["last"] = cut
        z.unlink()
    except Exception as e:   # the leaderboard is a nicety; never stop the watch for it
        log.append(f"- (leaderboard read failed: {type(e).__name__})")

    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    if first:
        log.insert(0, f"- baseline: {len(state['kernels'])} notebooks, {len(state['datasets'])} datasets recorded")
    if log:
        with REPORT.open("a") as f:
            f.write(f"\n### {stamp}\n" + "\n".join(log) + "\n")
    if big:
        notify("; ".join(big))
    STATE.write_text(json.dumps(state))
    print(f"{stamp}: {len(log)} event(s), {len(big)} flagged, {n} snapshot(s)")


if __name__ == "__main__":
    main()
