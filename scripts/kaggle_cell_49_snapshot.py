# ============================================================
# RSNA Knee — SNAPSHOT: keep our own copy of a public asset. CPU, internet ON.
#
# Public weights get withdrawn (heliosli's compact-v1 vanished on 2026-10-09 and
# every notebook built on it stopped running). A notebook's output stays
# attachable after its sources are gone, so copying an asset into this
# notebook's /kaggle/working keeps it usable for us.
#
#   --sub URL=<https url or "-">     file to download (e.g. a GitHub LFS weight)
#   --sub NAME=<file name>           name to save it under
#   attach any dataset with --dataset: every attached dataset is copied too
#
# Private: the copy is only visible to our account.
# ============================================================
import os, shutil, subprocess, hashlib, json, time
from pathlib import Path

URL, NAME = "__URL__", "__NAME__"
OUT = Path("/kaggle/working/snapshot")
OUT.mkdir(parents=True, exist_ok=True)
got = {}
if URL != "-":
    dst = OUT / NAME
    subprocess.run(["wget", "-q", "--tries=5", "--timeout=120", "-O", str(dst), URL], check=True, timeout=3 * 3600)
    h = hashlib.sha256()
    with dst.open("rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    got[NAME] = {"bytes": dst.stat().st_size, "sha256": h.hexdigest(), "url": URL}
    # a Git LFS pointer is ~130 bytes of text, not weights
    assert dst.stat().st_size > 1_000_000, f"{NAME} is only {dst.stat().st_size} bytes -- an LFS pointer?"

# every attached dataset (never the competition data)
for d in sorted(Path("/kaggle/input").glob("*")) + sorted(Path("/kaggle/input").glob("datasets/*/*")):
    if not d.is_dir() or d.name in ("rsna-knee-abnormality-detection", "competitions", "datasets"):
        continue
    if any(p.name == "test_series" for p in d.iterdir()):
        continue
    target = OUT / "datasets" / d.name
    t0 = time.time()
    shutil.copytree(d, target, dirs_exist_ok=True)
    n = sum(1 for _ in target.rglob("*") if _.is_file())
    size = sum(p.stat().st_size for p in target.rglob("*") if p.is_file())
    got[f"dataset:{d.name}"] = {"files": n, "bytes": size, "seconds": round(time.time() - t0)}

(OUT / "snapshot_manifest.json").write_text(json.dumps(got, indent=1))
print(json.dumps(got, indent=1))
total = sum(v["bytes"] for v in got.values())
print(f"snapshot: {len(got)} item(s), {total/1e9:.2f} GB")
assert total < 19e9, "over the 20 GB output limit"
