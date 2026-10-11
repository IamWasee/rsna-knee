"""Train an OrthoFoundation arm on Modal (A100), with our own train.py, folds and gold-58.

Two steps, both with hard time caps so nothing runs on past its budget:

  modal run scripts/modal_train.py::fetch           # once: Kaggle -> Modal volume (CPU)
  modal run scripts/modal_train.py::train --plane sag --fold 0 [--oai 1200] [--hours 3]

fetch pulls, with the Kaggle key stored as the Modal secret "kaggle"
(KAGGLE_USERNAME / KAGGLE_KEY): the competition's train.csv (gold labels only;
no DICOM), our cache-<plane> notebook outputs, the pseudo50 label table, the
private OAI batches, and OrthoFoundation-L from its GitHub release.

train runs src/train.py with the combo recipe (pseudo50, --sharpen-to none,
--lr-backbone 3e-5) on --backbone ortho:<weights>, writes to /data/runs/<tag>,
and prints the gold-58 numbers train.py stores in each fold checkpoint.
Code is cloned from master at run time, so a fix pushed to GitHub is picked up.
"""

from __future__ import annotations

import modal

app = modal.App("rsna-knee-orthofoundation")
vol = modal.Volume.from_name("rsna-knee-data", create_if_missing=True)
DATA = "/data"
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "unzip", "wget")
    .pip_install("torch==2.5.1", "torchvision==0.20.1", "timm==1.0.25", "transformers==4.46.3",
                 "pandas", "numpy<2.2", "scikit-learn", "pydicom", "opencv-python-headless",
                 "kaggle==1.7.4.5", "omegaconf", "termcolor", "ftfy")
    .run_commands("git clone https://github.com/facebookresearch/dinov3.git /opt/dinov3 "
                  "&& git -C /opt/dinov3 checkout -q 6876159")
)
KAGGLE = modal.Secret.from_name("kaggle")
OWNER = "abdullahwasee"
OF_URL = "https://github.com/ytrsk/OrthoFoundation/raw/main/OrthoFoudation-L.pth"
OF_SHA = "385a775822107b68eaa486336feb982e1ce7bd6d4e8c03ceb482a0bf546f2ff9"


def sh(cmd: str, timeout: int = 3600) -> None:
    import subprocess
    print("+", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True, timeout=timeout)


@app.function(image=image, volumes={DATA: vol}, secrets=[KAGGLE], timeout=4 * 3600, cpu=4)
def fetch(planes: str = "sag", oai: bool = True) -> None:
    import hashlib, os
    os.makedirs(f"{DATA}/comp", exist_ok=True)
    if not os.path.exists(f"{DATA}/comp/train.csv"):
        sh(f"kaggle competitions download rsna-knee-abnormality-detection -f train.csv -p {DATA}/comp")
        sh(f"cd {DATA}/comp && (unzip -o -q train.csv.zip && rm -f train.csv.zip || true)")
    for p in planes.split(","):
        d = f"{DATA}/k/cache-{p}"
        have = [os.path.join(r, "study_meta.csv") for r, _, fs in os.walk(d) if "cache_manifest.json" in fs]
        if not (have and os.path.exists(have[0])):
            sh(f"kaggle kernels output {OWNER}/cache-{p} -p {d}", timeout=3 * 3600)
    if not os.path.exists(f"{DATA}/labels/labels_pseudo50.csv"):
        sh(f"kaggle datasets download {OWNER}/rsna-knee-labels-pseudo -p {DATA}/labels --unzip")
    if oai:
        for i in range(1, 7):
            d = f"{DATA}/oai/b{i}"
            if not os.path.exists(f"{d}/oai_labels.csv"):
                sh(f"kaggle datasets download {OWNER}/oai-knee-b{i} -p {d} --unzip", timeout=3600)
    w = f"{DATA}/OrthoFoundation-L.pth"
    if not os.path.exists(w):
        sh(f"wget -q -O {w} {OF_URL}", timeout=3600)
    h = hashlib.sha256()
    with open(w, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    assert h.hexdigest() == OF_SHA, "OrthoFoundation-L differs from our snapshot's hash"
    vol.commit()
    sh(f"du -sh {DATA}/* && ls {DATA}/k {DATA}/oai")


# 4 h cap on the whole call (about $10): the fold-0 screen needs ~1.5 h. Raise it deliberately
# for more folds -- five need ~6 h, and a run killed by the cap loses the fold in progress.
@app.function(image=image, volumes={DATA: vol}, gpu="A100-80GB", cpu=8, timeout=4 * 3600)
def train(plane: str = "sag", fold: int = 0, oai: int = 0, hours: float = 3.0, tag: str = "of",
          unfreeze: int = 6, epochs: int = 16, batch: int = 8) -> None:
    import glob, json, os, subprocess, sys, time
    import torch
    sh("rm -rf /opt/code && git clone -q https://github.com/IamWasee/rsna-knee.git /opt/code "
       "&& git -C /opt/code log --oneline -1")
    caches = glob.glob(f"{DATA}/k/cache-{plane}/**/cache_{plane}/cache_manifest.json", recursive=True)
    assert caches, f"run fetch first (no cache_{plane})"
    cache = os.path.dirname(caches[0])
    # build_labels groups folds by scanner only with study_meta.csv; without it the folds change
    # silently and fold 0 is no longer sag-combo's fold 0
    assert os.path.exists(f"{cache}/study_meta.csv"), f"{cache}/study_meta.csv missing"
    man = json.load(open(caches[0]))
    out = f"{DATA}/runs/{tag}_{plane}" + (f"_f{fold}" if fold >= 0 else "")
    args = ["--cache", cache, "--labels", f"{DATA}/labels/labels_pseudo50.csv",
            "--backbone", f"ortho:{DATA}/OrthoFoundation-L.pth",
            "--size", str(man["size"]), "--slots", "1", "--n-slices", str(man["n_slices"]),
            "--folds", "5", "--head", "shared", "--pool", "focal", "--batch", str(batch),
            "--epochs", str(epochs), "--lr", "1e-3", "--lr-backbone", "3e-5",
            "--unfreeze-last", str(unfreeze), "--weight-decay", "0.02", "--sharpen-to", "none",
            "--seed", "42", "--avg-top", "3", "--grad-checkpoint", "--workers", "6", "--out", out]
    if fold >= 0:
        args += ["--only-fold", str(fold)]
    if oai:
        dirs = sorted(glob.glob(f"{DATA}/oai/b*/**/cache_{plane}", recursive=True))
        lab = sorted(glob.glob(f"{DATA}/oai/b*/**/oai_labels.csv", recursive=True))
        assert dirs and lab, "run fetch with oai"
        args += ["--extra-labels", lab[0], "--extra-cache", *dirs, "--extra-per-epoch", str(oai)]
    env = dict(os.environ, RSNA_KNEE_DATA=f"{DATA}/comp", DINOV3_REPO="/opt/dinov3")
    print(torch.cuda.get_device_name(0), flush=True)
    t = time.time()
    code = None
    try:
        code = subprocess.run([sys.executable, "-u", "/opt/code/src/train.py", *args], env=env,
                              timeout=int(hours * 3600)).returncode
    except subprocess.TimeoutExpired:
        print(f"train.py hit the {hours} h cap; keeping the folds it finished", flush=True)
    finally:
        vol.commit()          # whatever happened, finished fold checkpoints reach the volume
    print(f"train.py exit {code} after {(time.time()-t)/60:.0f} min", flush=True)
    for p in sorted(glob.glob(f"{out}/fold*.pt")):
        ck = torch.load(p, map_location="cpu", weights_only=False)
        print(f"{os.path.basename(p)}: gold {ck.get('gold_auc', float('nan')):.4f}  "
              f"single {ck.get('single_gold_auc', float('nan')):.4f}  OOF {ck.get('oof_auc', float('nan')):.4f}")
    assert code == 0, f"train.py exited {code}"
