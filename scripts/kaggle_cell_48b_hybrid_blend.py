# ============================================================
# RSNA Knee — HYBRID, step 4 of 4: blend the public pipeline with our team.
#
#   final = (1 - W) * rank(public 0.954 fusion) + W * rank(our nine arms),
#   per finding, ranks as fractions (the competition metric is rank-based AUC).
#
# W = __W__. No gold check is possible: the public weights were trained on all
# of train (gold included), so only the leaderboard can choose W.
#
# Never files one half alone. Without ours.csv, or without a public
# submission.csv written AFTER ours was stashed, it deletes submission.csv and
# raises, so the run fails instead of scoring as something we did not measure.
# ============================================================
import os, glob, shutil
import numpy as np, pandas as pd

W = float("__W__")
assert 0.0 < W < 1.0, W
H = "/kaggle/working/hybrid"
SUB = "/kaggle/working/submission.csv"
ID = "StudyInstanceUID"


def refuse(msg):
    if os.path.exists(SUB):
        os.remove(SUB)
    raise RuntimeError(msg)


# Any failure below -- not only the explicit refusals -- must not leave the
# public half standing alone as submission.csv.
try:
    if not os.path.exists(f"{H}/ours.csv"):
        refuse("hybrid/ours.csv missing -- our team did not run")
    if not os.path.exists(SUB):
        refuse("the public pipeline wrote no submission.csv")
    # Public cell 5 writes a CoAtNet-384-only submission.csv before the other legs
    # run, so "newer than the stash" alone does not prove the fusion ran. The c224 and
    # reader legs must both exist, written after the stash, and submission.csv must
    # be newer than the reader's output, which fusion writes just before it.
    t_stash = float(open(f"{H}/stashed_at.txt").read())
    for leg in ("_c224.csv", "_own.csv", "_sota_0949.csv"):
        f = f"/kaggle/working/{leg}"
        if not os.path.exists(f) or os.path.getmtime(f) <= t_stash:
            refuse(f"public leg {leg} missing or older than the stash")
    if os.path.getmtime(SUB) < os.path.getmtime("/kaggle/working/_own.csv"):
        refuse("submission.csv predates the reader leg -- the public fusion did not run")

    pub = pd.read_csv(SUB, dtype={ID: str})
    ours = pd.read_csv(f"{H}/ours.csv", dtype={ID: str})
    labels = [c for c in pub.columns if c != ID]
    if sorted(labels) != sorted(c for c in ours.columns if c != ID) or len(labels) != 12:
        refuse(f"label columns differ: {labels} vs {list(ours.columns)}")
    if set(pub[ID]) != set(ours[ID]) or not pub[ID].is_unique:
        refuse("the two halves cover different studies")
    if not ours[ID].is_unique:
        refuse("our half has duplicate study ids")
    ours = ours.set_index(ID).loc[pub[ID]].reset_index()
    for name, df in (("public", pub), ("ours", ours)):
        if df[labels].isna().any().any():
            refuse(f"{name} has NaN")

    c384 = pd.read_csv("/kaggle/working/_sota_0949.csv", dtype={ID: str}).set_index(ID).loc[pub[ID]]
    if np.allclose(c384[labels].values, pub[labels].values):
        refuse("public submission.csv is the CoAtNet-384 leg alone, not the fusion")

    rp = pub[labels].rank(method="average", pct=True)
    ro = ours[labels].rank(method="average", pct=True)
    rho = {c: rp[c].corr(ro[c], method="spearman") for c in labels}
    print("spearman public vs ours per finding:")
    print(pd.Series(rho).round(3).to_string())
    print(f"mean {np.mean(list(rho.values())):.3f}")

    shutil.copy(SUB, f"{H}/public.csv")
    final = pub[[ID]].copy()
    for c in labels:
        final[c] = ((1 - W) * rp[c] + W * ro[c]).values
    final[labels] = final[labels].rank(method="average", pct=True)
    final.to_csv(SUB, index=False)
    # leave only submission.csv at the top level: the public legs' _*.csv go aside
    for f in glob.glob("/kaggle/working/*.csv"):
        if f != SUB:
            shutil.move(f, f"{H}/{os.path.basename(f)}")
    assert final.shape == (len(pub), 13) and final[labels].notna().all().all()
    print(f"\nhybrid submission: {len(final)} studies, W(ours) = {W}")
    print(f"top level: {sorted(os.listdir('/kaggle/working'))}")
except BaseException:
    if os.path.exists(SUB):
        os.remove(SUB)
    raise
