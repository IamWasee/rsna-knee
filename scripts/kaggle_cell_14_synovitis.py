# ============================================================
# RSNA Knee — Synovitis: how much is reachable from the reports?
#
# Attach ONLY the competition. CPU. ~30 seconds. No model, no GPU, no other
# notebooks needed -- the keyword extractor runs instantly and isolates the
# question: does Effusion's ranking improve Synovitis?
# ============================================================
import sys, glob
CODE = "/kaggle/working/rsna-knee"
!rm -rf $CODE && git clone -q https://github.com/IamWasee/rsna-knee.git $CODE
sys.path.insert(0, f"{CODE}/src")

TRAIN = next(iter(glob.glob("/kaggle/input/**/train.csv", recursive=True)))
print("train.csv:", TRAIN)

# Full per-label AUC (now with severity grading) plus the borrowing sweep.
!python $CODE/src/report_labels.py --data "$TRAIN" --borrow Synovitis Effusion

# How much of each weak label is even mentioned in the reports? If a finding is
# rarely written down, no extractor can recover it and the ceiling is not ours.
import re, pandas as pd
train = pd.read_csv(TRAIN)
gold = train[train["Synovitis"].notna()]
TERMS = {
    "Synovitis": r"synovit|sinovit|υμεν|синови|sinovyal|synovial",
    "Fracture":  r"fract|fraktur|kırık|фрактур|счупван|κάταγμα|prijelom|breuk",
    "PF OA":     r"patellofemoral|patelofemoral|femoropatel|chondromalac|condromalac|retropatell",
    "Effusion":  r"effusion|derrame|erguss|efüzyon|épanchement|излив|συλλογή|izliv|hydrops",
    "ACL":       r"anterior cruciate|\bACL\b|cruzado anterior|\bLCA\b|kruisband|kreuzband|çapraz|κρυπτ|VKB",
}
print(f"\n{'label':<12} {'mentioned':>10} {'gold rate':>10} {'pos mentioned':>14}")
print("-" * 50)
for lab, pat in TERMS.items():
    hit = train["Report"].astype(str).str.contains(pat, case=False, regex=True)
    pos = gold[lab] == 1
    print(f"{lab:<12} {hit.mean():>10.1%} {gold[lab].mean():>10.1%} "
          f"{hit[gold.index][pos].mean():>14.1%}")
print("-" * 50)
print("'pos mentioned' is the ceiling: the share of true positives the text even")
print("refers to. Low means the label is unreachable from reports at any quality.")
