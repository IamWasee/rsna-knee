# RSNA Knee Abnormality Detection

12-label multi-label classification on knee MRI + paired radiology reports.
Metric: macro-averaged AUC-ROC. Kaggle code competition (≤9h notebook runtime, no internet at scoring).

## Where things run

| | |
|---|---|
| **This repo** | code only — no data, no weights (`.gitignore` enforces it) |
| **Colab / Kaggle** | data + GPU training |

The dataset does not fit on the local machine (~3 GB free), so nothing here reads local DICOMs.
`src/paths.py` resolves the data root from wherever the current environment mounted it, or from
`RSNA_KNEE_DATA`.

## Layout

```
src/config.py      the 12 labels, in the exact order the scorer expects
src/paths.py       data-root resolution (Kaggle / Colab / env var)
src/explore.py     dataset layout discovery — run this first
src/submission.py  build + strictly validate submission.csv
notebooks/         Colab driver notebooks
configs/           training hyperparameters
```

## What the data actually is

Measured, not assumed (see `src/config.py`):

- **4,407 train studies. 58 have labels.** All 12 labels present on those 58, none on the other 4,349.
- **Reports are the supervision.** 4,407 free-text reports in 9+ languages -- en/es/tr/hr/el/de/bg/nl/fr.
  English is a minority (~38% of a 600-report sample).
- **24,371 series**, median 5 per study. `train_series.csv` and `test_series.csv` both carry
  `Fluid_Sensitive`, `Fat_Suppression`, `Anatomical_Plane` -- sequence typing for free, no DICOM parsing.
- **Imaging is ~0.9-1.8 TB.** It cannot be downloaded to Colab (88 GB) or locally (3 GB).
  All imaging work happens in Kaggle notebooks against the mounted dataset.

## The actual problem

Not "train a CNN on knee MRI." It's:

1. Extract 12 labels from 4,349 multilingual reports -- limited by **negation detection**, not term
   coverage. Reports are dominated by normal findings ("Normal ACL, PCL, MCL", "Няма МР данни за
   ставен излив" = "no MR evidence of joint effusion"). Naive term matching scores those positive.
2. Train an image model on the resulting labels.
3. Infer from images alone -- the hidden test set has no reports.

The 58 gold studies validate step 1. They are far too few to validate step 2.

## Status

Written and verified against real data:
`config.py`, `paths.py`, `explore.py`, `submission.py`, `report_labels.py` (extractor + eval harness).

Not written yet -- `dataset.py`, `model.py`, `train.py`, `infer.py`. They come after label extraction,
since label quality bounds everything they can achieve.

## Workflow

1. `scripts/colab_cell_1.py`, `colab_cell_2.py` -- data discovery (done)
2. `src/report_labels.py` -- build and score the label extractor (runs on `train.csv`, 5.7 MB, laptop-sized)
3. Image pipeline, in Kaggle notebooks
