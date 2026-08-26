"""Competition constants, confirmed against the real data files."""

# Exact column order and spelling the scorer expects (from sample_submission.csv).
LABELS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
    "Medial OA", "Lateral OA", "PF OA", "Effusion",
    "Synovitis", "Baker's", "Contusion", "Fracture",
]

ID_COL = "StudyInstanceUID"
SERIES_COL = "SeriesInstanceUID"
SUBMISSION_COLUMNS = [ID_COL] + LABELS

METRIC = "macro_auc_roc"  # macro-averaged AUC-ROC over the 12 labels

# --- Confirmed dataset facts (measured, not assumed) ----------------------
N_TRAIN_STUDIES = 4407
N_TRAIN_LABELED = 58      # studies with ground truth; all 12 labels present
N_TRAIN_SERIES = 24371

# Per-series metadata available for BOTH train and test -- free sequence typing,
# no DICOM header parsing needed.
SERIES_META = ["Fluid_Sensitive", "Fat_Suppression", "Anatomical_Plane"]
PLANES = ["Sagittal", "Coronal", "Axial"]

# Positive rate among the 58 labeled studies. Small n -- treat as a rough prior,
# not a target distribution.
GOLD_PREVALENCE = {
    "ACL": 0.414, "MCL": 0.155, "Medial Meniscus": 0.448, "Lateral Meniscus": 0.397,
    "Medial OA": 0.259, "Lateral OA": 0.190, "PF OA": 0.362, "Effusion": 0.603,
    "Synovitis": 0.466, "Baker's": 0.207, "Contusion": 0.328, "Fracture": 0.310,
}

# Languages seen in the report corpus (langdetect over a 600-report sample).
# English is a MINORITY -- an English-only extractor discards most of the signal.
REPORT_LANGUAGES = ["en", "es", "tr", "hr", "el", "de", "bg", "nl", "fr"]

# De-identification placeholders present in report text.
DEID_PLACEHOLDERS = ["[DATE]", "[TIME]", "[ID]", "[REDACTED]",
                     "[NAME]", "[YEAR]", "[IDENTIFIER]", "[PROFESSION]"]
