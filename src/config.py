from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATASET_PATH = DATA_DIR / "raw" / "municipio_validacion_preguntas_400.csv"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SPLITS_DIR = DATA_DIR / "splits"
MODELS_DIR = PROJECT_ROOT / "models"
TRAINED_MODEL_DIR = MODELS_DIR / "trained" / "municipio_question_validator"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
METRICS_DIR = ARTIFACTS_DIR / "metrics"
REPORTS_DIR = ARTIFACTS_DIR / "reports"
FIGURES_DIR = ARTIFACTS_DIR / "figures"

DEFAULT_BASE_MODEL_NAME = "FacebookAI/xlm-roberta-base"
BASE_MODEL_NAME_OR_PATH = os.getenv("BASE_MODEL_DIR", DEFAULT_BASE_MODEL_NAME)

TEXT_COLUMN = "text"
LABEL_COLUMN = "label"
SUBTYPE_COLUMN = "subtype"
MAX_LENGTH = 128
RANDOM_STATE = 42
TEST_SIZE = 0.15
VALIDATION_SIZE = 0.15
NUM_TRAIN_EPOCHS = 4
LEARNING_RATE = 2e-5
TRAIN_BATCH_SIZE = 8
EVAL_BATCH_SIZE = 8
WEIGHT_DECAY = 0.01
SAVE_TOTAL_LIMIT = 2
REPORT_TO = "none"

TRAIN_SPLIT_PATH = SPLITS_DIR / "train.csv"
VALIDATION_SPLIT_PATH = SPLITS_DIR / "validation.csv"
TEST_SPLIT_PATH = SPLITS_DIR / "test.csv"
LABELS_PATH = TRAINED_MODEL_DIR / "labels.json"
METRICS_PATH = METRICS_DIR / "test_metrics.json"
CLASSIFICATION_REPORT_PATH = REPORTS_DIR / "classification_report.csv"
MISCLASSIFIED_PATH = REPORTS_DIR / "misclassified_examples.csv"
CONFUSION_MATRIX_PATH = FIGURES_DIR / "confusion_matrix.csv"
CONFUSION_MATRIX_FIGURE_PATH = FIGURES_DIR / "confusion_matrix.png"


def ensure_project_dirs() -> None:
    for path in [
        DATA_DIR,
        PROCESSED_DATA_DIR,
        SPLITS_DIR,
        MODELS_DIR,
        TRAINED_MODEL_DIR,
        ARTIFACTS_DIR,
        METRICS_DIR,
        REPORTS_DIR,
        FIGURES_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)
