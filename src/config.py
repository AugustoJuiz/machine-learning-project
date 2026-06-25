"""Configurações globais do projeto — paths, constantes e parâmetros de experimento."""

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR       = ROOT_DIR / "data"
RAW_DIR        = DATA_DIR / "raw"
PROCESSED_DIR  = DATA_DIR / "processed"

RAW_CSV        = RAW_DIR / "weatherAUS.csv"
PROCESSED_CSV  = PROCESSED_DIR / "weatherAUS_processed.csv"

OUTPUTS_DIR    = ROOT_DIR / "outputs"
FIGURES_DIR    = OUTPUTS_DIR / "figures"
TABLES_DIR     = OUTPUTS_DIR / "tables"
REPORTS_DIR    = OUTPUTS_DIR / "reports"

MODELS_DIR      = ROOT_DIR / "models"
BEST_MODEL_PATH = MODELS_DIR / "best_model.joblib"

DOCS_DIR       = ROOT_DIR / "docs"

KAGGLE_DATASET  = "jsphyg/weather-dataset-rattle-package"
KAGGLE_FILENAME = "weatherAUS.csv"

RANDOM_STATE = 42
TARGET       = "RainTomorrow"
TEST_SIZE    = 0.2
CV_FOLDS     = 5

MISSING_DROP_THRESHOLD = 0.40

SVM_SAMPLE_SIZE = 30_000

FIGURE_DPI    = 150
FIGURE_SIZE   = (10, 6)
PALETTE       = "Set2"

for _dir in (FIGURES_DIR, TABLES_DIR, REPORTS_DIR, MODELS_DIR, PROCESSED_DIR):
    _dir.mkdir(parents=True, exist_ok=True)
