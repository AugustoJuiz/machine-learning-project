"""
config.py — Configurações globais do projeto.

Centraliza paths, hiperparâmetros fixos, constantes e parâmetros
de experimento utilizados por todos os módulos de src/.
"""

from pathlib import Path

# ── Raiz do projeto ─────────────────────────────────────────────────────────────
# Resolve o diretório raiz independentemente de onde o script é chamado.
ROOT_DIR = Path(__file__).resolve().parent.parent

# ── Paths de dados ──────────────────────────────────────────────────────────────
DATA_DIR       = ROOT_DIR / "data"
RAW_DIR        = DATA_DIR / "raw"
PROCESSED_DIR  = DATA_DIR / "processed"

RAW_CSV        = RAW_DIR / "weatherAUS.csv"
PROCESSED_CSV  = PROCESSED_DIR / "weatherAUS_processed.csv"

# ── Paths de saída ──────────────────────────────────────────────────────────────
OUTPUTS_DIR    = ROOT_DIR / "outputs"
FIGURES_DIR    = OUTPUTS_DIR / "figures"
TABLES_DIR     = OUTPUTS_DIR / "tables"
REPORTS_DIR    = OUTPUTS_DIR / "reports"

# ── Paths de modelos ────────────────────────────────────────────────────────────
MODELS_DIR     = ROOT_DIR / "models"
BEST_MODEL_PATH = MODELS_DIR / "best_model.joblib"

# ── Paths de documentação ───────────────────────────────────────────────────────
DOCS_DIR       = ROOT_DIR / "docs"

# ── Kaggle ──────────────────────────────────────────────────────────────────────
# Slug do dataset: <owner>/<dataset-name>
KAGGLE_DATASET = "jsphyg/weather-dataset-rattle-package"
KAGGLE_FILENAME = "weatherAUS.csv"

# ── Parâmetros de experimento ───────────────────────────────────────────────────
RANDOM_STATE = 42        # Semente global para reprodutibilidade
TARGET       = "RainTomorrow"  # Variável-alvo
TEST_SIZE    = 0.2       # Proporção do conjunto de teste (20%)
CV_FOLDS     = 5         # Número de folds na validação cruzada estratificada

# ── Pré-processamento ───────────────────────────────────────────────────────────
# Colunas com percentual de ausentes acima desse limiar são candidatas a remoção.
# A decisão final é empírica (EDA + relevância meteorológica) e documentada
# no dicionário de dados e no relatório.
MISSING_DROP_THRESHOLD = 0.40  # 40% de ausentes como limiar de candidatura a remoção

# ── SVM ────────────────────────────────────────────────────────────────────────
# Tamanho da amostra estratificada do treino usada para LinearSVC.
# SVC com kernel RBF é O(n²) e inviável no dataset completo (~142k linhas).
# A avaliação final do SVM ocorre no mesmo conjunto de teste dos demais modelos.
SVM_SAMPLE_SIZE = 30_000  # Número de amostras do treino para LinearSVC

# ── Estilo de gráficos ──────────────────────────────────────────────────────────
FIGURE_DPI    = 150       # Resolução padrão das figuras salvas
FIGURE_SIZE   = (10, 6)   # Tamanho padrão (largura, altura) em polegadas
PALETTE       = "Set2"    # Paleta de cores padrão (seaborn)

# ── Garantia de pastas de saída ─────────────────────────────────────────────────
# Executado na importação do módulo para garantir que os diretórios existam.
for _dir in (FIGURES_DIR, TABLES_DIR, REPORTS_DIR, MODELS_DIR, PROCESSED_DIR):
    _dir.mkdir(parents=True, exist_ok=True)
