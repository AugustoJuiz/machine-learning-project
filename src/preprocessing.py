"""
preprocessing.py — Limpeza estrutural e construção do pipeline de pré-processamento.

IMPORTANTE — Separação entre limpeza estrutural e transformadores:
    - clean_data()        : faz APENAS limpeza estrutural (sem imputação, scaling ou OHE).
                            Salva o resultado em data/processed/. Seguro de aplicar antes do split.
    - build_preprocessor(): cria um ColumnTransformer (imputação + scaling + OHE) que
                            NUNCA é ajustado aqui. Será ajustado somente no treino, dentro
                            de um sklearn Pipeline, evitando data leakage.

Funções:
    clean_data()           — limpeza estrutural; retorna DataFrame e salva em processed/.
    load_processed()       — lê o CSV processado, regenerando se necessário.
    get_feature_lists()    — separa features numéricas e categóricas de X (sem alvo).
    build_preprocessor()   — cria ColumnTransformer não ajustado.
    split_data()           — divide treino/teste com estratificação.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (
    MISSING_DROP_THRESHOLD,
    PROCESSED_CSV,
    RANDOM_STATE,
    RAW_CSV,
    TARGET,
    TEST_SIZE,
)


# ── Helpers internos ─────────────────────────────────────────────────────────────

def _month_to_season_aus(month: int) -> str:
    """
    Converte mês (1–12) para estação do ano no hemisfério sul (Austrália).

    Estações australianas:
        Verão  (Summer) : dezembro, janeiro, fevereiro
        Outono (Autumn) : março, abril, maio
        Inverno (Winter): junho, julho, agosto
        Primavera (Spring): setembro, outubro, novembro
    """
    if month in (12, 1, 2):
        return "Summer"
    if month in (3, 4, 5):
        return "Autumn"
    if month in (6, 7, 8):
        return "Winter"
    return "Spring"


def _extract_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extrai variáveis de calendário da coluna Date e a remove.

    Adiciona:
        Month  (int 1–12)          — captura sazonalidade mensal.
        Season (str Summer/Autumn/Winter/Spring) — captura padrão sazonal australiano.

    Parâmetros:
        df : DataFrame com coluna 'Date' no formato YYYY-MM-DD.

    Retorno:
        DataFrame com colunas Month, Season adicionadas e Date removida.
    """
    df = df.copy()
    if "Date" not in df.columns:
        return df

    dates = pd.to_datetime(df["Date"], errors="coerce")
    df["Month"]  = dates.dt.month
    df["Season"] = dates.dt.month.map(_month_to_season_aus)
    df = df.drop(columns=["Date"])
    return df


def _encode_binary_cols(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converte colunas binárias Yes/No para 1/0 (mantém NaN intacto).

    Aplica em: RainToday e TARGET (RainTomorrow).
    Essa conversão é estrutural — não é imputação nem encoding de feature.

    Parâmetros:
        df : DataFrame bruto.

    Retorno:
        DataFrame com colunas binárias convertidas.
    """
    df = df.copy()
    yn_map = {"Yes": 1, "No": 0}
    for col in ["RainToday", TARGET]:
        if col in df.columns:
            df[col] = df[col].map(yn_map)
    return df


def _drop_high_missing_cols(
    df: pd.DataFrame,
    threshold: float,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """
    Remove colunas com percentual de ausentes acima do limiar.

    A decisão de remoção considera apenas o percentual de ausentes.
    A justificativa meteorológica deve ser discutida e documentada
    no relatório (dicionário de dados) após análise na EDA.

    Parâmetros:
        df        : DataFrame.
        threshold : fração de ausentes (0–1) para remoção; ex.: 0.40 = 40%.

    Retorno:
        Tupla (DataFrame sem as colunas removidas, dicionário {coluna: pct_ausentes}).
    """
    pct_missing = df.isnull().mean()
    # Nunca remove a variável-alvo
    cols_to_drop = [
        c for c in pct_missing[pct_missing > threshold].index
        if c != TARGET
    ]
    dropped_info = {c: round(pct_missing[c] * 100, 2) for c in cols_to_drop}

    if cols_to_drop:
        print(f"[preprocessing] Colunas removidas (ausentes > {threshold*100:.0f}%):")
        for col, pct in sorted(dropped_info.items(), key=lambda x: -x[1]):
            print(f"  {col:<22} {pct:.1f}% ausentes")
        df = df.drop(columns=cols_to_drop)
    else:
        print(f"[preprocessing] Nenhuma coluna acima do limiar de {threshold*100:.0f}%.")

    return df, dropped_info


# ── Limpeza estrutural ───────────────────────────────────────────────────────────

def clean_data(
    df: pd.DataFrame,
    drop_threshold: float | None = None,
) -> pd.DataFrame:
    """
    Realiza limpeza estrutural do dataset bruto.

    Operações (em ordem):
        1. Remove linhas onde RainTomorrow é NaN (alvo inválido).
        2. Converte RainToday e RainTomorrow: Yes→1, No→0 (NaN preservado).
        3. Extrai Month e Season a partir de Date; remove coluna Date.
        4. Remove colunas com percentual de ausentes acima de drop_threshold.
        5. Reseta o índice.
        6. Salva em data/processed/weatherAUS_processed.csv.

    O arquivo salvo em processed/ NÃO contém imputação, encoding nem scaling.
    Esses transformadores ficam exclusivamente dentro de Pipeline/ColumnTransformer
    e são ajustados apenas no conjunto de treino.

    Parâmetros:
        df             : DataFrame bruto retornado por data_loader.load_raw().
        drop_threshold : fração de ausentes para remoção (0–1).
                         Usa MISSING_DROP_THRESHOLD de config.py se None.

    Retorno:
        pd.DataFrame limpo (sem alvo nulo, com variáveis de calendário).
    """
    if drop_threshold is None:
        drop_threshold = MISSING_DROP_THRESHOLD

    df = df.copy()
    n_inicial = len(df)

    # 1. Remove linhas com alvo nulo
    df = df.dropna(subset=[TARGET])
    n_removidas = n_inicial - len(df)
    print(f"[preprocessing] Linhas removidas (alvo nulo): {n_removidas:,} "
          f"({n_removidas / n_inicial * 100:.1f}%)")

    # 2. Converte binárias Yes/No → 1/0
    df = _encode_binary_cols(df)

    # 3. Variáveis de calendário a partir de Date
    df = _extract_date_features(df)
    print("[preprocessing] Variáveis de calendário extraídas: Month, Season")

    # 4. Remove colunas com excesso de ausentes
    df, dropped = _drop_high_missing_cols(df, drop_threshold)

    # 5. Reset de índice
    df = df.reset_index(drop=True)

    # 6. Salva arquivo processado
    PROCESSED_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_CSV, index=False, encoding="utf-8")
    print(f"\n[preprocessing] Dataset processado salvo: {PROCESSED_CSV}")
    print(f"[preprocessing] Shape final: {df.shape[0]:,} linhas × {df.shape[1]} colunas\n")

    # Sumário de distribuição do alvo após limpeza
    target_counts = df[TARGET].value_counts()
    target_pct    = df[TARGET].value_counts(normalize=True) * 100
    print("[preprocessing] Distribuição do alvo após limpeza:")
    for val in target_counts.index:
        print(f"  {int(val)} → {target_counts[val]:,} ({target_pct[val]:.1f}%)")

    return df


def load_processed(df_raw: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Carrega o dataset processado.

    Se o arquivo processed/ não existir, regenera a partir de df_raw (se fornecido)
    ou a partir do CSV bruto em data/raw/.

    Parâmetros:
        df_raw : DataFrame bruto para regenerar processed/ se necessário.

    Retorno:
        pd.DataFrame processado.
    """
    if PROCESSED_CSV.exists():
        print(f"[preprocessing] Carregando dataset processado: {PROCESSED_CSV}")
        return pd.read_csv(PROCESSED_CSV, low_memory=False)

    print("[preprocessing] Arquivo processed/ não encontrado. Regenerando...")
    if df_raw is None:
        from src.data_loader import load_raw
        df_raw = load_raw()
    return clean_data(df_raw)


# ── Separação de features ────────────────────────────────────────────────────────

def get_feature_lists(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    """
    Separa as features em numéricas e categóricas.

    Deve ser chamado sobre X_train (sem a coluna alvo) para garantir que
    os transformadores sejam construídos com base apenas nas features de treino.

    Parâmetros:
        X : DataFrame de features (sem coluna TARGET).

    Retorno:
        Tupla (numeric_features, categorical_features) com listas de nomes de colunas.
    """
    numeric_features     = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()

    print(f"[preprocessing] Features numéricas  ({len(numeric_features)}): {numeric_features}")
    print(f"[preprocessing] Features categóricas ({len(categorical_features)}): {categorical_features}")

    return numeric_features, categorical_features


# ── ColumnTransformer ────────────────────────────────────────────────────────────

def build_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str],
) -> ColumnTransformer:
    """
    Constrói o ColumnTransformer de pré-processamento (sem ajuste).

    Pipeline numérico:
        SimpleImputer(strategy='median') → StandardScaler()

    Pipeline categórico:
        SimpleImputer(strategy='most_frequent') →
        OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    ATENÇÃO: este objeto NÃO é ajustado aqui. Ele será encapsulado em um
    sklearn Pipeline com o classificador e ajustado apenas em X_train,
    evitando data leakage.

    Parâmetros:
        numeric_features     : lista de nomes de colunas numéricas.
        categorical_features : lista de nomes de colunas categóricas.

    Retorno:
        ColumnTransformer não ajustado.
    """
    numeric_pipeline = SklearnPipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])

    categorical_pipeline = SklearnPipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline,     numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    print("[preprocessing] ColumnTransformer criado (não ajustado).")
    print("  Numerico  : SimpleImputer(median) -> StandardScaler")
    print("  Categorico: SimpleImputer(most_frequent) -> OneHotEncoder(handle_unknown='ignore')")

    return preprocessor


# ── Split treino/teste ───────────────────────────────────────────────────────────

def split_data(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Divide o dataset em conjuntos de treino e teste com estratificação.

    Parâmetros:
        test_size=0.2, stratify=y, random_state=42 (definidos em config.py).

    REGRA DE DATA LEAKAGE: nenhum transformador é ajustado aqui.
    O conjunto de teste é isolado e usado SOMENTE na avaliação final.

    Parâmetros:
        df : DataFrame processado por clean_data() (com alvo inteiro 0/1).

    Retorno:
        Tupla (X_train, X_test, y_train, y_test).
    """
    from sklearn.model_selection import train_test_split

    X = df.drop(columns=[TARGET])
    y = df[TARGET].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    print(f"[preprocessing] Split treino/teste (stratify=y, random_state={RANDOM_STATE}):")
    print(f"  X_train : {X_train.shape[0]:,} linhas × {X_train.shape[1]} colunas")
    print(f"  X_test  : {X_test.shape[0]:,} linhas × {X_test.shape[1]} colunas")

    for label, y_part in [("Treino", y_train), ("Teste", y_test)]:
        vc  = y_part.value_counts()
        pct = y_part.value_counts(normalize=True) * 100
        print(f"  {label}: classe 0={vc.get(0,0):,} ({pct.get(0,0):.1f}%), "
              f"classe 1={vc.get(1,0):,} ({pct.get(1,0):.1f}%)")

    return X_train, X_test, y_train, y_test
