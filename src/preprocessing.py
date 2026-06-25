"""Limpeza estrutural e pipeline de pré-processamento do dataset."""

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


def _month_to_season_aus(month: int) -> str:
    """Converte mês (1–12) para estação australiana (hemisfério sul)."""
    if month in (12, 1, 2):
        return "Summer"
    if month in (3, 4, 5):
        return "Autumn"
    if month in (6, 7, 8):
        return "Winter"
    return "Spring"


def _extract_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extrai Month e Season da coluna Date e a remove."""
    df = df.copy()
    if "Date" not in df.columns:
        return df

    dates = pd.to_datetime(df["Date"], errors="coerce")
    df["Month"]  = dates.dt.month
    df["Season"] = dates.dt.month.map(_month_to_season_aus)
    df = df.drop(columns=["Date"])
    return df


def _encode_binary_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Converte RainToday e RainTomorrow de Yes/No para 1/0, preservando NaN."""
    df = df.copy()
    yn_map = {"Yes": 1, "No": 0}
    for col in ["RainToday", TARGET]:
        if col in df.columns:
            df[col] = df[col].map(yn_map)
    return df


def _drop_high_missing_cols(df: pd.DataFrame, threshold: float) -> tuple[pd.DataFrame, dict[str, float]]:
    """Remove colunas com percentual de ausentes acima do limiar (nunca remove o alvo)."""
    pct_missing = df.isnull().mean()
    cols_to_drop = [c for c in pct_missing[pct_missing > threshold].index if c != TARGET]
    dropped_info = {c: round(pct_missing[c] * 100, 2) for c in cols_to_drop}

    if cols_to_drop:
        print(f"[preprocessing] Colunas removidas (ausentes > {threshold*100:.0f}%):")
        for col, pct in sorted(dropped_info.items(), key=lambda x: -x[1]):
            print(f"  {col:<22} {pct:.1f}% ausentes")
        df = df.drop(columns=cols_to_drop)
    else:
        print(f"[preprocessing] Nenhuma coluna acima do limiar de {threshold*100:.0f}%.")

    return df, dropped_info


def clean_data(df: pd.DataFrame, drop_threshold: float | None = None) -> pd.DataFrame:
    """Limpeza estrutural do dataset bruto; salva resultado em data/processed/."""
    if drop_threshold is None:
        drop_threshold = MISSING_DROP_THRESHOLD

    df = df.copy()
    n_inicial = len(df)

    df = df.dropna(subset=[TARGET])
    n_removidas = n_inicial - len(df)
    print(f"[preprocessing] Linhas removidas (alvo nulo): {n_removidas:,} ({n_removidas / n_inicial * 100:.1f}%)")

    df = _encode_binary_cols(df)
    df = _extract_date_features(df)
    print("[preprocessing] Variáveis de calendário extraídas: Month, Season")

    df, dropped = _drop_high_missing_cols(df, drop_threshold)

    df = df.reset_index(drop=True)

    PROCESSED_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_CSV, index=False, encoding="utf-8")
    print(f"\n[preprocessing] Dataset processado salvo: {PROCESSED_CSV}")
    print(f"[preprocessing] Shape final: {df.shape[0]:,} linhas × {df.shape[1]} colunas\n")

    target_counts = df[TARGET].value_counts()
    target_pct    = df[TARGET].value_counts(normalize=True) * 100
    print("[preprocessing] Distribuição do alvo após limpeza:")
    for val in target_counts.index:
        print(f"  {int(val)} → {target_counts[val]:,} ({target_pct[val]:.1f}%)")

    return df


def load_processed(df_raw: pd.DataFrame | None = None) -> pd.DataFrame:
    """Carrega o dataset processado ou regenera a partir de df_raw."""
    if PROCESSED_CSV.exists():
        print(f"[preprocessing] Carregando dataset processado: {PROCESSED_CSV}")
        return pd.read_csv(PROCESSED_CSV, low_memory=False)

    print("[preprocessing] Arquivo processed/ não encontrado. Regenerando...")
    if df_raw is None:
        from src.data_loader import load_raw
        df_raw = load_raw()
    return clean_data(df_raw)


def get_feature_lists(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Separa features em numéricas e categóricas a partir de X_train."""
    numeric_features     = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()

    print(f"[preprocessing] Features numéricas  ({len(numeric_features)}): {numeric_features}")
    print(f"[preprocessing] Features categóricas ({len(categorical_features)}): {categorical_features}")

    return numeric_features, categorical_features


def build_preprocessor(numeric_features: list[str], categorical_features: list[str]) -> ColumnTransformer:
    """Cria ColumnTransformer com imputação, scaling e OHE — não ajustado aqui."""
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


def split_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Divide em treino/teste estratificado (80/20); o teste é isolado para avaliação final."""
    from sklearn.model_selection import train_test_split

    X = df.drop(columns=[TARGET])
    y = df[TARGET].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)

    print(f"[preprocessing] Split treino/teste (stratify=y, random_state={RANDOM_STATE}):")
    print(f"  X_train : {X_train.shape[0]:,} linhas × {X_train.shape[1]} colunas")
    print(f"  X_test  : {X_test.shape[0]:,} linhas × {X_test.shape[1]} colunas")

    for label, y_part in [("Treino", y_train), ("Teste", y_test)]:
        vc  = y_part.value_counts()
        pct = y_part.value_counts(normalize=True) * 100
        print(f"  {label}: classe 0={vc.get(0,0):,} ({pct.get(0,0):.1f}%), "
              f"classe 1={vc.get(1,0):,} ({pct.get(1,0):.1f}%)")

    return X_train, X_test, y_train, y_test
