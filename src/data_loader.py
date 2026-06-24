"""
data_loader.py — Carregamento, inspeção e dicionário de dados.

Funções:
    download_dataset()      — baixa o dataset via Kaggle API (opcional).
    load_raw()              — lê o CSV bruto local com mensagem clara se ausente.
    basic_inspect()         — exibe shape, tipos, head, distribuição do alvo e % ausentes.
    build_data_dictionary() — gera dicionário de dados como DataFrame e salva em CSV.
"""

import sys
import textwrap
from pathlib import Path

import pandas as pd

from src.config import (
    KAGGLE_DATASET,
    KAGGLE_FILENAME,
    RAW_CSV,
    RAW_DIR,
    TABLES_DIR,
    TARGET,
)


# ── Mapeamento de relevância meteorológica das colunas ──────────────────────────
# Usado pelo dicionário de dados. Campos decisão/justificativa são preenchidos
# empiricamente na EDA (Etapa 2) com base nos dados reais.
_METEO_RELEVANCE: dict[str, str] = {
    "Date":            "Data da observação; usada para extrair Mês e Estação do ano.",
    "Location":        "Estação meteorológica; captura variações geográficas no clima.",
    "MinTemp":         "Temperatura mínima diária — influência indireta na umidade relativa.",
    "MaxTemp":         "Temperatura máxima diária — relacionada à evapotranspiração.",
    "Rainfall":        "Precipitação acumulada no dia — preditor direto de umidade e chuva futura.",
    "Evaporation":     "Evaporação de bandeja — indica condições de umidade do ar; alto % ausentes.",
    "Sunshine":        "Horas de sol — inversamente correlacionada com nebulosidade; alto % ausentes.",
    "WindGustDir":     "Direção da rajada máxima de vento.",
    "WindGustSpeed":   "Velocidade da rajada máxima — associada a sistemas de baixa pressão.",
    "WindDir9am":      "Direção do vento às 9h.",
    "WindDir3pm":      "Direção do vento às 15h.",
    "WindSpeed9am":    "Velocidade do vento às 9h.",
    "WindSpeed3pm":    "Velocidade do vento às 15h.",
    "Humidity9am":     "Umidade relativa às 9h.",
    "Humidity3pm":     "Umidade relativa às 15h — forte preditor de chuva no dia seguinte.",
    "Pressure9am":     "Pressão atmosférica às 9h.",
    "Pressure3pm":     "Pressão atmosférica às 15h — queda de pressão indica frentes chuvosas.",
    "Cloud9am":        "Nebulosidade às 9h (oktas 0–9); alto % ausentes.",
    "Cloud3pm":        "Nebulosidade às 15h (oktas 0–9) — associada a precipitação; alto % ausentes.",
    "Temp9am":         "Temperatura às 9h.",
    "Temp3pm":         "Temperatura às 15h.",
    "RainToday":       "Indica se choveu no dia atual (Yes/No) — preditor direto da variável-alvo.",
    "RainTomorrow":    "VARIÁVEL-ALVO: indica se choverá no dia seguinte (Yes → 1 / No → 0).",
}


def download_dataset() -> None:
    """
    Baixa o dataset Rain in Australia via Kaggle API.

    Comportamento:
        - Se `data/raw/weatherAUS.csv` já existir, pula o download.
        - Se a Kaggle API não estiver configurada, exibe instruções claras para
          download manual e encerra sem erro fatal.

    Retorno:
        None
    """
    if RAW_CSV.exists():
        print(f"[data_loader] Dataset já presente em: {RAW_CSV}")
        print("[data_loader] Download ignorado.")
        return

    print("[data_loader] Arquivo não encontrado. Tentando download via Kaggle API...")

    try:
        # Importação dentro da função para não forçar dependência em quem não usa
        from kaggle.api.kaggle_api_extended import KaggleApiExtended  # type: ignore
        api = KaggleApiExtended()
        api.authenticate()
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        api.dataset_download_files(
            KAGGLE_DATASET,
            path=str(RAW_DIR),
            unzip=True,
            quiet=False,
        )
        # Renomeia se o arquivo descompactado tiver nome diferente
        downloaded = RAW_DIR / KAGGLE_FILENAME
        if not downloaded.exists():
            # Tenta encontrar qualquer CSV na pasta
            csvs = list(RAW_DIR.glob("*.csv"))
            if csvs:
                csvs[0].rename(RAW_CSV)
                print(f"[data_loader] Arquivo renomeado para: {RAW_CSV.name}")
        print(f"[data_loader] Download concluído: {RAW_CSV}")

    except ImportError:
        _print_manual_fallback("O pacote 'kaggle' não está instalado.")
    except Exception as exc:
        _print_manual_fallback(str(exc))


def _print_manual_fallback(motivo: str) -> None:
    """Exibe instruções detalhadas de download manual e encerra sem exceção."""
    mensagem = textwrap.dedent(f"""
    ╔══════════════════════════════════════════════════════════════════════════════╗
    ║  KAGGLE API INDISPONÍVEL — SIGA AS INSTRUÇÕES ABAIXO                       ║
    ╠══════════════════════════════════════════════════════════════════════════════╣
    ║  Motivo: {motivo:<69}║
    ╠══════════════════════════════════════════════════════════════════════════════╣
    ║  OPÇÃO A — Download manual (mais simples):                                  ║
    ║  1. Acesse: https://www.kaggle.com/datasets/jsphyg/weather-dataset-rattle-package
    ║  2. Faça login no Kaggle (conta gratuita).                                  ║
    ║  3. Clique em "Download" e extraia o ZIP.                                   ║
    ║  4. Coloque o arquivo em:                                                   ║
    ║       machine-learning-project/data/raw/weatherAUS.csv                      ║
    ║                                                                              ║
    ║  OPÇÃO B — Configurar Kaggle API:                                           ║
    ║  1. Kaggle → perfil → Settings → API → "Create New Token" → kaggle.json    ║
    ║  2. Posicione o arquivo:                                                    ║
    ║       Windows:    %USERPROFILE%\\.kaggle\\kaggle.json                       ║
    ║       Linux/Mac:  ~/.kaggle/kaggle.json                                     ║
    ║  3. Instale: pip install kaggle                                              ║
    ║  4. Execute novamente: python -c "from src.data_loader import download_dataset; download_dataset()"
    ╚══════════════════════════════════════════════════════════════════════════════╝
    """)
    print(mensagem, file=sys.stderr)


def load_raw() -> pd.DataFrame:
    """
    Lê o CSV bruto do dataset Rain in Australia.

    Retorno:
        pd.DataFrame com os dados brutos.

    Levanta:
        SystemExit com mensagem clara se o arquivo não for encontrado.
    """
    if not RAW_CSV.exists():
        print(
            f"\n[ERRO] Arquivo não encontrado: {RAW_CSV}\n"
            "Execute uma das opções abaixo para obter o dataset:\n"
            "  1. Automático: from src.data_loader import download_dataset; download_dataset()\n"
            "  2. Manual:     coloque weatherAUS.csv em data/raw/  (ver README.md)\n",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"[data_loader] Carregando dataset: {RAW_CSV}")
    df = pd.read_csv(RAW_CSV, low_memory=False)
    print(f"[data_loader] Dataset carregado — {df.shape[0]:,} linhas × {df.shape[1]} colunas.")
    return df


def basic_inspect(df: pd.DataFrame) -> None:
    """
    Exibe inspeção básica do dataset: shape, tipos, primeiras linhas,
    distribuição da variável-alvo e percentual de ausentes por coluna.

    Parâmetros:
        df: DataFrame bruto retornado por load_raw().

    Retorno:
        None (imprime no stdout).
    """
    sep = "─" * 70

    print(f"\n{sep}")
    print("INSPEÇÃO BÁSICA DO DATASET")
    print(sep)

    print(f"\n● Shape: {df.shape[0]:,} linhas × {df.shape[1]} colunas")

    print("\n● Tipos de dados:")
    dtype_counts = df.dtypes.value_counts().to_dict()
    for dtype, count in dtype_counts.items():
        print(f"    {str(dtype):<12}: {count} coluna(s)")

    print("\n● Primeiras 3 linhas:")
    print(df.head(3).to_string())

    print(f"\n● Valores únicos de '{TARGET}':")
    if TARGET in df.columns:
        print(df[TARGET].value_counts(dropna=False).to_string())
    else:
        print(f"  [AVISO] Coluna '{TARGET}' não encontrada no dataset.")

    print("\n● Percentual de valores ausentes por coluna (apenas colunas com ausentes):")
    missing = (df.isnull().mean() * 100).sort_values(ascending=False)
    missing = missing[missing > 0]
    if missing.empty:
        print("  Nenhum valor ausente encontrado.")
    else:
        for col, pct in missing.items():
            barra = "█" * int(pct / 5)
            print(f"  {col:<20} {pct:5.1f}%  {barra}")

    print(f"\n{sep}\n")


def build_data_dictionary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Gera o dicionário de dados do dataset.

    Cria uma tabela com: nome da variável, tipo pandas, percentual de
    ausentes, relevância meteorológica e campos para preenchimento empírico
    na EDA (decisão de manter/remover e justificativa).

    Parâmetros:
        df: DataFrame bruto retornado por load_raw().

    Retorno:
        pd.DataFrame com o dicionário de dados.
        O arquivo também é salvo em outputs/tables/data_dictionary.csv.
    """
    records = []
    for col in df.columns:
        pct_missing = df[col].isnull().mean() * 100
        records.append({
            "variavel":               col,
            "tipo_pandas":            str(df[col].dtype),
            "valores_unicos":         df[col].nunique(dropna=False),
            "pct_ausentes":           round(pct_missing, 2),
            "relevancia_meteorologica": _METEO_RELEVANCE.get(col, "—"),
            "decisao":                "a definir na EDA",
            "justificativa":          "a definir na EDA",
        })

    dictionary = pd.DataFrame(records)

    # Salva em CSV
    output_path = TABLES_DIR / "data_dictionary.csv"
    dictionary.to_csv(output_path, index=False, encoding="utf-8")
    print(f"[data_loader] Dicionário de dados salvo em: {output_path}")

    return dictionary
