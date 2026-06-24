"""
eda.py — Análise Exploratória de Dados (EDA).

Funções:
    get_numeric_cols()          — retorna colunas numéricas (excluindo alvo e Date).
    get_categorical_cols()      — retorna colunas categóricas (excluindo alvo e Date).
    descriptive_stats()         — estatísticas descritivas das variáveis numéricas.
    outlier_analysis()          — análise de outliers pelo método IQR.
    plot_target_distribution()  — Fig 1: distribuição da variável-alvo RainTomorrow.
    plot_missing_values()       — Fig 2: percentual de ausentes por coluna.
    plot_numeric_histograms()   — Fig 3: histogramas das principais variáveis numéricas.
    plot_numeric_boxplots()     — Fig 4: boxplots das principais variáveis numéricas.
    plot_correlation_heatmap()  — Fig 5: heatmap de correlação entre variáveis numéricas.
    plot_rain_today_vs_tomorrow() — Fig 6: relação entre RainToday e RainTomorrow.
    run_full_eda()              — orquestra toda a EDA e retorna um dicionário de resultados.
"""

import math
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.config import PALETTE, RANDOM_STATE, TABLES_DIR, TARGET
from src.visualization import add_value_labels, save_fig, setup_style

# ── Colunas de interesse meteorológico para visualizações focadas ───────────────
# Filtradas contra as colunas reais do DataFrame na execução.
MAIN_NUMERIC_COLS = [
    "Humidity3pm",
    "Pressure3pm",
    "Temp3pm",
    "Rainfall",
    "WindGustSpeed",
    "Cloud3pm",
    "Humidity9am",
    "Pressure9am",
    "Temp9am",
    "MinTemp",
    "MaxTemp",
    "WindSpeed3pm",
    "WindSpeed9am",
    "Sunshine",
    "Evaporation",
    "Cloud9am",
]


# ── Helpers internos ─────────────────────────────────────────────────────────────

def get_numeric_cols(df: pd.DataFrame) -> list[str]:
    """
    Retorna lista de colunas numéricas do DataFrame, excluindo a variável-alvo
    e a coluna Date.

    Parâmetros:
        df : DataFrame bruto ou parcialmente processado.

    Retorno:
        Lista de nomes de colunas numéricas.
    """
    exclude = {TARGET, "Date"}
    return [c for c in df.select_dtypes(include="number").columns if c not in exclude]


def get_categorical_cols(df: pd.DataFrame) -> list[str]:
    """
    Retorna lista de colunas categóricas do DataFrame, excluindo a variável-alvo
    e a coluna Date.

    Parâmetros:
        df : DataFrame bruto ou parcialmente processado.

    Retorno:
        Lista de nomes de colunas categóricas (object/category).
    """
    exclude = {TARGET, "Date"}
    return [c for c in df.select_dtypes(include=["object", "category"]).columns
            if c not in exclude]


def _resolve_numeric_cols(df: pd.DataFrame, cols: Optional[list[str]]) -> list[str]:
    """Filtra MAIN_NUMERIC_COLS contra colunas existentes no DataFrame."""
    all_num = set(get_numeric_cols(df))
    if cols is not None:
        return [c for c in cols if c in all_num]
    priority = [c for c in MAIN_NUMERIC_COLS if c in all_num]
    remaining = [c for c in all_num if c not in set(priority)]
    return priority + remaining


# ── Estatísticas descritivas ─────────────────────────────────────────────────────

def descriptive_stats(
    df: pd.DataFrame,
    numeric_cols: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Calcula estatísticas descritivas das variáveis numéricas.

    Inclui: média, mediana, desvio padrão, mínimo, máximo, Q1 e Q3.
    O resultado é salvo em outputs/tables/descriptive_stats.csv.

    Parâmetros:
        df           : DataFrame bruto.
        numeric_cols : lista de colunas a analisar; usa todas numéricas se None.

    Retorno:
        pd.DataFrame com as estatísticas por coluna.
    """
    cols = _resolve_numeric_cols(df, numeric_cols)
    subset = df[cols]

    stats = pd.DataFrame({
        "media":          subset.mean(),
        "mediana":        subset.median(),
        "desvio_padrao":  subset.std(),
        "minimo":         subset.min(),
        "maximo":         subset.max(),
        "Q1_(25%)":       subset.quantile(0.25),
        "Q3_(75%)":       subset.quantile(0.75),
        "pct_ausentes":   (subset.isnull().mean() * 100).round(2),
    }).round(4)

    output_path = TABLES_DIR / "descriptive_stats.csv"
    stats.to_csv(output_path, encoding="utf-8")
    print(f"[eda] Estatísticas descritivas salvas em: {output_path}")
    return stats


# ── Análise de outliers ──────────────────────────────────────────────────────────

def outlier_analysis(
    df: pd.DataFrame,
    numeric_cols: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Analisa outliers nas variáveis numéricas usando o método IQR.

    Limites: inferior = Q1 − 1,5·IQR ; superior = Q3 + 1,5·IQR.
    O resultado é salvo em outputs/tables/outlier_analysis.csv.

    Parâmetros:
        df           : DataFrame bruto.
        numeric_cols : lista de colunas a analisar; usa todas numéricas se None.

    Retorno:
        pd.DataFrame com Q1, Q3, IQR, limites e contagem/percentual de outliers.
    """
    cols = _resolve_numeric_cols(df, numeric_cols)
    records = []
    for col in cols:
        serie = df[col].dropna()
        q1 = serie.quantile(0.25)
        q3 = serie.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        n_out = int(((serie < lower) | (serie > upper)).sum())
        records.append({
            "variavel":        col,
            "Q1":              round(q1, 4),
            "Q3":              round(q3, 4),
            "IQR":             round(iqr, 4),
            "limite_inferior": round(lower, 4),
            "limite_superior": round(upper, 4),
            "n_outliers":      n_out,
            "pct_outliers":    round(n_out / len(serie) * 100, 2),
        })

    result = pd.DataFrame(records).sort_values("pct_outliers", ascending=False)
    output_path = TABLES_DIR / "outlier_analysis.csv"
    result.to_csv(output_path, index=False, encoding="utf-8")
    print(f"[eda] Análise de outliers salva em: {output_path}")
    return result


# ── Fig 1: Distribuição da variável-alvo ────────────────────────────────────────

def plot_target_distribution(df: pd.DataFrame) -> plt.Figure:
    """
    Gera o gráfico de distribuição da variável-alvo RainTomorrow (Fig 1).

    Exibe contagem e percentual de cada classe (Yes/No).
    Salva em outputs/figures/01_target_distribution.png.

    Parâmetros:
        df : DataFrame bruto com coluna RainTomorrow.

    Retorno:
        plt.Figure
    """
    setup_style()
    counts = df[TARGET].value_counts(dropna=True)
    pcts   = df[TARGET].value_counts(normalize=True, dropna=True) * 100

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    palette   = sns.color_palette(PALETTE, n_colors=2)

    # Contagem absoluta
    ax0 = axes[0]
    bars = ax0.bar(counts.index, counts.values, color=palette, edgecolor="white", linewidth=0.8)
    ax0.set_title("Distribuição de RainTomorrow — Contagem Absoluta")
    ax0.set_xlabel("RainTomorrow")
    ax0.set_ylabel("Número de Observações")
    for bar, val in zip(bars, counts.values):
        ax0.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 200,
            f"{val:,}",
            ha="center", va="bottom", fontsize=10, fontweight="bold",
        )

    # Percentual
    ax1 = axes[1]
    wedge_colors = palette
    wedges, texts, autotexts = ax1.pie(
        pcts.values,
        labels=pcts.index,
        autopct="%1.1f%%",
        colors=wedge_colors,
        startangle=90,
        wedgeprops={"edgecolor": "white", "linewidth": 1.5},
    )
    for at in autotexts:
        at.set_fontsize(12)
        at.set_fontweight("bold")
    ax1.set_title("Distribuição de RainTomorrow — Proporção (%)")

    fig.suptitle(
        "Variável-Alvo: RainTomorrow\n"
        "O desbalanceamento entre classes influencia a escolha de métricas e modelos.",
        fontsize=11, y=1.02,
    )

    save_fig(fig, "01_target_distribution")
    return fig


# ── Fig 2: Valores ausentes ──────────────────────────────────────────────────────

def plot_missing_values(df: pd.DataFrame) -> plt.Figure:
    """
    Gera o gráfico de percentual de valores ausentes por coluna (Fig 2).

    Salva em outputs/figures/02_missing_values.png.

    Parâmetros:
        df : DataFrame bruto.

    Retorno:
        plt.Figure
    """
    setup_style()
    missing = (df.isnull().mean() * 100).sort_values(ascending=True)
    missing = missing[missing > 0]

    fig, ax = plt.subplots(figsize=(10, max(5, len(missing) * 0.35)))

    colors = ["#e74c3c" if v >= 40 else "#f39c12" if v >= 20 else "#3498db"
              for v in missing.values]

    bars = ax.barh(missing.index, missing.values, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Percentual de Valores Ausentes (%)")
    ax.set_title("Percentual de Valores Ausentes por Coluna")
    ax.axvline(x=40, color="red", linestyle="--", linewidth=1.2, alpha=0.7,
               label="Limiar de remoção (40%)")
    ax.axvline(x=20, color="orange", linestyle="--", linewidth=1.0, alpha=0.6,
               label="Referência (20%)")
    ax.legend(loc="lower right", fontsize=9)

    for bar, val in zip(bars, missing.values):
        ax.text(
            val + 0.3,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}%",
            va="center", fontsize=8,
        )

    ax.set_xlim(0, max(missing.values) + 8)

    fig.suptitle(
        "Colunas com alto percentual de ausentes são candidatas a remoção (avaliação empírica na EDA).",
        fontsize=9, style="italic", y=1.01,
    )

    save_fig(fig, "02_missing_values")
    return fig


# ── Fig 3: Histogramas ───────────────────────────────────────────────────────────

def plot_numeric_histograms(
    df: pd.DataFrame,
    numeric_cols: Optional[list[str]] = None,
    cols_per_row: int = 4,
) -> plt.Figure:
    """
    Gera histogramas (com KDE) das principais variáveis numéricas (Fig 3).

    Salva em outputs/figures/03_numeric_histograms.png.

    Parâmetros:
        df           : DataFrame bruto.
        numeric_cols : colunas a plotar; usa MAIN_NUMERIC_COLS filtrado se None.
        cols_per_row : número de colunas na grade de subplots.

    Retorno:
        plt.Figure
    """
    setup_style()
    cols = _resolve_numeric_cols(df, numeric_cols)
    n_cols_plot = cols_per_row
    n_rows = math.ceil(len(cols) / n_cols_plot)

    fig, axes = plt.subplots(n_rows, n_cols_plot,
                              figsize=(n_cols_plot * 4, n_rows * 3.2))
    axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]

    color = sns.color_palette(PALETTE)[0]

    for i, col in enumerate(cols):
        ax = axes_flat[i]
        data = df[col].dropna()
        ax.hist(data, bins=40, color=color, alpha=0.7, density=True, edgecolor="white")
        # KDE sobreposta
        try:
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(data)
            x_range = np.linspace(data.min(), data.max(), 300)
            ax.plot(x_range, kde(x_range), color="darkblue", linewidth=1.5)
        except Exception:
            pass
        ax.set_title(col, fontsize=10, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("Densidade")

    # Remove subplots não utilizados
    for j in range(len(cols), len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(
        "Distribuição das Principais Variáveis Numéricas\n"
        "(histograma com curva de densidade estimada — KDE)",
        fontsize=12, y=1.01,
    )

    save_fig(fig, "03_numeric_histograms")
    return fig


# ── Fig 4: Boxplots ──────────────────────────────────────────────────────────────

def plot_numeric_boxplots(
    df: pd.DataFrame,
    numeric_cols: Optional[list[str]] = None,
    cols_per_row: int = 4,
) -> plt.Figure:
    """
    Gera boxplots das principais variáveis numéricas para identificação de outliers (Fig 4).

    Salva em outputs/figures/04_numeric_boxplots.png.

    Parâmetros:
        df           : DataFrame bruto.
        numeric_cols : colunas a plotar; usa MAIN_NUMERIC_COLS filtrado se None.
        cols_per_row : número de colunas na grade de subplots.

    Retorno:
        plt.Figure
    """
    setup_style()
    cols = _resolve_numeric_cols(df, numeric_cols)
    n_cols_plot = cols_per_row
    n_rows = math.ceil(len(cols) / n_cols_plot)

    palette_list = sns.color_palette(PALETTE, n_colors=len(cols))

    fig, axes = plt.subplots(n_rows, n_cols_plot,
                              figsize=(n_cols_plot * 4, n_rows * 3.2))
    axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for i, col in enumerate(cols):
        ax = axes_flat[i]
        data = df[[col, TARGET]].dropna(subset=[col])
        # Boxplot agrupado por RainTomorrow quando possível
        if TARGET in data.columns and data[TARGET].nunique() > 1:
            sns.boxplot(
                x=TARGET, y=col, data=data,
                palette=palette_list[:2], ax=ax,
                fliersize=2, linewidth=0.8,
            )
            ax.set_xlabel("RainTomorrow", fontsize=8)
        else:
            sns.boxplot(y=data[col], ax=ax, color=palette_list[i],
                        fliersize=2, linewidth=0.8)
            ax.set_xlabel("")
        ax.set_title(col, fontsize=10, fontweight="bold")
        ax.set_ylabel("")

    for j in range(len(cols), len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(
        "Boxplots das Principais Variáveis Numéricas por RainTomorrow\n"
        "(pontos além dos whiskers são outliers pelo critério IQR)",
        fontsize=12, y=1.01,
    )

    save_fig(fig, "04_numeric_boxplots")
    return fig


# ── Fig 5: Heatmap de correlação ─────────────────────────────────────────────────

def plot_correlation_heatmap(
    df: pd.DataFrame,
    numeric_cols: Optional[list[str]] = None,
) -> plt.Figure:
    """
    Gera heatmap de correlação de Pearson entre as variáveis numéricas (Fig 5).

    Utiliza máscara triangular superior para evitar redundância.
    Salva em outputs/figures/05_correlation_heatmap.png.

    Parâmetros:
        df           : DataFrame bruto.
        numeric_cols : colunas a incluir; usa todas numéricas se None.

    Retorno:
        plt.Figure
    """
    setup_style()
    cols = _resolve_numeric_cols(df, numeric_cols)

    # Adiciona codificação binária da variável-alvo para incluir no heatmap
    df_temp = df[cols].copy()
    if TARGET in df.columns:
        target_map = {"Yes": 1, "No": 0}
        df_temp[TARGET] = df[TARGET].map(target_map)
        # Mantém alvo por último
        ordered_cols = cols + [TARGET]
        df_temp = df_temp[ordered_cols]
    else:
        df_temp = df_temp[cols]

    corr = df_temp.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))  # mascara triângulo superior

    n = len(corr)
    fig_size = max(10, n * 0.7)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.85))

    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        center=0,
        vmin=-1, vmax=1,
        linewidths=0.4,
        linecolor="white",
        cbar_kws={"shrink": 0.7, "label": "Correlação de Pearson"},
        ax=ax,
        annot_kws={"size": 7},
    )

    ax.set_title(
        "Heatmap de Correlação de Pearson — Variáveis Numéricas\n"
        "(triângulo inferior; diagonal = 1,00 implícita)",
        fontsize=12,
    )
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    ax.tick_params(axis="y", rotation=0, labelsize=8)

    save_fig(fig, "05_correlation_heatmap")
    return fig


# ── Fig 6: RainToday vs RainTomorrow ─────────────────────────────────────────────

def plot_rain_today_vs_tomorrow(df: pd.DataFrame) -> plt.Figure:
    """
    Gera gráfico comparativo entre RainToday e RainTomorrow (Fig 6).

    Mostra a proporção de RainTomorrow=Yes para cada valor de RainToday,
    evidenciando o poder preditivo direto da chuva atual sobre a chuva futura.
    Salva em outputs/figures/06_raintoday_vs_tomorrow.png.

    Parâmetros:
        df : DataFrame bruto com colunas RainToday e RainTomorrow.

    Retorno:
        plt.Figure
    """
    setup_style()
    col_today = "RainToday"

    if col_today not in df.columns:
        print(f"[eda] Coluna '{col_today}' não encontrada. Figura 6 ignorada.")
        return plt.figure()

    # Remove linhas com alvo ausente
    data = df[[col_today, TARGET]].dropna()
    palette = sns.color_palette(PALETTE, n_colors=2)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Painel A: contagem cruzada
    ax0 = axes[0]
    ct = pd.crosstab(data[col_today], data[TARGET])
    ct.plot(kind="bar", ax=ax0, color=palette, edgecolor="white", linewidth=0.7, rot=0)
    ax0.set_title("Contagem: RainToday × RainTomorrow")
    ax0.set_xlabel("RainToday")
    ax0.set_ylabel("Número de Observações")
    ax0.legend(title="RainTomorrow", labels=ct.columns.tolist())
    add_value_labels(ax0, fmt="{:.0f}", fontsize=8)

    # Painel B: proporção de RainTomorrow=Yes por grupo
    ax1 = axes[1]
    prop = (
        data.groupby(col_today)[TARGET]
        .value_counts(normalize=True)
        .unstack(fill_value=0)
        * 100
    )
    prop.plot(kind="bar", stacked=True, ax=ax1, color=palette, edgecolor="white",
              linewidth=0.7, rot=0)
    ax1.set_title("Proporção (%): RainToday × RainTomorrow")
    ax1.set_xlabel("RainToday")
    ax1.set_ylabel("Percentual (%)")
    ax1.set_ylim(0, 110)
    ax1.legend(title="RainTomorrow", labels=prop.columns.tolist(), loc="upper right")

    fig.suptitle(
        "Relação entre RainToday e RainTomorrow\n"
        "Dias com chuva hoje apresentam maior probabilidade de chuva amanhã.",
        fontsize=11, y=1.02,
    )

    save_fig(fig, "06_raintoday_vs_tomorrow")
    return fig


# ── Orquestrador ─────────────────────────────────────────────────────────────────

def run_full_eda(df: pd.DataFrame) -> dict:
    """
    Executa toda a análise exploratória de dados em sequência.

    Chama: descriptive_stats, outlier_analysis e todas as funções de plot (Fig 1–6).
    Salva todas as tabelas e figuras em seus diretórios correspondentes.

    Parâmetros:
        df : DataFrame bruto retornado por data_loader.load_raw().

    Retorno:
        Dicionário com chaves:
            'stats'    : pd.DataFrame de estatísticas descritivas.
            'outliers' : pd.DataFrame de análise de outliers.
            'figures'  : lista de plt.Figure gerados.
    """
    print("\n" + "═" * 60)
    print("ETAPA 2 — ANÁLISE EXPLORATÓRIA DE DADOS (EDA)")
    print("═" * 60)

    print("\n[eda] Calculando estatísticas descritivas...")
    stats = descriptive_stats(df)

    print("\n[eda] Analisando outliers (método IQR)...")
    outliers = outlier_analysis(df)

    figures = []

    print("\n[eda] Gerando Fig 1 — Distribuição da variável-alvo...")
    figures.append(plot_target_distribution(df))
    plt.close("all")

    print("[eda] Gerando Fig 2 — Valores ausentes...")
    figures.append(plot_missing_values(df))
    plt.close("all")

    print("[eda] Gerando Fig 3 — Histogramas...")
    figures.append(plot_numeric_histograms(df))
    plt.close("all")

    print("[eda] Gerando Fig 4 — Boxplots...")
    figures.append(plot_numeric_boxplots(df))
    plt.close("all")

    print("[eda] Gerando Fig 5 — Heatmap de correlação...")
    figures.append(plot_correlation_heatmap(df))
    plt.close("all")

    print("[eda] Gerando Fig 6 — RainToday vs RainTomorrow...")
    figures.append(plot_rain_today_vs_tomorrow(df))
    plt.close("all")

    print("\n[eda] EDA concluída. Figuras e tabelas salvas.")
    print("═" * 60 + "\n")

    return {"stats": stats, "outliers": outliers, "figures": figures}
