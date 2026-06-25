"""Helpers de plotagem e salvamento de figuras."""

import matplotlib.pyplot as plt
import seaborn as sns

from src.config import FIGURES_DIR, FIGURE_DPI, FIGURE_SIZE, PALETTE


def setup_style() -> None:
    """Configura o tema visual padrão (whitegrid, paleta Set2, fontes legíveis)."""
    sns.set_theme(style="whitegrid", palette=PALETTE, font_scale=1.1)
    plt.rcParams.update({
        "figure.dpi":       FIGURE_DPI,
        "savefig.dpi":      FIGURE_DPI,
        "figure.figsize":   FIGURE_SIZE,
        "axes.titlesize":   13,
        "axes.titleweight": "bold",
        "axes.labelsize":   11,
        "xtick.labelsize":  9,
        "ytick.labelsize":  9,
        "legend.fontsize":  9,
        "figure.facecolor": "white",
        "axes.facecolor":   "white",
    })


def save_fig(fig: plt.Figure, filename: str, tight: bool = True) -> None:
    """Salva uma figura em outputs/figures/ com DPI padrão."""
    if tight:
        try:
            fig.tight_layout()
        except Exception:
            pass

    if not filename.endswith(".png"):
        filename = filename + ".png"

    output_path = FIGURES_DIR / filename
    fig.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight", facecolor="white")
    print(f"[visualization] Figura salva: {output_path}")


def add_value_labels(ax: plt.Axes, fmt: str = "{:.1f}", fontsize: int = 9, padding: float = 3) -> None:
    """Adiciona rótulos de valor no topo de cada barra de um gráfico."""
    for patch in ax.patches:
        height = patch.get_height()
        if height == 0:
            continue
        ax.annotate(
            fmt.format(height),
            xy=(patch.get_x() + patch.get_width() / 2, height),
            xytext=(0, padding),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=fontsize,
        )
