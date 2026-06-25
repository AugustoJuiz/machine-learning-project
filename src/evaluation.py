"""Avaliação dos modelos no conjunto de teste e geração das figuras de análise."""

from __future__ import annotations

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
)
from sklearn.pipeline import Pipeline as SklearnPipeline

from src.config import MODELS_DIR, TABLES_DIR, TARGET
from src.visualization import save_fig, setup_style


def evaluate_on_test(name: str, pipeline: SklearnPipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Calcula accuracy, precision, recall, F1, ROC-AUC e AP para um modelo no conjunto de teste."""
    y_pred = pipeline.predict(X_test)

    if hasattr(pipeline, "predict_proba"):
        y_prob = pipeline.predict_proba(X_test)[:, 1]
    else:
        y_prob = None

    metrics = {
        "modelo":        name,
        "accuracy":      round(accuracy_score(y_test, y_pred), 4),
        "precision":     round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall":        round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1":            round(f1_score(y_test, y_pred, zero_division=0), 4),
        "roc_auc":       round(roc_auc_score(y_test, y_prob) if y_prob is not None else 0, 4),
        "avg_precision": round(average_precision_score(y_test, y_prob) if y_prob is not None else 0, 4),
        "y_pred": y_pred,
        "y_prob": y_prob,
    }
    return metrics


def build_metrics_table(results: list[dict]) -> pd.DataFrame:
    """Gera e salva tabela comparativa de métricas em outputs/tables/metrics_comparison.csv."""
    cols_display = ["modelo", "accuracy", "precision", "recall", "f1", "roc_auc", "avg_precision"]
    rows = [{k: r[k] for k in cols_display} for r in results]
    df = pd.DataFrame(rows).sort_values("f1", ascending=False).reset_index(drop=True)

    output_path = TABLES_DIR / "metrics_comparison.csv"
    df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"[evaluation] Tabela de metricas salva: {output_path}")
    print(df.to_string(index=False))
    return df


def plot_confusion_matrix(name: str, y_test: pd.Series, y_pred: np.ndarray) -> plt.Figure:
    """Gera a matriz de confusão (absoluta e percentual) do modelo indicado — Fig 8."""
    setup_style()
    cm = confusion_matrix(y_test, y_pred)
    cm_pct = cm / cm.sum() * 100

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0], xticklabels=["Pred: No", "Pred: Yes"], yticklabels=["Real: No", "Real: Yes"], linewidths=0.5, linecolor="white", cbar_kws={"label": "Contagem"})
    axes[0].set_title(f"Matriz de Confusao — {name}\n(valores absolutos)")

    annot = np.array([[f"{v:.1f}%" for v in row] for row in cm_pct])
    sns.heatmap(cm_pct, annot=annot, fmt="", cmap="Blues", ax=axes[1], xticklabels=["Pred: No", "Pred: Yes"], yticklabels=["Real: No", "Real: Yes"], linewidths=0.5, linecolor="white", cbar_kws={"label": "Percentual (%)"})
    axes[1].set_title(f"Matriz de Confusao — {name}\n(percentual do total)")

    fig.suptitle("TN = Verdadeiro Negativo | FP = Falso Positivo | FN = Falso Negativo | TP = Verdadeiro Positivo", fontsize=9, style="italic", y=1.01)

    save_fig(fig, "08_confusion_matrix")
    return fig


def plot_roc_curves(results: list[dict], y_test: pd.Series) -> plt.Figure:
    """Gera curvas ROC de todos os modelos em um gráfico — Fig 9."""
    setup_style()
    palette = sns.color_palette("tab10", n_colors=len(results))

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1, label="Aleatório (AUC = 0.50)")

    for i, res in enumerate(results):
        if res["y_prob"] is None:
            continue
        fpr, tpr, _ = roc_curve(y_test, res["y_prob"])
        auc = res["roc_auc"]
        ax.plot(fpr, tpr, color=palette[i], linewidth=1.8, label=f"{res['modelo']} (AUC = {auc:.4f})")

    ax.set_xlabel("Taxa de Falso Positivo (FPR)")
    ax.set_ylabel("Taxa de Verdadeiro Positivo (TPR / Recall)")
    ax.set_title("Curvas ROC — Comparacao entre Modelos")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.01])

    save_fig(fig, "09_roc_curves")
    return fig


def plot_pr_curve(name: str, y_test: pd.Series, y_prob: np.ndarray) -> plt.Figure:
    """Gera a curva Precision-Recall do modelo — Fig 10."""
    setup_style()
    precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_prob)
    ap = average_precision_score(y_test, y_prob)
    baseline = y_test.mean()

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall_vals, precision_vals, linewidth=2, color=sns.color_palette("tab10")[0], label=f"{name} (AP = {ap:.4f})")
    ax.axhline(y=baseline, linestyle="--", color="gray", linewidth=1, label=f"Baseline aleatorio (AP = {baseline:.4f})")
    ax.set_xlabel("Recall (Sensibilidade)")
    ax.set_ylabel("Precision (Precisao)")
    ax.set_title(f"Curva Precision-Recall — {name}")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])

    save_fig(fig, "10_pr_curve")
    return fig


def plot_metrics_comparison(df_metrics: pd.DataFrame) -> plt.Figure:
    """Gera gráfico de barras comparando accuracy, precision, recall, F1 e ROC-AUC — Fig 11."""
    setup_style()
    metric_cols = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    df_plot = df_metrics.set_index("modelo")[metric_cols]

    fig, ax = plt.subplots(figsize=(12, 5))
    df_plot.plot(kind="bar", ax=ax, edgecolor="white", linewidth=0.5, colormap="Set2", rot=30, width=0.75)

    ax.set_title("Comparacao de Metricas entre Modelos\n(conjunto de teste)")
    ax.set_xlabel("")
    ax.set_ylabel("Valor da Metrica")
    ax.set_ylim(0, 1.12)
    ax.legend(title="Metrica", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=9)
    ax.axhline(y=1.0, linestyle="--", color="gray", linewidth=0.6, alpha=0.5)

    save_fig(fig, "11_metrics_comparison")
    return fig


def feature_importance(results: list[dict], tuned_models: dict[str, SklearnPipeline], X_train: pd.DataFrame) -> plt.Figure:
    """Gera gráfico de importância das variáveis para RandomForest e LogReg — Fig 12."""
    setup_style()

    def _get_feature_names(pipeline: SklearnPipeline) -> list[str]:
        """Obtém nomes das features após transformação do ColumnTransformer."""
        try:
            preprocessor = pipeline.named_steps["preprocessor"]
            return list(preprocessor.get_feature_names_out())
        except Exception:
            return [f"feature_{i}" for i in range(100)]

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    palette = sns.color_palette("viridis", n_colors=20)

    ax0 = axes[0]
    rf_pipe = tuned_models.get("RandomForest")
    if rf_pipe is not None:
        try:
            feature_names = _get_feature_names(rf_pipe)
            importances   = rf_pipe.named_steps["classifier"].feature_importances_
            top_n = 20
            indices = np.argsort(importances)[-top_n:]
            top_names = [feature_names[i] if i < len(feature_names) else f"feat_{i}" for i in indices]
            top_vals  = importances[indices]
            top_names_short = [n.split("__")[-1][:30] for n in top_names]

            ax0.barh(range(top_n), top_vals, color=palette[::-1], edgecolor="white")
            ax0.set_yticks(range(top_n))
            ax0.set_yticklabels(top_names_short, fontsize=8)
            ax0.set_title(f"Random Forest\nTop {top_n} features por importancia")
            ax0.set_xlabel("Importancia (Mean Decrease Impurity)")
        except Exception as e:
            ax0.text(0.5, 0.5, f"Erro: {e}", ha="center", transform=ax0.transAxes)

    ax1 = axes[1]
    lr_pipe = tuned_models.get("LogReg")
    if lr_pipe is not None:
        try:
            feature_names = _get_feature_names(lr_pipe)
            coefs = np.abs(lr_pipe.named_steps["classifier"].coef_[0])
            top_n = 20
            indices = np.argsort(coefs)[-top_n:]
            top_names = [feature_names[i] if i < len(feature_names) else f"feat_{i}" for i in indices]
            top_vals  = coefs[indices]
            top_names_short = [n.split("__")[-1][:30] for n in top_names]

            ax1.barh(range(top_n), top_vals, color=palette[::-1], edgecolor="white")
            ax1.set_yticks(range(top_n))
            ax1.set_yticklabels(top_names_short, fontsize=8)
            ax1.set_title(f"Logistic Regression\nTop {top_n} features por |coeficiente|")
            ax1.set_xlabel("|Coeficiente| (escala padronizada)")
        except Exception as e:
            ax1.text(0.5, 0.5, f"Erro: {e}", ha="center", transform=ax1.transAxes)

    fig.suptitle("Importancia das Variaveis\nVariaveis com maior relevancia para a predicao de RainTomorrow", fontsize=13)

    save_fig(fig, "12_feature_importance")
    return fig


def select_best_model(df_metrics: pd.DataFrame, tuned_models: dict[str, SklearnPipeline], priority_metric: str = "f1") -> tuple[str, SklearnPipeline]:
    """Seleciona o melhor modelo por F1 (excluindo Baseline)."""
    df_valid = df_metrics.dropna(subset=[priority_metric])
    df_valid = df_valid[df_valid["modelo"] != "Baseline"]

    if df_valid.empty:
        raise ValueError("[evaluation] Nenhum modelo valido encontrado na tabela de metricas.")

    best_name = df_valid.loc[df_valid[priority_metric].idxmax(), "modelo"]
    best_pipeline = tuned_models[best_name]

    best_row = df_valid[df_valid["modelo"] == best_name].iloc[0]
    print(f"\n[evaluation] Melhor modelo selecionado: {best_name}")
    print(f"  F1       = {best_row.get('f1', '-')}")
    print(f"  ROC-AUC  = {best_row.get('roc_auc', '-')}")
    print(f"  Recall   = {best_row.get('recall', '-')}")

    return best_name, best_pipeline


def save_best_model(name: str, pipeline: SklearnPipeline) -> None:
    """Serializa o melhor Pipeline em models/best_model.joblib."""
    import joblib

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = MODELS_DIR / "best_model.joblib"
    joblib.dump({"model_name": name, "pipeline": pipeline}, path)
    print(f"[evaluation] Melhor modelo salvo: {path}")


def print_classification_reports(results: list[dict], y_test: pd.Series) -> None:
    """Imprime o classification_report do sklearn para cada modelo."""
    sep = "=" * 60
    for res in results:
        print(f"\n{sep}")
        print(f"Classification Report — {res['modelo']}")
        print(sep)
        print(classification_report(y_test, res["y_pred"], target_names=["Nao chove (0)", "Chove (1)"], zero_division=0))
