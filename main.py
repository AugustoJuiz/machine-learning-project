"""
main.py — Orquestrador do pipeline de ciência de dados.

Executa as etapas do projeto em sequência:
    1. Download/carregamento do dataset
    2. Inspeção básica e dicionário de dados
    3. Análise exploratória (EDA) — gera figuras 1–6
    4. Limpeza estrutural e split treino/teste
    5. Validação cruzada dos modelos
    6. Tuning de hiperparâmetros
    7. Avaliação final no conjunto de teste — gera figuras 7–10
    8. Exportação do melhor modelo

Uso:
    python main.py            # pipeline completo
    python main.py --skip-eda # pula a EDA (usa processed/ se existir)
    python main.py --skip-tuning # pula tuning (usa parametros padrao)
"""

import argparse
import sys
import time
from pathlib import Path


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Pipeline de predicao de chuva — Rain in Australia"
    )
    parser.add_argument(
        "--skip-eda",
        action="store_true",
        help="Pula a EDA (carrega processed/ se existir, senao gera sem figuras).",
    )
    parser.add_argument(
        "--skip-tuning",
        action="store_true",
        help="Pula o tuning; treina modelos com parametros padrao.",
    )
    return parser.parse_args()


def _banner(titulo: str) -> None:
    sep = "=" * 65
    print(f"\n{sep}")
    print(f"  {titulo}")
    print(f"{sep}")


def run_pipeline(skip_eda: bool = False, skip_tuning: bool = False) -> None:
    """
    Executa o pipeline completo ponta a ponta.

    Parâmetros:
        skip_eda    : se True, pula geração das figuras de EDA.
        skip_tuning : se True, pula busca de hiperparâmetros.
    """
    t_inicio = time.time()

    # ── 1. Carregamento ─────────────────────────────────────────────────────────
    _banner("ETAPA 1 — Carregamento do Dataset")
    from src.data_loader import download_dataset, load_raw, basic_inspect, build_data_dictionary

    download_dataset()
    df_raw = load_raw()
    basic_inspect(df_raw)

    # ── 2. Dicionário de dados ──────────────────────────────────────────────────
    _banner("ETAPA 2a — Dicionário de Dados")
    build_data_dictionary(df_raw)

    # ── 3. EDA ──────────────────────────────────────────────────────────────────
    if not skip_eda:
        _banner("ETAPA 2b — Análise Exploratória (EDA)")
        from src.eda import run_full_eda
        run_full_eda(df_raw)
    else:
        print("\n[main] EDA ignorada (--skip-eda).")

    # ── 4. Pré-processamento e split ─────────────────────────────────────────────
    _banner("ETAPA 3 — Pré-processamento e Split Treino/Teste")
    from src.preprocessing import load_processed, split_data, get_feature_lists

    df_clean = load_processed(df_raw)
    X_train, X_test, y_train, y_test = split_data(df_clean)
    numeric_features, categorical_features = get_feature_lists(X_train)

    # ── 5. Criação dos modelos e amostra SVM ─────────────────────────────────────
    _banner("ETAPA 4a — Definição dos Modelos")
    from src.modeling import get_models, get_svm_sample

    models = get_models(numeric_features, categorical_features)
    X_svm, y_svm = get_svm_sample(X_train, y_train)

    # ── 6. Validação cruzada ─────────────────────────────────────────────────────
    _banner("ETAPA 4b — Validação Cruzada (StratifiedKFold k=5)")
    from src.modeling import cross_validate_models

    df_cv = cross_validate_models(models, X_train, y_train, X_svm, y_svm)

    # ── 7. Tuning ───────────────────────────────────────────────────────────────
    if not skip_tuning:
        _banner("ETAPA 4c — Tuning de Hiperparâmetros")
        from src.modeling import tune_hyperparameters
        tuned_models = tune_hyperparameters(models, X_train, y_train, X_svm, y_svm)
    else:
        print("\n[main] Tuning ignorado (--skip-tuning). Treinando com parametros padrao...")
        from src.modeling import tune_hyperparameters
        # treina apenas Baseline e SVM (nao passam por busca)
        tuned_models = dict(models)
        tuned_models["Baseline"].fit(X_train, y_train)
        X_s = X_svm if X_svm is not None else X_train
        y_s = y_svm if y_svm is not None else y_train
        tuned_models["SVM"].fit(X_s, y_s)
        # treina restantes com parametros padrao
        for name in ["LogReg", "DecTree", "RandomForest"]:
            print(f"  Treinando {name}...")
            tuned_models[name].fit(X_train, y_train)

    # ── 8. Avaliação final ───────────────────────────────────────────────────────
    _banner("ETAPA 5a — Avaliação Final no Conjunto de Teste")
    from src.evaluation import (
        evaluate_on_test,
        build_metrics_table,
        print_classification_reports,
        select_best_model,
        save_best_model,
        plot_confusion_matrix,
        plot_roc_curves,
        plot_pr_curve,
        plot_metrics_comparison,
        feature_importance,
    )

    results = []
    for name, pipeline in tuned_models.items():
        print(f"  Avaliando {name}...")
        res = evaluate_on_test(name, pipeline, X_test, y_test)
        results.append(res)

    print_classification_reports(results, y_test)
    df_metrics = build_metrics_table(results)

    # ── 9. Melhor modelo ─────────────────────────────────────────────────────────
    _banner("ETAPA 5b — Seleção e Exportação do Melhor Modelo")
    best_name, best_pipeline = select_best_model(df_metrics, tuned_models)
    save_best_model(best_name, best_pipeline)

    # ── 10. Figuras de avaliação ──────────────────────────────────────────────────
    _banner("ETAPA 5c — Geração de Figuras (Fig 7–10)")
    import matplotlib
    matplotlib.use("Agg")  # backend sem display para execucao via terminal
    import matplotlib.pyplot as plt

    best_result = next(r for r in results if r["modelo"] == best_name)

    print("[main] Fig 7 — Matriz de confusão...")
    plot_confusion_matrix(best_name, y_test, best_result["y_pred"])
    plt.close("all")

    print("[main] Fig 8 — Curvas ROC...")
    plot_roc_curves(results, y_test)
    plt.close("all")

    if best_result["y_prob"] is not None:
        print("[main] Fig 9a — Curva Precision-Recall...")
        plot_pr_curve(best_name, y_test, best_result["y_prob"])
        plt.close("all")

    print("[main] Fig 9b — Comparativo de métricas...")
    plot_metrics_comparison(df_metrics)
    plt.close("all")

    print("[main] Fig 10 — Importância das variáveis...")
    feature_importance(results, tuned_models, X_train)
    plt.close("all")

    # ── Sumário final ─────────────────────────────────────────────────────────────
    _banner("PIPELINE CONCLUIDO")
    from src.config import FIGURES_DIR, TABLES_DIR, MODELS_DIR

    t_total = time.time() - t_inicio
    print(f"\nTempo total: {t_total:.1f}s ({t_total/60:.1f} min)\n")

    print(f"Melhor modelo: {best_name}")
    best_row = df_metrics[df_metrics["modelo"] == best_name].iloc[0]
    print(f"  F1      = {best_row['f1']}")
    print(f"  ROC-AUC = {best_row['roc_auc']}")
    print(f"  Recall  = {best_row['recall']}")

    print(f"\nFiguras salvas em: {FIGURES_DIR}")
    for f in sorted(FIGURES_DIR.glob("*.png")):
        print(f"  {f.name}")

    print(f"\nTabelas salvas em: {TABLES_DIR}")
    for f in sorted(TABLES_DIR.glob("*.csv")):
        print(f"  {f.name}")

    print(f"\nModelo exportado: {MODELS_DIR / 'best_model.joblib'}")


if __name__ == "__main__":
    args = _parse_args()
    run_pipeline(skip_eda=args.skip_eda, skip_tuning=args.skip_tuning)
