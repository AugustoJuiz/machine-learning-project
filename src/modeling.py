"""Definição, validação cruzada e tuning dos modelos de classificação."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV

from src.config import CV_FOLDS, RANDOM_STATE, SVM_SAMPLE_SIZE
from src.preprocessing import build_preprocessor


def get_models(numeric_features: list[str], categorical_features: list[str]) -> dict[str, SklearnPipeline]:
    """Retorna dicionário de Pipelines completos (preprocessor + classifier) para cada modelo."""
    def _make_pipeline(clf) -> SklearnPipeline:
        preprocessor = build_preprocessor(numeric_features, categorical_features)
        return SklearnPipeline([("preprocessor", preprocessor), ("classifier", clf)])

    models = {
        "Baseline": _make_pipeline(DummyClassifier(strategy="most_frequent", random_state=RANDOM_STATE)),
        "LogReg": _make_pipeline(LogisticRegression(class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE)),
        "DecTree": _make_pipeline(DecisionTreeClassifier(class_weight="balanced", random_state=RANDOM_STATE)),
        "RandomForest": _make_pipeline(RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1)),
        "SVM": _make_pipeline(CalibratedClassifierCV(LinearSVC(class_weight="balanced", max_iter=2000, random_state=RANDOM_STATE), cv=3)),
    }

    print(f"[modeling] {len(models)} modelos criados: {list(models.keys())}")
    return models


def get_svm_sample(X_train: pd.DataFrame, y_train: pd.Series, sample_size: int | None = None) -> tuple[pd.DataFrame, pd.Series]:
    """Retorna amostra estratificada do treino para LinearSVC (SVC com kernel RBF é O(n²) no dataset completo)."""
    if sample_size is None:
        sample_size = SVM_SAMPLE_SIZE

    n = min(sample_size, len(X_train))
    if n == len(X_train):
        print(f"[modeling] SVM: usando treino completo ({n:,} amostras).")
        return X_train, y_train

    from sklearn.model_selection import train_test_split
    X_svm, _, y_svm, _ = train_test_split(X_train, y_train, train_size=n, stratify=y_train, random_state=RANDOM_STATE)
    print(f"[modeling] SVM: amostra estratificada de {n:,} amostras (treino completo: {len(X_train):,}).")
    pos_pct = y_svm.mean() * 100
    print(f"  Classe 1 na amostra: {pos_pct:.1f}%")
    return X_svm, y_svm


def cross_validate_models(models: dict[str, SklearnPipeline], X_train: pd.DataFrame, y_train: pd.Series, X_svm: pd.DataFrame | None = None, y_svm: pd.Series | None = None) -> pd.DataFrame:
    """Avalia todos os modelos com validação cruzada estratificada (StratifiedKFold k=5)."""
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scoring = ["f1", "roc_auc", "precision", "recall"]

    records = []
    print(f"\n[modeling] Validacao cruzada — StratifiedKFold(n_splits={CV_FOLDS})\n")

    for name, pipeline in models.items():
        if name == "SVM" and X_svm is not None:
            X_cv, y_cv = X_svm, y_svm
        else:
            X_cv, y_cv = X_train, y_train

        print(f"  [{name}] treinando {CV_FOLDS} folds... ", end="", flush=True)
        try:
            cv_results = cross_validate(pipeline, X_cv, y_cv, cv=cv, scoring=scoring, n_jobs=-1, return_train_score=False)
            record = {
                "modelo":          name,
                "f1_mean":         round(cv_results["test_f1"].mean(), 4),
                "f1_std":          round(cv_results["test_f1"].std(), 4),
                "roc_auc_mean":    round(cv_results["test_roc_auc"].mean(), 4),
                "roc_auc_std":     round(cv_results["test_roc_auc"].std(), 4),
                "precision_mean":  round(cv_results["test_precision"].mean(), 4),
                "recall_mean":     round(cv_results["test_recall"].mean(), 4),
            }
            print(f"F1={record['f1_mean']:.4f} (+/-{record['f1_std']:.4f})  ROC-AUC={record['roc_auc_mean']:.4f}")
        except Exception as exc:
            print(f"ERRO: {exc}")
            record = {"modelo": name, "f1_mean": None, "f1_std": None, "roc_auc_mean": None, "roc_auc_std": None, "precision_mean": None, "recall_mean": None}
        records.append(record)

    df_cv = pd.DataFrame(records).sort_values("f1_mean", ascending=False)
    print(f"\n[modeling] Resultados de CV:\n{df_cv.to_string(index=False)}\n")
    return df_cv


def tune_hyperparameters(models: dict[str, SklearnPipeline], X_train: pd.DataFrame, y_train: pd.Series, X_svm: pd.DataFrame | None = None, y_svm: pd.Series | None = None) -> dict[str, SklearnPipeline]:
    """Busca de hiperparâmetros para DecTree, RandomForest e LogReg; treina Baseline e SVM com parâmetros fixos."""
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    tuned_models = dict(models)

    print("[modeling] Tuning DecTree (GridSearchCV)...")
    dt_param_grid = {
        "classifier__max_depth":         [3, 5, 8, None],
        "classifier__min_samples_split":  [2, 10, 20],
        "classifier__min_samples_leaf":   [1, 5, 10],
        "classifier__criterion":          ["gini", "entropy"],
    }
    dt_search = GridSearchCV(models["DecTree"], dt_param_grid, scoring="f1", cv=cv, n_jobs=-1, refit=True, verbose=0)
    dt_search.fit(X_train, y_train)
    tuned_models["DecTree"] = dt_search.best_estimator_
    print(f"  Melhores params: {dt_search.best_params_}")
    print(f"  Melhor F1 (CV): {dt_search.best_score_:.4f}")

    print("\n[modeling] Tuning RandomForest (RandomizedSearchCV, n_iter=20)...")
    rf_param_dist = {
        "classifier__n_estimators":      [50, 100, 200],
        "classifier__max_depth":         [5, 10, 20, None],
        "classifier__min_samples_split": [2, 5, 10],
        "classifier__min_samples_leaf":  [1, 3, 5],
    }
    rf_search = RandomizedSearchCV(models["RandomForest"], rf_param_dist, n_iter=20, scoring="f1", cv=cv, n_jobs=-1, refit=True, random_state=RANDOM_STATE, verbose=0)
    rf_search.fit(X_train, y_train)
    tuned_models["RandomForest"] = rf_search.best_estimator_
    print(f"  Melhores params: {rf_search.best_params_}")
    print(f"  Melhor F1 (CV): {rf_search.best_score_:.4f}")

    print("\n[modeling] Tuning LogReg (GridSearchCV, grade C)...")
    lr_param_grid = {"classifier__C": [0.01, 0.1, 1.0, 10.0]}
    lr_search = GridSearchCV(models["LogReg"], lr_param_grid, scoring="f1", cv=cv, n_jobs=-1, refit=True, verbose=0)
    lr_search.fit(X_train, y_train)
    tuned_models["LogReg"] = lr_search.best_estimator_
    print(f"  Melhores params: {lr_search.best_params_}")
    print(f"  Melhor F1 (CV): {lr_search.best_score_:.4f}")

    print("\n[modeling] Baseline: sem tuning (modelo majoritario).")
    print("[modeling] SVM: sem tuning (LinearSVC calibrado; custo proibiria busca).")

    print("\n[modeling] Treinando Baseline em X_train completo...")
    tuned_models["Baseline"].fit(X_train, y_train)

    print("[modeling] Treinando SVM em amostra estratificada...")
    X_s = X_svm if X_svm is not None else X_train
    y_s = y_svm if y_svm is not None else y_train
    tuned_models["SVM"].fit(X_s, y_s)

    print("\n[modeling] Tuning concluido.")
    return tuned_models
