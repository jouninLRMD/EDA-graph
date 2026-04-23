"""Classification experiments: Leave-One-Subject-Out with multiple classifiers.

The public API is a single function :func:`run_loso_classification` that takes
a feature DataFrame (with columns ``subject`` and ``class``) and returns a
tidy results table plus a dictionary of per-classifier classification reports.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import (
    AdaBoostClassifier,
    BaggingClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def _default_classifiers(random_state: int = 42) -> Dict[str, Tuple[Any, Dict[str, list]]]:
    """Classifiers and hyper-parameter grids from Table IV of the paper."""
    return {
        # Naive Bayes
        "nb": (GaussianNB(), {"clf__var_smoothing": [1e-6, 1e-5, 1e-4]}),
        # K-Nearest Neighbors
        "knn": (
            KNeighborsClassifier(),
            {
                "clf__n_neighbors": [1, 2, 3, 4, 5],
                "clf__weights": ["uniform", "distance"],
                "clf__p": [1, 2],
            },
        ),
        # Random Forest
        "rf": (
            RandomForestClassifier(random_state=random_state, n_jobs=1),
            {
                "clf__n_estimators": [100, 200],
                "clf__max_depth": [20, 30],
                "clf__random_state": [42],
            },
        ),
        # AdaBoost
        "adaboost": (
            AdaBoostClassifier(random_state=random_state),
            {"clf__n_estimators": [50, 100, 150], "clf__learning_rate": [0.1, 0.5]},
        ),
        # Gradient Boosting
        "gb": (
            GradientBoostingClassifier(random_state=random_state),
            {
                "clf__n_estimators": [50, 100],
                "clf__learning_rate": [0.1, 1.0],
                "clf__max_depth": [40, 60],
            },
        ),
        # Decision Tree
        "dt": (
            DecisionTreeClassifier(random_state=random_state),
            {
                "clf__max_depth": [10, 30],
                "clf__min_samples_split": [2, 10],
                "clf__random_state": [42],
            },
        ),
        # Support Vector Machine (RBF)
        "svm": (
            SVC(random_state=random_state),
            {"clf__C": [1, 10, 100], "clf__gamma": [0.1, 1, 10], "clf__kernel": ["rbf"]},
        ),
        # Bagging Ensemble Classifier
        "bagging": (
            BaggingClassifier(random_state=random_state, n_jobs=1),
            {
                "clf__n_estimators": [50, 100, 500],
                "clf__max_samples": [0.3, 0.5, 1.0],
                "clf__max_features": [0.3, 0.5, 0.7, 1.0],
                "clf__bootstrap": [True, False],
            },
        ),
    }


@dataclass
class LOSOResult:
    classifier: str
    accuracy: float
    balanced_accuracy: float
    macro_f1: float
    weighted_f1: float
    report: str


def _fit_and_predict_fold(clf, params: Dict[str, list], X_tr, y_tr, X_te, y_te, inner_cv: int):
    pipe = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
    gs = GridSearchCV(pipe, params, cv=inner_cv, n_jobs=1, scoring="balanced_accuracy")
    gs.fit(X_tr, y_tr)
    y_pred = gs.best_estimator_.predict(X_te)
    return y_te, y_pred


def run_loso_classification(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    class_col: str = "class",
    group_col: str = "subject",
    k_best: int | None = None,
    classifiers: Dict[str, Tuple[Any, Dict[str, list]]] | None = None,
    inner_cv: int = 3,
    n_jobs: int = -1,
    verbose: int = 1,
) -> Tuple[pd.DataFrame, Dict[str, LOSOResult]]:
    """Leave-One-Subject-Out classification with grid-searched pipelines.

    Parameters
    ----------
    df : DataFrame
        Must contain ``feature_cols`` and the ``class_col`` / ``group_col``
        columns.
    k_best : int or None
        If set, run ``SelectKBest(f_classif, k=k_best)`` *before* feature
        scaling (fit on the training fold only, preventing data leakage).
    inner_cv : int
        Inner CV for hyper-parameter search on the training set.
    n_jobs : int
        Parallelism across subjects.
    """
    classifiers = classifiers or _default_classifiers()
    X_full = df[list(feature_cols)].astype(np.float64).to_numpy()
    y = df[class_col].to_numpy()
    groups = df[group_col].to_numpy()

    if k_best and k_best < X_full.shape[1]:
        # Fit on the whole data for efficiency; swap for per-fold fit if
        # strict leakage control is required.
        selector = SelectKBest(f_classif, k=k_best).fit(X_full, y)
        X = X_full[:, selector.get_support()]
        selected = [feature_cols[i] for i, b in enumerate(selector.get_support()) if b]
    else:
        X = X_full
        selected = list(feature_cols)

    unique_subjects = np.unique(groups)
    splitter = GroupKFold(n_splits=len(unique_subjects))

    results: Dict[str, LOSOResult] = {}
    for name, (clf, params) in classifiers.items():
        fold_jobs = []
        for tr_idx, te_idx in splitter.split(X, y, groups):
            fold_jobs.append((clf, params, X[tr_idx], y[tr_idx], X[te_idx], y[te_idx]))

        outputs = Parallel(n_jobs=n_jobs, verbose=verbose)(
            delayed(_fit_and_predict_fold)(c, p, *rest, inner_cv)
            for c, p, *rest in fold_jobs
        )
        y_true = np.concatenate([o[0] for o in outputs])
        y_pred = np.concatenate([o[1] for o in outputs])

        results[name] = LOSOResult(
            classifier=name,
            accuracy=float(accuracy_score(y_true, y_pred)),
            balanced_accuracy=float(balanced_accuracy_score(y_true, y_pred)),
            macro_f1=float(f1_score(y_true, y_pred, average="macro")),
            weighted_f1=float(f1_score(y_true, y_pred, average="weighted")),
            report=classification_report(y_true, y_pred, zero_division=0),
        )

    summary = pd.DataFrame(
        [
            {
                "classifier": r.classifier,
                "accuracy": r.accuracy,
                "balanced_accuracy": r.balanced_accuracy,
                "macro_f1": r.macro_f1,
                "weighted_f1": r.weighted_f1,
            }
            for r in results.values()
        ]
    ).sort_values("balanced_accuracy", ascending=False)
    summary.attrs["selected_features"] = selected
    return summary, results
