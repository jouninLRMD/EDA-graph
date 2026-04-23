"""Statistical analysis utilities: normality, group comparison, post-hoc."""
from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import anderson, kruskal
from statsmodels.stats.multitest import multipletests


def anderson_darling_by_class(
    df: pd.DataFrame, features: Sequence[str], class_col: str = "class", alpha: float = 0.05
) -> pd.DataFrame:
    """Anderson-Darling normality test per feature and class.

    The critical value for ``alpha = 0.05`` is at index 2 of the
    ``critical_values`` array returned by SciPy.
    """
    rows = []
    idx_alpha = {0.15: 0, 0.10: 1, 0.05: 2, 0.025: 3, 0.01: 4}.get(alpha, 2)
    for feat in features:
        for cls in sorted(df[class_col].unique()):
            data = df.loc[df[class_col] == cls, feat].dropna().to_numpy()
            if data.size < 8:
                rows.append((feat, cls, np.nan, np.nan, False))
                continue
            res = anderson(data, dist="norm")
            stat = float(res.statistic)
            crit = float(res.critical_values[idx_alpha])
            rows.append((feat, int(cls), stat, crit, stat < crit))
    return pd.DataFrame(rows, columns=["feature", "class", "statistic", "critical", "is_normal"])


def kruskal_wallis(
    df: pd.DataFrame, features: Sequence[str], class_col: str = "class"
) -> pd.DataFrame:
    """Kruskal-Wallis H-test across all classes for each feature."""
    rows = []
    classes = sorted(df[class_col].unique())
    for feat in features:
        groups = [df.loc[df[class_col] == c, feat].dropna().to_numpy() for c in classes]
        groups = [g for g in groups if g.size > 0]
        if len(groups) < 2:
            rows.append((feat, np.nan, np.nan))
            continue
        h, p = kruskal(*groups)
        rows.append((feat, float(h), float(p)))
    return pd.DataFrame(rows, columns=["feature", "H", "p_value"])


def dunn_posthoc_fdr(
    df: pd.DataFrame,
    features: Sequence[str],
    class_col: str = "class",
    p_adjust: str = "holm",
    fdr_method: str = "fdr_bh",
) -> Dict[str, pd.DataFrame]:
    """Per-feature Dunn post-hoc test followed by Benjamini-Hochberg FDR.

    Returns a dict ``{feature: DataFrame}`` where each DataFrame contains
    the class-by-class adjusted p-values.
    """
    import scikit_posthocs as sp  # lazy import

    out: Dict[str, pd.DataFrame] = {}
    for feat in features:
        sub = df[[feat, class_col]].dropna()
        posthoc = sp.posthoc_dunn(sub, val_col=feat, group_col=class_col, p_adjust=p_adjust)
        # FDR across all pairwise comparisons.
        p = posthoc.to_numpy().ravel()
        _, p_corr, _, _ = multipletests(p, method=fdr_method)
        posthoc.iloc[:, :] = p_corr.reshape(posthoc.shape)
        out[feat] = posthoc
    return out
