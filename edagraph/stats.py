"""Statistical analysis utilities.

Mirrors Section II-E of the paper:

1. **Anderson-Darling** normality test for each feature and class
   (Section II-E, first paragraph).
2. **Kruskal-Wallis** one-way ANOVA on ranks - features with
   :math:`p < 0.05` move on to pairwise comparisons.
3. **Dunn's** post-hoc pair-wise test with **Holm-Bonferroni**
   correction (paper threshold: adjusted :math:`p < 0.005`).

A Benjamini-Hochberg FDR step is also exposed for convenience but is
**off** by default to match the paper's protocol.
"""
from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
import pandas as pd
from scipy.stats import anderson, kruskal
from statsmodels.stats.multitest import multipletests


def anderson_darling_by_class(
    df: pd.DataFrame,
    features: Sequence[str],
    class_col: str = "class",
    alpha: float = 0.05,
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
                rows.append((feat, int(cls), np.nan, np.nan, False))
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


def dunn_posthoc(
    df: pd.DataFrame,
    features: Sequence[str],
    class_col: str = "class",
    p_adjust: str = "holm",
    alpha: float = 0.005,
    apply_fdr: bool = False,
    fdr_method: str = "fdr_bh",
) -> Dict[str, pd.DataFrame]:
    """Per-feature Dunn post-hoc test with Holm-Bonferroni correction.

    Parameters
    ----------
    p_adjust : str
        Correction method passed to :func:`scikit_posthocs.posthoc_dunn`.
        The paper uses Holm-Bonferroni (``"holm"``).
    alpha : float
        Threshold used in the ``significant`` column (paper: ``0.005``).
    apply_fdr : bool
        If ``True``, apply an additional Benjamini-Hochberg FDR correction
        on top of the Dunn/Holm p-values (disabled by default to match
        the paper).

    Returns
    -------
    dict ``{feature: DataFrame}``
        Each DataFrame is the class-by-class p-value matrix. A companion
        boolean matrix of ``significant = p < alpha`` is also stored in
        ``df.attrs['significant']``.
    """
    import scikit_posthocs as sp  # lazy import

    out: Dict[str, pd.DataFrame] = {}
    for feat in features:
        sub = df[[feat, class_col]].dropna()
        posthoc = sp.posthoc_dunn(sub, val_col=feat, group_col=class_col, p_adjust=p_adjust)
        if apply_fdr:
            p = posthoc.to_numpy().ravel()
            _, p_corr, _, _ = multipletests(p, method=fdr_method)
            posthoc.iloc[:, :] = p_corr.reshape(posthoc.shape)
        posthoc.attrs["significant"] = posthoc < alpha
        out[feat] = posthoc
    return out


# Backward-compatible alias used by older scripts.
def dunn_posthoc_fdr(*args, **kwargs):
    """Deprecated - use :func:`dunn_posthoc` (FDR now off by default)."""
    kwargs.setdefault("apply_fdr", True)
    return dunn_posthoc(*args, **kwargs)
