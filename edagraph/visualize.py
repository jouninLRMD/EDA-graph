"""Plotting helpers for diagnostic figures."""
from __future__ import annotations

from typing import Dict, Sequence

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from .config import CASE_CLASS_MAP


def plot_eda_graph(graph: nx.Graph, ax=None, title: str = "EDA-graph"):
    """Render an EDA-graph laid out by node value.

    Nodes are 1-D (they carry a single quantised amplitude). To obtain a
    readable 2-D layout we use a spring layout initialised from the node
    values on the y-axis.
    """
    ax = ax or plt.gca()
    values = {n: float(graph.nodes[n].get("value", n)) for n in graph.nodes}
    init = {n: (0.0, v) for n, v in values.items()}
    pos = nx.spring_layout(graph, pos=init, fixed=None, seed=0, k=1.0 / max(1, len(graph) ** 0.5))
    sizes = [50 + 25 * graph.degree(n) for n in graph.nodes]
    nx.draw_networkx_nodes(graph, pos, node_size=sizes, node_color="tab:blue", alpha=0.8, ax=ax)
    nx.draw_networkx_edges(graph, pos, alpha=0.3, ax=ax)
    labels = {n: f"{values[n]:.2f}" for n in graph.nodes}
    nx.draw_networkx_labels(graph, pos, labels=labels, font_size=8, ax=ax)
    ax.set_title(title)
    ax.set_axis_off()
    return ax


def boxplot_feature_by_class(
    df: pd.DataFrame, feature: str, class_col: str = "class", ax=None, class_map: Dict[int, str] | None = None
):
    ax = ax or plt.gca()
    class_map = class_map or CASE_CLASS_MAP
    classes = sorted(df[class_col].unique())
    data = [df.loc[df[class_col] == c, feature].dropna().to_numpy() for c in classes]
    bp = ax.boxplot(data, notch=True, patch_artist=True, showmeans=True, meanline=True, whis=(5, 95))
    palette = plt.get_cmap("tab10").colors
    for patch, color in zip(bp["boxes"], palette[: len(classes)]):
        patch.set_facecolor(color)
    ax.set_xticklabels([class_map.get(c, str(c)) for c in classes], rotation=0)
    ax.set_title(feature.replace("_", " "))
    return ax


def plot_top_features(df: pd.DataFrame, features: Sequence[str], class_col: str = "class", n_cols: int = 3):
    n = len(features)
    n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = np.atleast_1d(axes).ravel()
    for ax, feat in zip(axes, features):
        boxplot_feature_by_class(df, feat, class_col=class_col, ax=ax)
    for ax in axes[n:]:
        ax.set_visible(False)
    fig.tight_layout()
    return fig
