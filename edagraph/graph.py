"""Graph construction utilities.

The core building block is :func:`build_eda_graph` which takes a raw EDA
window and returns a :class:`networkx.Graph`.

The implementation uses :class:`scipy.spatial.cKDTree` for the :math:`k`-NN
search (``O(N log N)``) and builds the graph from numpy arrays, which is an
order of magnitude faster than iterating node by node in pure Python.
"""
from __future__ import annotations

from typing import Iterable

import networkx as nx
import numpy as np
from scipy.spatial import cKDTree

from .config import Config
from .quantization import graph_nodes_from_quantization, quantize_signal


def knn_graph_from_points(points: np.ndarray, k: int) -> nx.Graph:
    """Return an undirected :math:`k`-NN graph over a 2-D point cloud.

    Edge weights are ``1 / (1 + distance)`` so that closer nodes have
    higher weights - the form used by the paper.
    """
    n = points.shape[0]
    g = nx.Graph()
    g.add_nodes_from(range(n))
    if n < 2:
        return g

    kq = min(k + 1, n)  # +1 because the point itself is the first neighbour
    tree = cKDTree(points)
    dist, idx = tree.query(points, k=kq)

    # idx[:, 0] is always the point itself; drop it.
    src = np.repeat(np.arange(n), kq - 1)
    dst = idx[:, 1:].ravel()
    d = dist[:, 1:].ravel()

    # Deduplicate undirected edges by sorting endpoints.
    a = np.minimum(src, dst)
    b = np.maximum(src, dst)
    pair = np.unique(np.stack([a, b, d], axis=1), axis=0)
    weights = 1.0 / (1.0 + pair[:, 2])
    g.add_weighted_edges_from(
        ((int(u), int(v), float(w)) for u, v, w in zip(pair[:, 0], pair[:, 1], weights)),
        weight="weight",
    )
    return g


def build_eda_graph(window: np.ndarray, cfg: Config) -> nx.Graph:
    """Convert a single EDA window to an ``nx.Graph``.

    Nodes are unique quantised amplitude levels observed in the window.
    Edges connect each node to its :math:`k` nearest neighbours in
    ``(time, level)`` space, using Euclidean distance. Node attributes
    ``level`` and ``time`` and edge attribute ``weight`` are stored.
    """
    q = quantize_signal(window, cfg)
    levels, coords = graph_nodes_from_quantization(q, cfg)
    g = knn_graph_from_points(coords, cfg.knn_k)

    # Relabel nodes with their quantisation level and attach coordinates.
    mapping = {i: int(levels[i]) for i in range(levels.size)}
    g = nx.relabel_nodes(g, mapping, copy=True)
    for i, lvl in enumerate(levels):
        g.nodes[int(lvl)]["level"] = int(lvl)
        g.nodes[int(lvl)]["time"] = float(coords[i, 0])
    return g
