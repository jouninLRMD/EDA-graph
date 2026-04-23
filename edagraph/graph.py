"""Graph construction utilities.

Implements Steps 3-5 of Section II-C of the paper:

* Step 3 - Distance calculation: the pair-wise Euclidean distance between
  node values (``M = 1`` in the paper, so it reduces to
  :math:`D_{ij} = |x_i - x_j|`).
* Step 4 - Nearest neighbours: each node is connected to its 8 closest
  neighbours by ``D_{ij}``.
* Step 5 - Adjacency matrix: edge weights are :math:`w_{ij} = 1/D_{ij}`;
  ``A`` is the symmetric weighted adjacency.

Uses :class:`scipy.spatial.cKDTree` for the :math:`k`-NN search, so the
whole step is ``O(N log N)``.
"""
from __future__ import annotations

import networkx as nx
import numpy as np
from scipy.spatial import cKDTree

from .config import Config
from .quantization import graph_nodes_from_quantization, quantize_signal


def knn_graph_from_points(points: np.ndarray, k: int) -> nx.Graph:
    """Build an undirected :math:`k`-NN graph.

    Parameters
    ----------
    points : (N, M) array
        Node feature vectors. ``M = 1`` for the EDA-graph case.
    k : int
        Number of neighbours per node.

    Notes
    -----
    Edge weight follows equation (5) of the paper: ``w_st = 1 / D_st``.
    A small epsilon is added to the denominator to avoid division by zero
    when two nodes sit on top of each other (which should not happen for
    the EDA-graph, since the node-definition step removes duplicate
    values, but keeps the function robust for generic inputs).
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

    # Deduplicate undirected edges.
    a = np.minimum(src, dst)
    b = np.maximum(src, dst)
    pair = np.unique(np.stack([a, b, d], axis=1), axis=0)
    weights = 1.0 / np.maximum(pair[:, 2], 1e-12)
    g.add_weighted_edges_from(
        ((int(u), int(v), float(w)) for u, v, w in zip(pair[:, 0], pair[:, 1], weights)),
        weight="weight",
    )
    return g


def build_eda_graph(window: np.ndarray, cfg: Config) -> nx.Graph:
    """Convert a single EDA window to an ``nx.Graph``.

    Nodes are the unique quantised amplitude values in the window.
    Edges connect each node to its :math:`k = \\text{cfg.knn\\_k}`
    nearest neighbours using the Euclidean distance
    :math:`D_{ij} = |x_i - x_j|`.

    The returned graph uses the original quantised value as node id, and
    stores it both as an attribute and as the node label for convenience.
    """
    q = quantize_signal(window, cfg)
    values, points = graph_nodes_from_quantization(q, cfg)
    g = knn_graph_from_points(points, cfg.knn_k)

    # Relabel integer indices with the actual quantised values.
    mapping = {i: float(values[i]) for i in range(values.size)}
    g = nx.relabel_nodes(g, mapping, copy=True)
    for v in values:
        g.nodes[float(v)]["value"] = float(v)
    return g
