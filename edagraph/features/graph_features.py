"""Graph-level, node-level and spectral features for an EDA-graph.

The output of :func:`extract_graph_features` is a ``dict`` whose keys match
1:1 the column order of the original ``EDA_graph_features.csv`` published
with the paper. This allows dropping the new extractor in place of the
legacy code without breaking downstream scripts.

Optimisation notes
------------------
* All centralities are computed on the same graph in a single pass.
* The adjacency matrix and its eigen-decomposition are reused for both
  ``graph_spectrum_*`` and ``graph_energy``.
* The Laplacian spectrum is computed once and reused for every
  ``laplacian_spectrum_*`` feature.
* For disconnected graphs the slow ``center / diameter / radius / etc.``
  quantities are evaluated on the largest connected component only (the
  ``P_`` prefix in the column names stands for "principal component"),
  matching the behaviour of the original code but running in a fraction
  of the time because we reuse the eccentricity dictionary.
"""
from __future__ import annotations

from typing import Dict, List

import networkx as nx
import numpy as np
from scipy.stats import kurtosis, skew, spearmanr


# ---------------------------------------------------------------------------
# Canonical column order - must match EDA_graph_features.csv
# ---------------------------------------------------------------------------
GRAPH_FEATURE_NAMES: List[str] = [
    "total_triangle_number",
    "total_degree_centrality",
    "total_closeness_centrality",
    "total_betweenness_centrality",
    "total_flow_centrality",
    "total_log_flow_centrality",
    "total_eigenvector_centrality",
    "total_load_centrality",
    "total_harmonic_centrality",
    "total_pagerank",
    "total_hubs",
    "spearmanr_correlation",
    "number_of_nodes",
    "number_of_edges",
    "maximum_degree",
    "minimum_degree",
    "median_degree",
    "transitivity",
    "avg_clustering_coefficient",
    "assortativity",
    "graph_clique_num",
    "graph_number_of_cliqs",
    "P_is_chordal",
    "P_center",
    "P_diameter",
    "P_periphery",
    "P_radius",
    "P_average_clustering",
    "P_closeness_centrality",
    "eccentricity",
    "analyze_weisfeiler_lehman_kernel",
    "graph_energy",
    "graph_spectrum_std",
    "laplacian_spectrum_std",
    "avg_triangle_participation",
    "graph_spectrum_mean_magnitude",
    "graph_spectrum_min_magnitude",
    "graph_spectrum_max_magnitude",
    "graph_spectrum_median_magnitude",
    "graph_spectrum_skewness_magnitude",
    "graph_spectrum_kurtosis_magnitude",
    "laplacian_spectrum_mean_magnitude",
    "laplacian_spectrum_min_magnitude",
    "laplacian_spectrum_max_magnitude",
    "laplacian_spectrum_median_magnitude",
    "laplacian_spectrum_skewness_magnitude",
    "laplacian_spectrum_kurtosis_magnitude",
    "graph_spectrum_mean_phase",
    "graph_spectrum_min_phase",
    "graph_spectrum_max_phase",
    "graph_spectrum_median_phase",
    "graph_spectrum_skewness_phase",
    "graph_spectrum_kurtosis_phase",
    "laplacian_spectrum_mean_phase",
    "laplacian_spectrum_min_phase",
    "laplacian_spectrum_max_phase",
    "laplacian_spectrum_median_phase",
    "laplacian_spectrum_skewness_phase",
    "laplacian_spectrum_kurtosis_phase",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe(value, default=0.0):
    try:
        if value is None or (isinstance(value, float) and (np.isnan(value) or np.isinf(value))):
            return default
        return float(value)
    except Exception:
        return default


def _spectrum_stats(values: np.ndarray, prefix: str, out: Dict[str, float]) -> None:
    if values.size == 0:
        for k in ("mean", "min", "max", "median", "skewness", "kurtosis"):
            out[f"{prefix}_{k}_magnitude"] = 0.0
        return
    mag = np.abs(values)
    out[f"{prefix}_mean_magnitude"] = float(np.mean(mag))
    out[f"{prefix}_min_magnitude"] = float(np.min(mag))
    out[f"{prefix}_max_magnitude"] = float(np.max(mag))
    out[f"{prefix}_median_magnitude"] = float(np.median(mag))
    out[f"{prefix}_skewness_magnitude"] = _safe(skew(mag))
    out[f"{prefix}_kurtosis_magnitude"] = _safe(kurtosis(mag))


def _phase_stats(values: np.ndarray, prefix: str, out: Dict[str, float]) -> None:
    if values.size == 0:
        for k in ("mean", "min", "max", "median", "skewness", "kurtosis"):
            out[f"{prefix}_{k}_phase"] = 0.0
        return
    phase = np.angle(values.astype(np.complex128))
    out[f"{prefix}_mean_phase"] = float(np.mean(phase))
    out[f"{prefix}_min_phase"] = float(np.min(phase))
    out[f"{prefix}_max_phase"] = float(np.max(phase))
    out[f"{prefix}_median_phase"] = float(np.median(phase))
    out[f"{prefix}_skewness_phase"] = _safe(skew(phase))
    out[f"{prefix}_kurtosis_phase"] = _safe(kurtosis(phase))


# ---------------------------------------------------------------------------
# Main extractor
# ---------------------------------------------------------------------------

def extract_graph_features(g: nx.Graph) -> Dict[str, float]:
    """Compute all 59 graph features for a single EDA-graph.

    The returned dictionary contains every key in
    :data:`GRAPH_FEATURE_NAMES`, always in the same order.
    """
    out: Dict[str, float] = {k: 0.0 for k in GRAPH_FEATURE_NAMES}
    n = g.number_of_nodes()
    m = g.number_of_edges()
    out["number_of_nodes"] = float(n)
    out["number_of_edges"] = float(m)
    if n == 0:
        return out

    # ----- degree stats -------------------------------------------------
    degrees = np.array([d for _, d in g.degree()], dtype=np.float64)
    out["maximum_degree"] = float(degrees.max())
    out["minimum_degree"] = float(degrees.min())
    out["median_degree"] = float(np.median(degrees))

    # ----- centralities -------------------------------------------------
    deg_c = nx.degree_centrality(g)
    out["total_degree_centrality"] = float(sum(deg_c.values()))
    out["total_closeness_centrality"] = float(sum(nx.closeness_centrality(g).values()))
    out["total_betweenness_centrality"] = float(sum(nx.betweenness_centrality(g, normalized=True).values()))
    out["total_load_centrality"] = float(sum(nx.load_centrality(g).values()))
    out["total_harmonic_centrality"] = float(sum(nx.harmonic_centrality(g).values()))

    # Flow centrality uses a Laplacian pseudo-inverse; skip it for tiny or
    # disconnected graphs to avoid spurious warnings.
    flow_total = 0.0
    if nx.is_connected(g) and n > 2:
        try:
            flow_total = float(sum(nx.current_flow_betweenness_centrality(g, weight="weight").values()))
        except Exception:
            flow_total = 0.0
    out["total_flow_centrality"] = flow_total
    out["total_log_flow_centrality"] = float(np.log1p(flow_total))

    try:
        eig_c = nx.eigenvector_centrality_numpy(g, weight="weight")
        out["total_eigenvector_centrality"] = float(sum(eig_c.values()))
    except Exception:
        out["total_eigenvector_centrality"] = 0.0

    try:
        pr = nx.pagerank(g, weight="weight")
        out["total_pagerank"] = float(sum(pr.values()))
    except Exception:
        out["total_pagerank"] = 0.0

    try:
        hubs, _ = nx.hits(g, max_iter=200, normalized=True)
        out["total_hubs"] = float(sum(hubs.values()))
    except Exception:
        out["total_hubs"] = 0.0

    # ----- correlation between node index and degree (Spearman) ---------
    if n > 1:
        nodes = list(g.nodes())
        rho, _ = spearmanr(nodes, [g.degree(v) for v in nodes])
        out["spearmanr_correlation"] = _safe(rho)

    # ----- triangle / clustering statistics -----------------------------
    triangles = nx.triangles(g)
    tri_values = np.array(list(triangles.values()), dtype=np.float64)
    out["total_triangle_number"] = float(tri_values.sum() / 3.0)
    out["avg_triangle_participation"] = float(tri_values.mean())
    out["transitivity"] = _safe(nx.transitivity(g))
    out["avg_clustering_coefficient"] = _safe(nx.average_clustering(g))
    if m >= 1:
        out["assortativity"] = _safe(nx.degree_assortativity_coefficient(g))

    # ----- clique statistics -------------------------------------------
    try:
        out["graph_clique_num"] = float(nx.graph_clique_number(g))
        out["graph_number_of_cliqs"] = float(nx.graph_number_of_cliques(g))
    except Exception:
        out["graph_clique_num"] = 0.0
        out["graph_number_of_cliqs"] = 0.0

    # ----- largest connected component metrics --------------------------
    if n > 0:
        cc = max(nx.connected_components(g), key=len)
        sub = g.subgraph(cc).copy()
        if sub.number_of_nodes() > 1:
            ecc = nx.eccentricity(sub)
            ecc_values = np.array(list(ecc.values()), dtype=np.float64)
            diameter = float(ecc_values.max())
            radius = float(ecc_values.min())
            center = [v for v, e in ecc.items() if e == radius]
            periphery = [v for v, e in ecc.items() if e == diameter]
            out["eccentricity"] = float(ecc_values.mean())
            out["P_diameter"] = diameter
            out["P_radius"] = radius
            out["P_center"] = float(len(center))
            out["P_periphery"] = float(len(periphery))
            out["P_average_clustering"] = _safe(nx.average_clustering(sub))
            out["P_closeness_centrality"] = float(np.mean(list(nx.closeness_centrality(sub).values())))
            try:
                out["P_is_chordal"] = float(int(nx.is_chordal(sub)))
            except Exception:
                out["P_is_chordal"] = 0.0

    # ----- Weisfeiler-Lehman hash (stable scalar feature) ---------------
    try:
        wl_hash = nx.weisfeiler_lehman_graph_hash(g, iterations=3)
        # Convert the hash hex string to a bounded float in [0, 1].
        out["analyze_weisfeiler_lehman_kernel"] = int(wl_hash, 16) / float(16 ** len(wl_hash))
    except Exception:
        out["analyze_weisfeiler_lehman_kernel"] = 0.0

    # ----- Spectra ------------------------------------------------------
    if n > 0:
        # Adjacency spectrum (can be complex for directed; ours is symmetric real).
        A = nx.to_numpy_array(g, weight="weight")
        eig_A = np.linalg.eigvals(A)
        out["graph_energy"] = float(np.sum(np.abs(eig_A)))
        out["graph_spectrum_std"] = float(np.std(np.abs(eig_A)))
        _spectrum_stats(eig_A, "graph_spectrum", out)
        _phase_stats(eig_A, "graph_spectrum", out)

        # Laplacian spectrum.
        L = nx.laplacian_matrix(g, weight="weight").toarray().astype(np.float64)
        eig_L = np.linalg.eigvalsh(L)  # symmetric -> real eigenvalues
        out["laplacian_spectrum_std"] = float(np.std(eig_L))
        _spectrum_stats(eig_L, "laplacian_spectrum", out)
        _phase_stats(eig_L.astype(np.complex128), "laplacian_spectrum", out)

    return out
