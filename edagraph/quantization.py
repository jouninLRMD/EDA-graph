"""Amplitude quantisation and node construction.

The paper turns an EDA segment into a graph by:

1. Normalising the signal (min-max by default).
2. Quantising the amplitude into ``Q`` discrete levels, i.e. mapping
   :math:`x_t \\to q_t = \\lfloor x_t\\, Q \\rfloor \\in \\{0,\\dots,Q-1\\}`.
3. Turning the quantised stream into a sequence of 2-D points
   :math:`(t, q_t)` that can be linked via :math:`k`-NN.

Only unique levels visited in the window become graph nodes. Each node's
coordinate is ``(mean_time_of_occurrence, level_value)`` both normalised to
``[0, 1]`` so that the Euclidean distance is dimensionless.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np

from .config import Config


def _normalize(x: np.ndarray, method: str) -> np.ndarray:
    if method == "minmax":
        lo, hi = float(np.min(x)), float(np.max(x))
        if hi - lo < 1e-12:
            return np.zeros_like(x)
        return (x - lo) / (hi - lo)
    if method == "zscore":
        mu, sd = float(np.mean(x)), float(np.std(x))
        if sd < 1e-12:
            return np.zeros_like(x)
        z = (x - mu) / sd
        # map to [0, 1] with a logistic squash for quantisation
        return 1.0 / (1.0 + np.exp(-z))
    raise ValueError(f"Unknown normalisation '{method}'.")


def quantize_signal(window: np.ndarray, cfg: Config) -> np.ndarray:
    """Return an integer array of the same length with values in ``[0, Q)``."""
    window = np.asarray(window, dtype=np.float64).ravel()
    if window.size == 0:
        return np.empty(0, dtype=np.int64)
    norm = _normalize(window, cfg.normalize)
    q = np.floor(norm * cfg.q_levels).astype(np.int64)
    np.clip(q, 0, cfg.q_levels - 1, out=q)
    return q


def graph_nodes_from_quantization(
    quantized: np.ndarray, cfg: Config
) -> Tuple[np.ndarray, np.ndarray]:
    """Build node coordinates from a quantised EDA window.

    Parameters
    ----------
    quantized : (N,) int array
        Output of :func:`quantize_signal`.
    cfg : Config

    Returns
    -------
    levels : (M,) array
        Unique quantisation levels observed in the window, sorted ascending.
        These are the node identifiers.
    coords : (M, 2) array
        2-D coordinates for each node: ``(mean_time / N, level / Q)``.
        Both axes are normalised to ``[0, 1]`` so the Euclidean distance
        used for :math:`k`-NN is isotropic.
    """
    if quantized.size == 0:
        return np.empty(0, dtype=np.int64), np.empty((0, 2), dtype=np.float64)

    n = quantized.size
    # Using np.unique with return_inverse avoids a Python loop over levels.
    levels, inverse = np.unique(quantized, return_inverse=True)
    # Accumulate sums of time indices per unique level with bincount.
    times = np.arange(n, dtype=np.float64)
    sum_t = np.bincount(inverse, weights=times, minlength=levels.size)
    counts = np.bincount(inverse, minlength=levels.size)
    mean_t = sum_t / np.maximum(counts, 1)

    coords = np.column_stack([mean_t / max(n - 1, 1), levels / max(cfg.q_levels - 1, 1)])
    return levels, coords
