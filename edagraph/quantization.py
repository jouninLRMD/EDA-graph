"""Amplitude quantisation and node construction.

Follows Section II-C of the paper.

Step 1 - Quantisation
---------------------
Each sample is mapped to the closest integer multiple of the quantisation
step :math:`Q` (in microsiemens):

.. math:: x_{\\text{quantized}} = Q \\cdot \\operatorname{round}\\left(\\frac{x_{\\text{original}}}{Q}\\right).

The paper reports :math:`Q = 0.05\\, \\mu S` as optimal (Section II-C).

Step 2 - Node definition
------------------------
Nodes are the *unique* values in ``x_quantized`` - only the first
occurrence of each new value is kept. This is the discrete-state
abstraction of the EDA-graph:

.. math:: X_{\\text{nodes}} = x_{\\text{quantized}}\\bigl[x_{\\text{quantized}} \\neq x_{\\text{quantized,shifted}}\\bigr]

where ``x_{quantized,shifted}`` is the quantised signal shifted by one
sample.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np

from .config import Config


def quantize_signal(window: np.ndarray, cfg: Config) -> np.ndarray:
    """Return the quantised signal ``Q * round(x / Q)``.

    The output preserves the original amplitude units (microsiemens).
    """
    window = np.asarray(window, dtype=np.float64).ravel()
    if window.size == 0:
        return window
    q = float(cfg.q_step)
    if q <= 0:
        raise ValueError("q_step must be positive")
    return q * np.round(window / q)


def node_values_from_quantization(quantized: np.ndarray) -> np.ndarray:
    """Extract the node values from a quantised signal.

    Implements equation (2) of the paper: keep only samples where the
    quantised value changes from the previous one. Then deduplicate
    because the same value may be revisited later in the window.
    """
    if quantized.size == 0:
        return quantized
    # First-difference mask: keep indices where the value changes.
    keep = np.empty(quantized.size, dtype=bool)
    keep[0] = True
    np.not_equal(quantized[1:], quantized[:-1], out=keep[1:])
    first_occurrences = quantized[keep]
    # ``np.unique`` sorts and deduplicates - final node set of equation (2).
    return np.unique(first_occurrences)


def graph_nodes_from_quantization(
    quantized: np.ndarray, cfg: Config
) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(node_values, node_values_column)`` for a quantised window.

    ``node_values_column`` is the same as ``node_values`` reshaped to
    ``(N, 1)`` - i.e. the one-dimensional feature vector that feeds the
    Euclidean distance step (equation 3 of the paper, with M = 1).
    """
    node_values = node_values_from_quantization(quantized)
    return node_values, node_values.reshape(-1, 1)
