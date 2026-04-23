"""Smoke test - runs on a synthetic EDA signal, no external data required."""
from __future__ import annotations

import numpy as np

from edagraph import (
    Config,
    build_eda_graph,
    extract_graph_features,
    preprocess_eda,
    quantize_signal,
)
from edagraph.features import GRAPH_FEATURE_NAMES


def _synthetic_eda(duration: float = 90.0, fs: int = 1000, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(int(duration * fs)) / fs
    scl = 5 + 0.2 * t + 0.1 * np.sin(2 * np.pi * 0.01 * t)
    bumps = np.zeros_like(t)
    for t0 in rng.uniform(5, duration - 5, size=8):
        bumps += np.exp(-((t - t0) ** 2) / (2 * 1.5 ** 2))
    return scl + bumps + 0.02 * rng.normal(size=t.size)


def test_preprocess_and_quantize():
    cfg = Config()
    sig = _synthetic_eda(90.0)
    filt = preprocess_eda(sig, cfg)
    assert filt.size == int(90 * cfg.fs)
    q = quantize_signal(filt[: cfg.window_samples], cfg)
    # quantised values must be multiples of the step size
    ratios = np.round(q / cfg.q_step)
    assert np.allclose(q, ratios * cfg.q_step)


def test_graph_features_schema():
    cfg = Config()
    sig = _synthetic_eda(90.0)
    window = preprocess_eda(sig, cfg)[: cfg.window_samples]
    g = build_eda_graph(window, cfg)
    feats = extract_graph_features(g)
    assert set(feats.keys()) == set(GRAPH_FEATURE_NAMES)
    assert all(np.isfinite(list(feats.values())))
