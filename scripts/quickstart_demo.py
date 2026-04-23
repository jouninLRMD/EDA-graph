#!/usr/bin/env python
"""Runnable demo that requires no external data.

Generates a synthetic EDA signal, builds an EDA-graph and extracts
every feature defined in the paper. Useful to verify that the
installation works end-to-end before pointing the pipeline at the CASE
dataset.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib.pyplot as plt

from edagraph import (
    Config,
    build_eda_graph,
    extract_graph_features,
    extract_traditional_features,
    preprocess_eda,
)
from edagraph.visualize import plot_eda_graph


def synthetic_eda(seed: int = 0, duration: float = 90.0, fs: int = 1000) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(int(duration * fs)) / fs
    scl = 5 + 0.3 * t / duration
    phasic = sum(np.exp(-((t - tau) ** 2) / 2) for tau in rng.uniform(4, duration - 4, size=6))
    return scl + 0.8 * phasic + 0.04 * rng.normal(size=t.size)


def main() -> int:
    cfg = Config()
    raw = synthetic_eda()
    processed = preprocess_eda(raw, cfg)
    window = processed[: cfg.window_samples]

    g = build_eda_graph(window, cfg)
    graph_feats = extract_graph_features(g)
    trad_feats = extract_traditional_features(
        raw[: int(cfg.window_sec * cfg.fs_raw)], cfg.fs_raw
    )

    print(f"Graph:     {g.number_of_nodes()} nodes, {g.number_of_edges()} edges")
    print(f"Graph features extracted: {len(graph_feats)}")
    print(f"Traditional features extracted: {len(trad_feats)}")
    print("\nTop-10 graph features (by magnitude):")
    for k, v in sorted(graph_feats.items(), key=lambda kv: -abs(kv[1]))[:10]:
        print(f"  {k:40s}  {v:.4f}")
    print("\nTraditional features:")
    for k, v in trad_feats.items():
        print(f"  {k:10s}  {v:.4f}")

    # Plot a summary figure.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    t = np.arange(window.size) / cfg.fs
    ax1.plot(t, window, "k")
    ax1.set_xlabel("time (s)")
    ax1.set_ylabel("EDA (normalised)")
    ax1.set_title(f"Input window ({cfg.window_sec:.0f} s)")
    plot_eda_graph(g, ax=ax2, title=f"EDA-graph (Q={cfg.q_step} uS, k={cfg.knn_k})")
    fig.tight_layout()
    out = Path("quickstart_demo.png")
    fig.savefig(out, dpi=150)
    print(f"\nSaved figure to {out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
