#!/usr/bin/env python
"""Reproduce Fig. 4 of the paper: select the optimal quantisation step Q.

The paper sweeps Q from 0.001 to 0.99 microsiemens and picks the value
that maximises the classification balanced accuracy / F1 on the CASE
dataset. This script does the same: for each Q it

1. extracts the 58 EDA-graph features from every window,
2. runs the Leave-One-Subject-Out classification with a single
   classifier (Random Forest by default, fastest for a sweep),
3. records the balanced accuracy and macro-F1.

Example
-------
    python scripts/q_sweep.py \\
        --data-root /path/to/CASE/interpolated \\
        --q-values 0.005,0.01,0.02,0.03,0.04,0.05,0.07,0.1,0.2 \\
        --output results/q_sweep.csv

The resulting ``q_sweep.csv`` and accompanying PNG plot mirror Fig. 4
of the paper.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from edagraph import Config, EDAGraphPipeline
from edagraph.experiments import run_loso_classification


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", required=True, help="Folder with annotations/ and physiological/.")
    ap.add_argument(
        "--q-values",
        default="0.005,0.01,0.02,0.03,0.04,0.05,0.07,0.1,0.2,0.5",
        help="Comma-separated list of Q step sizes (microsiemens).",
    )
    ap.add_argument("--classifier", default="rf", help="Classifier key from experiments.py.")
    ap.add_argument("--k-best", type=int, default=None)
    ap.add_argument("--output", default="results/q_sweep.csv")
    ap.add_argument("--plot", default="results/q_sweep.png")
    ap.add_argument("--subjects", default=None, help="Comma-separated subject ids (default: all).")
    ap.add_argument("--n-jobs", type=int, default=-1)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    q_values = [float(q) for q in args.q_values.split(",")]
    subjects = [int(s) for s in args.subjects.split(",")] if args.subjects else None

    rows = []
    for q in q_values:
        cfg = Config(q_step=q)
        print(f"\n=== Q = {q} uS ===")
        pipe = EDAGraphPipeline(cfg=cfg, n_jobs=args.n_jobs, verbose=0)
        df = pipe.extract_graph_features_dataset(args.data_root, subjects=subjects)
        feat_cols = [c for c in df.columns if c not in {"subject", "class", "valence", "arousal"}]

        summary, _ = run_loso_classification(
            df,
            feature_cols=feat_cols,
            k_best=args.k_best,
            n_jobs=args.n_jobs,
            classifiers={args.classifier: __import__("edagraph.experiments",
                                                    fromlist=["_default_classifiers"])._default_classifiers()[args.classifier]},
            verbose=0,
        )
        r = summary.iloc[0]
        rows.append(
            dict(
                q=q,
                n_windows=len(df),
                accuracy=r.accuracy,
                balanced_accuracy=r.balanced_accuracy,
                macro_f1=r.macro_f1,
                weighted_f1=r.weighted_f1,
            )
        )
        print(f"  windows={len(df):6d}  balanced_acc={r.balanced_accuracy:.3f}  macro_f1={r.macro_f1:.3f}")

    out = pd.DataFrame(rows).sort_values("q")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)

    best = out.iloc[out.balanced_accuracy.idxmax()]
    print(f"\nOptimal Q (balanced accuracy): Q = {best.q} uS  bal_acc = {best.balanced_accuracy:.3f}")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(out.q, out.balanced_accuracy, "o-", label="balanced accuracy")
    ax.plot(out.q, out.macro_f1, "s--", label="macro F1")
    ax.axvline(best.q, color="gray", lw=0.5, ls=":")
    ax.set_xscale("log")
    ax.set_xlabel("Quantisation step Q  (uS)")
    ax.set_ylabel("LOSO score")
    ax.set_title(f"Selection of optimal Q (best = {best.q} uS)")
    ax.legend()
    ax.grid(True, which="both", ls=":", alpha=0.5)
    fig.tight_layout()
    Path(args.plot).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.plot, dpi=150)
    print(f"Saved {args.output} and {args.plot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
