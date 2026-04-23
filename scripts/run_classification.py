#!/usr/bin/env python
"""Run the Leave-One-Subject-Out classification experiment on a features CSV.

Usage
-----
    python scripts/run_classification.py \
        --features EDA_graph_features.csv \
        --k-best 5 \
        --output results/graph_classification.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from edagraph.experiments import run_loso_classification


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features", required=True, help="CSV produced by extract_features.py")
    ap.add_argument("--output", default=None, help="Path to write the summary CSV.")
    ap.add_argument("--k-best", type=int, default=None, help="Select K best features via ANOVA F-score.")
    ap.add_argument("--class-col", default="class")
    ap.add_argument("--group-col", default="subject")
    ap.add_argument("--n-jobs", type=int, default=-1)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    df = pd.read_csv(args.features)
    drop_cols = {args.class_col, args.group_col, "valence", "arousal"}
    feat_cols = [c for c in df.columns if c not in drop_cols]

    summary, reports = run_loso_classification(
        df,
        feature_cols=feat_cols,
        class_col=args.class_col,
        group_col=args.group_col,
        k_best=args.k_best,
        n_jobs=args.n_jobs,
    )

    print("\n=== Summary ===")
    print(summary.to_string(index=False))
    if "selected_features" in summary.attrs:
        print("\nSelected features:", summary.attrs["selected_features"])

    for name, res in reports.items():
        print(f"\n--- {name} ---")
        print(res.report)

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(args.output, index=False)
        print(f"\nWrote summary to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
