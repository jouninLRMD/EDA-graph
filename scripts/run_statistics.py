#!/usr/bin/env python
"""Run the Anderson-Darling / Kruskal-Wallis / Dunn + Holm-Bonferroni
statistical pipeline (Section II-E of the paper).

Example
-------
    python scripts/run_statistics.py ^
        --features EDA_graph_features.csv ^
        --output-dir results/stats ^
        --top-k 5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from edagraph.stats import anderson_darling_by_class, dunn_posthoc, kruskal_wallis


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features", required=True, help="CSV produced by extract_features.py")
    ap.add_argument("--output-dir", default="results/stats")
    ap.add_argument("--top-k", type=int, default=5, help="Top-K features by Kruskal-Wallis H.")
    ap.add_argument("--class-col", default="class")
    ap.add_argument("--alpha", type=float, default=0.005, help="Significance threshold (paper: 0.005).")
    ap.add_argument("--apply-fdr", action="store_true", help="Also run BH-FDR on top of Holm.")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    df = pd.read_csv(args.features)
    feat_cols = [c for c in df.columns if c not in {args.class_col, "subject", "valence", "arousal"}]

    kw = kruskal_wallis(df, feat_cols, class_col=args.class_col).sort_values("p_value")
    top = kw.head(args.top_k)["feature"].tolist()

    ad = anderson_darling_by_class(df, top, class_col=args.class_col)
    dunn = dunn_posthoc(
        df, top, class_col=args.class_col, alpha=args.alpha, apply_fdr=args.apply_fdr
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    kw.to_csv(out_dir / "kruskal_wallis.csv", index=False)
    ad.to_csv(out_dir / "anderson_darling.csv", index=False)
    for feat, table in dunn.items():
        table.to_csv(out_dir / f"dunn_{feat}.csv")

    print("\n=== Kruskal-Wallis (top 15) ===")
    print(kw.head(15).to_string(index=False))
    print("\n=== Anderson-Darling on top features ===")
    print(ad.to_string(index=False))
    print(f"\nWrote all tables to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
