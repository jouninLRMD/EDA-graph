#!/usr/bin/env python
"""Batch-extract EDA-Graph and/or traditional features from the CASE dataset.

Example
-------
    python scripts/extract_features.py \
        --data-root "E:/OneDrive/Research/emotion_graph/Data/data/interpolated" \
        --output-graph EDA_graph_features.csv \
        --output-traditional EDA_Traditional_Features.csv \
        --n-jobs -1
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from edagraph import Config, EDAGraphPipeline


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", required=True, help="Folder with annotations/ and physiological/ sub-dirs.")
    ap.add_argument("--output-graph", default=None, help="Path to write EDA-graph features CSV.")
    ap.add_argument("--output-traditional", default=None, help="Path to write traditional features CSV.")
    ap.add_argument("--config", default=None, help="Optional YAML config file.")
    ap.add_argument("--subjects", default=None, help="Comma-separated subject ids (default: all).")
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--verbose", type=int, default=5)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    cfg = Config.from_yaml(args.config) if args.config else Config()
    pipe = EDAGraphPipeline(cfg=cfg, n_jobs=args.n_jobs, verbose=args.verbose)

    subjects = [int(s) for s in args.subjects.split(",")] if args.subjects else None

    if not args.output_graph and not args.output_traditional:
        print("Nothing to do: pass --output-graph and/or --output-traditional", file=sys.stderr)
        return 2

    if args.output_graph:
        print(f"[EDA-Graph] extracting graph features from {args.data_root} ...")
        df_graph = pipe.extract_graph_features_dataset(args.data_root, subjects=subjects)
        df_graph.to_csv(args.output_graph, index=False)
        print(f"[EDA-Graph] wrote {len(df_graph)} windows to {args.output_graph}")

    if args.output_traditional:
        print(f"[EDA-Graph] extracting traditional features from {args.data_root} ...")
        df_trad = pipe.extract_traditional_features_dataset(args.data_root, subjects=subjects)
        df_trad.to_csv(args.output_traditional, index=False)
        print(f"[EDA-Graph] wrote {len(df_trad)} windows to {args.output_traditional}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
