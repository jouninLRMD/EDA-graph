#!/usr/bin/env python
"""Batch-extract the 58 EDA-Graph features from a CASE-formatted dataset.

Example
-------
    python scripts/extract_features.py \
        --data-root "E:/OneDrive/Research/emotion_graph/Data/data/interpolated" \
        --output EDA_graph_features.csv \
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
    ap.add_argument("--output", required=True, help="Path to write the EDA-graph features CSV.")
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

    print(f"[EDA-Graph] extracting from {args.data_root} ...")
    df = pipe.extract_graph_features_dataset(args.data_root, subjects=subjects)
    df.to_csv(args.output, index=False)
    print(f"[EDA-Graph] wrote {len(df)} windows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
