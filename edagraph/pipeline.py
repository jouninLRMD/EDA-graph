"""High-level EDA-graph feature-extraction pipeline."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from .config import Config
from .dataset import iter_case_windows
from .features import GRAPH_FEATURE_NAMES, extract_graph_features
from .graph import build_eda_graph


def _process_window(window: np.ndarray, cfg: Config) -> dict:
    g = build_eda_graph(window, cfg)
    return extract_graph_features(g)


def _subject_features(root: str, subject: int, cfg: Config) -> List[dict]:
    records: List[dict] = []
    for cls, v, a, window, _val, _aro in iter_case_windows(root, subject, cfg):
        row = {"subject": subject, "class": cls, "valence": v, "arousal": a}
        row.update(_process_window(window, cfg))
        records.append(row)
    return records


class EDAGraphPipeline:
    """End-to-end EDA-graph feature extractor.

    Example
    -------
    >>> pipe = EDAGraphPipeline()
    >>> df = pipe.extract_graph_features_dataset("CASE/interpolated")
    >>> df.to_csv("EDA_graph_features.csv", index=False)
    """

    def __init__(self, cfg: Optional[Config] = None, n_jobs: int = -1, verbose: int = 1):
        self.cfg = cfg or Config()
        self.n_jobs = n_jobs
        self.verbose = verbose

    def extract_graph_features_dataset(
        self, root: str | Path, subjects: Iterable[int] | None = None
    ) -> pd.DataFrame:
        root = Path(root)
        if subjects is None:
            subjects = sorted(
                int(p.stem.split("_")[-1])
                for p in (root / "physiological").glob("sub_*.csv")
            )

        results = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
            delayed(_subject_features)(str(root), s, self.cfg) for s in subjects
        )
        records = [r for sub in results for r in sub]
        if not records:
            raise RuntimeError(f"No windows extracted from {root}.")

        columns = ["subject"] + list(GRAPH_FEATURE_NAMES) + ["valence", "arousal", "class"]
        df = pd.DataFrame.from_records(records)
        return df[[c for c in columns if c in df.columns]]
