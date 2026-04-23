"""High-level end-to-end feature extraction pipeline."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from .config import Config
from .dataset import iter_case_windows
from .features import (
    GRAPH_FEATURE_NAMES,
    TRADITIONAL_FEATURE_NAMES,
    extract_graph_features,
    extract_traditional_features,
)
from .graph import build_eda_graph


def _process_window(window: np.ndarray, cfg: Config, *, compute_graph: bool, compute_trad: bool):
    row = {}
    if compute_graph:
        g = build_eda_graph(window, cfg)
        row.update(extract_graph_features(g))
    if compute_trad:
        row.update(extract_traditional_features(window, cfg.fs))
    return row


def _subject_features(
    root: str,
    subject: int,
    cfg: Config,
    compute_graph: bool,
    compute_trad: bool,
) -> List[dict]:
    records: List[dict] = []
    for cls, v, a, window, _val, _aro in iter_case_windows(root, subject, cfg):
        row = {"subject": subject, "class": cls, "valence": v, "arousal": a}
        row.update(_process_window(window, cfg, compute_graph=compute_graph, compute_trad=compute_trad))
        records.append(row)
    return records


class EDAGraphPipeline:
    """End-to-end feature extractor.

    Example
    -------
    >>> pipe = EDAGraphPipeline()
    >>> df_graph = pipe.extract_graph_features_dataset("CASE/interpolated")
    >>> df_trad = pipe.extract_traditional_features_dataset("CASE/interpolated")
    """

    def __init__(self, cfg: Optional[Config] = None, n_jobs: int = -1, verbose: int = 1):
        self.cfg = cfg or Config()
        self.n_jobs = n_jobs
        self.verbose = verbose

    # -----------------------------------------------------------------
    def _collect(
        self,
        root: str | Path,
        subjects: Iterable[int] | None,
        compute_graph: bool,
        compute_trad: bool,
    ) -> pd.DataFrame:
        root = Path(root)
        if subjects is None:
            subjects = sorted(
                int(p.stem.split("_")[-1])
                for p in (root / "physiological").glob("sub_*.csv")
            )

        results = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
            delayed(_subject_features)(str(root), s, self.cfg, compute_graph, compute_trad)
            for s in subjects
        )
        records = [r for sub in results for r in sub]
        if not records:
            raise RuntimeError(f"No windows extracted from {root}.")

        # Enforce canonical column order.
        columns = ["subject"]
        if compute_graph:
            columns += list(GRAPH_FEATURE_NAMES)
        if compute_trad:
            columns += list(TRADITIONAL_FEATURE_NAMES)
        columns += ["valence", "arousal", "class"]
        df = pd.DataFrame.from_records(records)
        return df[[c for c in columns if c in df.columns]]

    def extract_graph_features_dataset(self, root, subjects=None) -> pd.DataFrame:
        return self._collect(root, subjects, compute_graph=True, compute_trad=False)

    def extract_traditional_features_dataset(self, root, subjects=None) -> pd.DataFrame:
        return self._collect(root, subjects, compute_graph=False, compute_trad=True)

    def extract_all(self, root, subjects=None) -> pd.DataFrame:
        return self._collect(root, subjects, compute_graph=True, compute_trad=True)
