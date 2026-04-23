"""CASE dataset loader.

The public CASE dataset released by Sharma et al. (2019) is organised as::

    <root>/
        annotations/          # valence, arousal, video id at ~20 Hz
            sub_1.csv
            sub_2.csv
            ...
        physiological/        # 1000 Hz multi-sensor recordings (EDA = "gsr")
            sub_1.csv
            ...
        interpolated/
            annotations/      # annotations resampled to 1000 Hz (same time base as physiological)
                sub_1.csv
                ...
            physiological/    # same as physiological/ but sometimes re-sampled
                sub_1.csv
                ...

This module supports both ``interpolated`` and ``raw`` layouts; pass the
folder that contains the ``annotations/`` and ``physiological/``
sub-folders.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator, Tuple

import numpy as np
import pandas as pd

from .config import Config


def _pick_column(df: pd.DataFrame, candidates) -> str:
    """Return the first column matching one of ``candidates`` (case-insensitive)."""
    lower = {c.lower(): c for c in df.columns}
    for name in candidates:
        if name.lower() in lower:
            return lower[name.lower()]
    raise KeyError(f"None of {candidates} found in columns {list(df.columns)}")


def load_case_subject(root: str | Path, subject: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load the annotation and physiological DataFrames for one CASE subject.

    Returns
    -------
    annotations : DataFrame
        Columns: ``time`` (s), ``valence``, ``arousal``, ``video``.
    physiological : DataFrame
        Columns: ``time`` (s), ``eda``.
    """
    root = Path(root)
    ann_path = root / "annotations" / f"sub_{subject}.csv"
    phys_path = root / "physiological" / f"sub_{subject}.csv"
    if not ann_path.exists():
        raise FileNotFoundError(ann_path)
    if not phys_path.exists():
        raise FileNotFoundError(phys_path)

    ann = pd.read_csv(ann_path)
    phys = pd.read_csv(phys_path)

    # Standardise column names.
    time_a = _pick_column(ann, ["daqtime", "jstime", "time"])
    val_col = _pick_column(ann, ["valence"])
    aro_col = _pick_column(ann, ["arousal"])
    vid_col = _pick_column(ann, ["video", "videoid", "video_id"])
    ann = ann.rename(columns={time_a: "time", val_col: "valence", aro_col: "arousal", vid_col: "video"})
    ann["time"] = ann["time"].astype(float) / (1000.0 if ann["time"].max() > 1e4 else 1.0)

    time_p = _pick_column(phys, ["daqtime", "time"])
    eda_col = _pick_column(phys, ["gsr", "eda"])
    phys = phys.rename(columns={time_p: "time", eda_col: "eda"})
    phys["time"] = phys["time"].astype(float) / (1000.0 if phys["time"].max() > 1e4 else 1.0)

    return ann[["time", "valence", "arousal", "video"]], phys[["time", "eda"]]


def _resample_to_grid(series_time: np.ndarray, series_values: np.ndarray, target_time: np.ndarray, *, kind: str = "nearest") -> np.ndarray:
    """Resample a stream onto ``target_time`` using nearest or linear interp."""
    if kind == "nearest":
        # np.searchsorted is O(N log N); fast enough for our 1000 Hz / 20 min sessions.
        idx = np.clip(np.searchsorted(series_time, target_time), 0, series_time.size - 1)
        # choose nearest of left/right neighbour
        left = np.maximum(idx - 1, 0)
        pick_left = np.abs(series_time[left] - target_time) < np.abs(series_time[idx] - target_time)
        idx = np.where(pick_left, left, idx)
        return series_values[idx]
    return np.interp(target_time, series_time, series_values)


def iter_case_windows(
    root: str | Path,
    subject: int,
    cfg: Config,
) -> Iterator[Tuple[int, float, float, np.ndarray, np.ndarray, np.ndarray]]:
    """Yield ``(class, valence, arousal, eda_window, val_window, aro_window)``.

    ``class`` is the categorical label according to :attr:`Config.class_map`.
    Windows whose majority video-id is not in the class map are skipped.
    """
    ann, phys = load_case_subject(root, subject)

    # Decimate the EDA to ``cfg.fs``.
    from .preprocessing import preprocess_eda, segment_signal, label_window

    eda = preprocess_eda(phys["eda"].to_numpy(), cfg)
    # Time axis after decimation.
    t_grid = np.arange(eda.size, dtype=np.float64) / cfg.fs

    # Align annotations onto the decimated grid.
    val_grid = _resample_to_grid(ann["time"].to_numpy(), ann["valence"].to_numpy(), t_grid, kind="linear")
    aro_grid = _resample_to_grid(ann["time"].to_numpy(), ann["arousal"].to_numpy(), t_grid, kind="linear")
    vid_grid = _resample_to_grid(ann["time"].to_numpy(), ann["video"].to_numpy(), t_grid, kind="nearest")

    for start, end, window in segment_signal(eda, cfg):
        cls, v, a = label_window(
            vid_grid[start:end],
            val_grid[start:end],
            aro_grid[start:end],
            cfg.class_map,
            majority_ratio=cfg.majority_ratio,
        )
        if cls is None:
            continue
        yield cls, v, a, window, val_grid[start:end], aro_grid[start:end]
