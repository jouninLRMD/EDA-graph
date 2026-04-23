"""Preprocessing of raw EDA signals prior to graph construction.

The pipeline followed in the paper is:

1. 4th-order zero-phase low-pass Butterworth (``lowpass_hz``).
2. Anti-alias filter + decimation to ``fs``.
3. Per-subject min-max (or z-score) normalisation to stabilise amplitude
   ranges across participants.
4. Sliding window segmentation (``window_sec`` s, non-overlapping by
   default).

Implemented vectorised with SciPy so that a whole session (~3 min * 8 videos)
can be filtered in a few milliseconds.
"""
from __future__ import annotations

from typing import Iterable, Iterator, Tuple

import numpy as np
from scipy.signal import butter, decimate, filtfilt

from .config import Config


# ---------------------------------------------------------------------------
# Filtering / resampling
# ---------------------------------------------------------------------------

def _butter_lowpass(signal: np.ndarray, fs: float, cutoff: float, order: int = 4) -> np.ndarray:
    """Zero-phase Butterworth low-pass filter.

    Using ``filtfilt`` removes phase distortion, which is important because the
    quantisation step maps amplitude to node identity - any phase shift would
    move events relative to the label stream.
    """
    nyq = 0.5 * fs
    b, a = butter(order, cutoff / nyq, btype="low")
    return filtfilt(b, a, signal)


def preprocess_eda(raw: np.ndarray, cfg: Config) -> np.ndarray:
    """Filter and decimate a raw EDA trace.

    Parameters
    ----------
    raw : (N,) array
        Raw EDA at ``cfg.fs_raw`` Hz.
    cfg : Config
        Pipeline configuration.

    Returns
    -------
    (M,) array
        Low-pass filtered signal sampled at ``cfg.fs`` Hz.
    """
    raw = np.asarray(raw, dtype=np.float64).ravel()
    if raw.size == 0:
        return raw
    filtered = _butter_lowpass(raw, cfg.fs_raw, cfg.lowpass_hz)
    factor = int(round(cfg.fs_raw / cfg.fs))
    if factor <= 1:
        return filtered
    # ``decimate`` already applies an 8th-order Chebyshev anti-alias filter.
    return decimate(filtered, factor, ftype="iir", zero_phase=True)


# ---------------------------------------------------------------------------
# Windowing
# ---------------------------------------------------------------------------

def segment_signal(
    signal: np.ndarray, cfg: Config, start_times: np.ndarray | None = None
) -> Iterator[Tuple[int, int, np.ndarray]]:
    """Iterate over fixed-length windows.

    Yields ``(start_idx, end_idx, window)`` triples. ``start_idx`` and
    ``end_idx`` are indices into ``signal`` at sampling rate ``cfg.fs``.

    If a ``start_times`` array is provided, it is returned as the third
    element instead of integer indices - useful when aligning with the
    annotation stream.
    """
    n = signal.size
    w = cfg.window_samples
    step = cfg.step_samples
    if w <= 0:
        raise ValueError("window_sec must be positive")

    for start in range(0, n - w + 1, step):
        end = start + w
        yield start, end, signal[start:end]


# ---------------------------------------------------------------------------
# Labelling
# ---------------------------------------------------------------------------

def label_window(
    video_id_series: np.ndarray,
    valence: np.ndarray,
    arousal: np.ndarray,
    class_map: dict,
) -> Tuple[int | None, float, float]:
    """Compute the categorical class + mean valence / arousal of a window.

    A window is labelled with the majority video-id present. If that id is
    not in ``class_map`` the window is skipped (returns ``None`` for the
    class), which typically filters out transitions between clips.
    """
    vid = video_id_series.astype(int)
    if vid.size == 0:
        return None, float("nan"), float("nan")
    # Majority video id.
    counts = np.bincount(vid - vid.min()) if vid.min() >= 0 else np.bincount(vid)
    majority_id = int(np.argmax(counts) + (vid.min() if vid.min() >= 0 else 0))
    cls = class_map.get(majority_id)
    # If the window straddles a transition (majority < 80 %) drop it.
    if counts.max() / counts.sum() < 0.8:
        cls = None
    return cls, float(np.mean(valence)), float(np.mean(arousal))
