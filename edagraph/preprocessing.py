"""Preprocessing of raw EDA signals prior to graph construction.

Matches Section II-B of the paper:

1. Standard decimation from ``fs_raw`` (1 kHz) to ``fs`` (8 Hz).
2. 4th-order zero-phase Butterworth **low-pass** at ``lowpass_hz``
   (1 Hz) applied *after* decimation, to keep tonic baseline shifts and
   phasic drivers below that threshold.
3. 1-second (8-sample) **median filter** to further smooth the signal.
4. Sliding window segmentation: ``window_sec = 60 s`` with 50 % overlap.
5. Majority-label windowing: a window is kept only if at least
   ``majority_ratio`` (80 %) of its samples share the same emotional
   label (based on the CASE video id).
"""
from __future__ import annotations

from typing import Iterator, Tuple

import numpy as np
from scipy.signal import butter, decimate, filtfilt, medfilt

from .config import Config


# ---------------------------------------------------------------------------
# Filtering / resampling
# ---------------------------------------------------------------------------

def _butter_lowpass(signal: np.ndarray, fs: float, cutoff: float, order: int = 4) -> np.ndarray:
    """Zero-phase Butterworth low-pass (``filtfilt``)."""
    nyq = 0.5 * fs
    b, a = butter(order, cutoff / nyq, btype="low")
    return filtfilt(b, a, signal)


def preprocess_eda(raw: np.ndarray, cfg: Config) -> np.ndarray:
    """Decimate -> low-pass -> median filter.

    Parameters
    ----------
    raw : (N,) array
        Raw EDA at ``cfg.fs_raw`` Hz.
    cfg : Config
        Pipeline configuration.

    Returns
    -------
    (M,) array
        Preprocessed signal sampled at ``cfg.fs`` Hz.
    """
    raw = np.asarray(raw, dtype=np.float64).ravel()
    if raw.size == 0:
        return raw

    # 1. Decimation 1000 Hz -> 8 Hz.
    factor = int(round(cfg.fs_raw / cfg.fs))
    if factor > 1:
        # ``decimate`` applies an 8th-order Chebyshev anti-alias filter.
        signal = decimate(raw, factor, ftype="iir", zero_phase=True)
    else:
        signal = raw.copy()

    # 2. 4th-order Butterworth low-pass at lowpass_hz (after decimation).
    if cfg.lowpass_hz and cfg.lowpass_hz < cfg.fs / 2:
        signal = _butter_lowpass(signal, cfg.fs, cfg.lowpass_hz)

    # 3. 1-second median filter (8 samples at 8 Hz).
    k = cfg.median_filter_samples
    if k and k >= 3:
        signal = medfilt(signal, kernel_size=k)

    return signal


# ---------------------------------------------------------------------------
# Windowing
# ---------------------------------------------------------------------------

def segment_signal(signal: np.ndarray, cfg: Config) -> Iterator[Tuple[int, int, np.ndarray]]:
    """Iterate over ``(start_idx, end_idx, window)`` tuples.

    Uses fixed-length windows of ``cfg.window_sec`` with a step of
    ``cfg.window_step_sec`` (50 % overlap by default).
    """
    n = signal.size
    w = cfg.window_samples
    step = cfg.step_samples
    if w <= 0 or step <= 0:
        raise ValueError("window_sec and window_step_sec must be positive")

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
    majority_ratio: float = 0.8,
) -> Tuple[int | None, float, float]:
    """Compute the majority class + mean valence / arousal of a window.

    A window is retained only if at least ``majority_ratio`` of the samples
    share the same video id **and** that id is in ``class_map``. Otherwise
    the returned class is ``None`` (the window is dropped).
    """
    vid = video_id_series.astype(int).ravel()
    if vid.size == 0:
        return None, float("nan"), float("nan")

    uniq, counts = np.unique(vid, return_counts=True)
    majority_idx = int(np.argmax(counts))
    majority_id = int(uniq[majority_idx])
    fraction = counts[majority_idx] / counts.sum()

    if fraction < majority_ratio or majority_id not in class_map:
        return None, float(np.mean(valence)), float(np.mean(arousal))
    return class_map[majority_id], float(np.mean(valence)), float(np.mean(arousal))
