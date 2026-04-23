"""Traditional EDA features used as a baseline in the paper.

Features
--------
* ``mean_SCL``  - mean Skin Conductance Level (microsiemens).
* ``nsSCR``     - number of non-specific Skin Conductance Responses per
                  minute, obtained from ``neurokit2.eda_peaks``.
* ``TVSymp``    - time-varying sympathetic activity index, computed from
                  the spectral power of the phasic component in
                  :math:`[0.045, 0.25]` Hz.
* ``LF``        - low frequency power of EDA in :math:`[0.045, 0.25]` Hz
                  (expressed in mS^2).
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
from scipy.signal import welch


# ``np.trapz`` was removed in NumPy 2.0 in favour of ``np.trapezoid``.
_trapezoid = getattr(np, "trapezoid", None) or np.trapz  # type: ignore[attr-defined]


TRADITIONAL_FEATURE_NAMES: List[str] = ["mean_SCL", "nsSCR", "TVSymp", "LF"]


def _phasic_tonic(eda: np.ndarray, fs: float):
    """Return tonic (SCL) and phasic (SCR) components using NeuroKit2 if available."""
    try:
        import neurokit2 as nk  # type: ignore
        decomposed = nk.eda_phasic(eda, sampling_rate=fs, method="cvxEDA")
        return (
            np.asarray(decomposed["EDA_Tonic"].values, dtype=np.float64),
            np.asarray(decomposed["EDA_Phasic"].values, dtype=np.float64),
        )
    except Exception:
        # Fallback: low-pass filter as SCL, residual as SCR.
        from scipy.signal import butter, filtfilt
        b, a = butter(2, 0.05 / (fs / 2), btype="low")
        tonic = filtfilt(b, a, eda)
        return tonic, eda - tonic


def _count_scrs(phasic: np.ndarray, fs: float) -> float:
    try:
        import neurokit2 as nk  # type: ignore
        _, info = nk.eda_peaks(phasic, sampling_rate=fs, method="gamboa2008")
        n_peaks = len(info.get("SCR_Peaks", []))
    except Exception:
        # Fallback: derivative threshold at 0.05 uS/s as per Boucsein (2012).
        d = np.diff(phasic) * fs
        thr = 0.05
        crossings = np.where((d[:-1] < thr) & (d[1:] >= thr))[0]
        n_peaks = crossings.size
    duration_min = max(len(phasic) / fs / 60.0, 1e-6)
    return float(n_peaks / duration_min)


def _band_power(signal: np.ndarray, fs: float, fmin: float, fmax: float) -> float:
    """Power-spectral-density integrated over ``[fmin, fmax]`` via Welch."""
    if signal.size < 8:
        return 0.0
    nperseg = min(signal.size, max(int(fs * 30), 32))
    freqs, psd = welch(signal, fs=fs, nperseg=nperseg)
    mask = (freqs >= fmin) & (freqs <= fmax)
    if not mask.any():
        return 0.0
    return float(_trapezoid(psd[mask], freqs[mask]))


def extract_traditional_features(eda: np.ndarray, fs: float) -> Dict[str, float]:
    """Compute the 4 traditional EDA features on a window.

    Parameters
    ----------
    eda : (N,) array
        EDA signal in microsiemens sampled at ``fs`` Hz.
    fs : float
        Sampling rate of ``eda``.
    """
    eda = np.asarray(eda, dtype=np.float64).ravel()
    tonic, phasic = _phasic_tonic(eda, fs)

    mean_scl = float(np.mean(tonic))
    ns_scr = _count_scrs(phasic, fs)
    tvsymp = _band_power(phasic, fs, 0.045, 0.25)
    lf = tvsymp * 1e3  # scale to mS^2 units used by the paper

    return {
        "mean_SCL": mean_scl,
        "nsSCR": ns_scr,
        "TVSymp": tvsymp,
        "LF": lf,
    }
