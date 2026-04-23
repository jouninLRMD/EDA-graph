"""Default configuration for the EDA-Graph pipeline.

All numeric values follow the methodology section of the EDA-Graph paper
(Mercado-Diaz et al., 2024, IEEE JBHI) and the CASE dataset
(Sharma et al., 2019). Overrides may be supplied through a YAML file or by
passing a :class:`Config` instance to the pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict

import yaml


# Mapping of CASE video-id -> categorical emotion class used in the paper.
# The CASE protocol plays 2 clips per category plus start/end neutral blue
# screens. Numerical labels follow the order used by EDA_graph_features.csv.
CASE_VIDEO_CLASSES: Dict[int, int] = {
    10: 0, 11: 0,          # Neutral (start / end blue screens)
    1: 1, 2: 1,            # Amused
    3: 2, 4: 2,            # Bored
    5: 3, 6: 3,            # Relaxed
    7: 4, 8: 4,            # Scared
}

# Class index -> human readable name.
CASE_CLASS_MAP: Dict[int, str] = {
    0: "Neutral",
    1: "Amused",
    2: "Bored",
    3: "Relaxed",
    4: "Scared",
}


@dataclass
class Config:
    """Pipeline configuration.

    Attributes
    ----------
    fs_raw : int
        Sampling rate of the raw EDA/GSR recording (CASE: ``1000 Hz``).
    fs : int
        Target sampling rate after decimation (paper: ``8 Hz``).
    lowpass_hz : float
        Cut-off of the 4th-order zero-phase Butterworth low-pass
        applied **after** decimation (paper: ``1 Hz``).
    median_filter_sec : float
        Length in seconds of the median filter applied after the
        low-pass (paper: ``1 s``, i.e. ``8`` samples at 8 Hz). Set to
        ``0`` to disable.
    window_sec : float
        Length of the analysis window in seconds. The paper uses
        ``60 s`` windows with 50 % overlap for the categorical
        classification task.
    window_step_sec : float
        Step between consecutive windows. Equal to ``window_sec / 2``
        for the 50 %-overlap configuration.
    majority_ratio : float
        Minimum fraction of samples in a window that must share the
        same emotional label for the window to be retained
        (paper: ``0.8``).
    q_step : float
        Quantisation *step size* in microsiemens (paper: ``Q = 0.05
        uS``). The discrete values are produced through
        :math:`x_q = Q \\cdot \\operatorname{round}(x / Q)`.
    knn_k : int
        Number of nearest neighbours connected to each graph node
        (paper: ``K = 8``, matched to the 8 Hz sampling rate).
    """

    # ---- acquisition / preprocessing -----------------------------------
    fs_raw: int = 1000
    fs: int = 8
    lowpass_hz: float = 1.0
    median_filter_sec: float = 1.0

    # ---- windowing ------------------------------------------------------
    window_sec: float = 60.0
    window_step_sec: float = 30.0       # 50 % overlap
    majority_ratio: float = 0.8

    # ---- graph construction --------------------------------------------
    q_step: float = 0.05                # quantisation step size in microsiemens
    knn_k: int = 8

    # ---- labelling ------------------------------------------------------
    class_map: Dict[int, int] = field(default_factory=lambda: dict(CASE_VIDEO_CLASSES))

    # ---- misc -----------------------------------------------------------
    random_state: int = 42

    # ---- io helpers -----------------------------------------------------
    def to_yaml(self, path: str | Path) -> None:
        path = Path(path)
        path.write_text(yaml.safe_dump(asdict(self), sort_keys=False))

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        data = yaml.safe_load(Path(path).read_text())
        if "class_map" in data:
            data["class_map"] = {int(k): int(v) for k, v in data["class_map"].items()}
        return cls(**data)

    @property
    def window_samples(self) -> int:
        return int(round(self.window_sec * self.fs))

    @property
    def step_samples(self) -> int:
        return int(round(self.window_step_sec * self.fs))

    @property
    def median_filter_samples(self) -> int:
        n = int(round(self.median_filter_sec * self.fs))
        # median filter kernel must be odd
        return max(n + (1 - n % 2), 0)
