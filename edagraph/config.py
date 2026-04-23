"""Default configuration for the EDA-Graph pipeline.

All numeric values here follow the description of the EDA-Graph paper and the
CASE dataset (Sharma et al., 2019). They can be overridden either by editing a
``config.yaml`` file or by passing a :class:`Config` instance to the pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, Sequence

import yaml


# Mapping of CASE video-id -> categorical emotion class used in the paper.
# The CASE protocol plays 2 clips per category plus "start/end" neutral blue
# screens. Sharma et al. (2019) label the videos as follows:
#   1, 2  -> amusing (A)
#   3, 4  -> boring  (B)
#   5, 6  -> relaxing(R)
#   7, 8  -> scary   (S)
#   10,11 -> start/end neutral blue screens (N)
# Numerical labels follow the order used by the original EDA_graph_features.csv.
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
        Sampling rate of the raw EDA/GSR recording (CASE: 1000 Hz).
    fs : int
        Target sampling rate after decimation. Using 8 Hz keeps all
        information of the Skin Conductance Level / Response while
        drastically shrinking the graph.
    lowpass_hz : float
        Cut-off of the 4th-order zero-phase Butterworth low-pass used to
        remove high frequency noise before decimation.
    window_sec : float
        Length of the analysis window in seconds. The paper reports 20 s
        windows (non-overlapping) for the categorical classification task.
    window_step_sec : float
        Step between consecutive windows. Equal to ``window_sec`` for the
        non-overlapping configuration.
    q_levels : int
        Number of amplitude quantisation levels ``Q`` used to build the
        graph. The paper found ``Q = 10`` to be the best trade-off
        between graph complexity and classification performance.
    knn_k : int
        Number of nearest-neighbours connected to each graph node. The
        paper uses ``K = 8``.
    normalize : str
        Normalisation applied before quantisation, one of
        ``"minmax"`` (default) or ``"zscore"``.
    class_map : dict
        Mapping of CASE video ids to categorical classes.
    """

    # ---- acquisition / preprocessing -----------------------------------
    fs_raw: int = 1000
    fs: int = 8
    lowpass_hz: float = 1.0
    window_sec: float = 20.0
    window_step_sec: float = 20.0

    # ---- graph construction --------------------------------------------
    q_levels: int = 10
    knn_k: int = 8
    normalize: str = "minmax"

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
        # yaml loads int keys in class_map as strings when dumped; fix it.
        if "class_map" in data:
            data["class_map"] = {int(k): int(v) for k, v in data["class_map"].items()}
        return cls(**data)

    @property
    def window_samples(self) -> int:
        return int(round(self.window_sec * self.fs))

    @property
    def step_samples(self) -> int:
        return int(round(self.window_step_sec * self.fs))
