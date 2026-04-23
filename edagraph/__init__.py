"""EDA-Graph: Graph Signal Processing of Electrodermal Activity for Emotional States Detection.

Reference
---------
Mercado-Diaz, L. R., Veeranki, Y. R., Marmolejo-Ramos, F., & Posada-Quintero, H. F. (2024).
EDA-Graph: Graph Signal Processing of Electrodermal Activity for Emotional States Detection.
IEEE Journal of Biomedical and Health Informatics. https://doi.org/10.1109/JBHI.2024.3405975

This package only contains the code required to
1. turn an EDA segment into an EDA-graph (quantisation + k-NN) and
2. extract the 58 graph / node / edge-level features described in Table II
   of the paper.

No classification, statistics or plotting utilities are provided here -
those belong to the downstream analysis scripts used to produce the
published results.
"""
from .config import Config, CASE_CLASS_MAP, CASE_VIDEO_CLASSES
from .preprocessing import preprocess_eda, segment_signal, label_window
from .quantization import (
    quantize_signal,
    node_values_from_quantization,
    graph_nodes_from_quantization,
)
from .graph import build_eda_graph, knn_graph_from_points
from .features import FEATURE_LEVELS, GRAPH_FEATURE_NAMES, extract_graph_features
from .pipeline import EDAGraphPipeline
from .dataset import load_case_subject, iter_case_windows

__all__ = [
    "Config",
    "CASE_CLASS_MAP",
    "CASE_VIDEO_CLASSES",
    "preprocess_eda",
    "segment_signal",
    "label_window",
    "quantize_signal",
    "node_values_from_quantization",
    "graph_nodes_from_quantization",
    "build_eda_graph",
    "knn_graph_from_points",
    "extract_graph_features",
    "FEATURE_LEVELS",
    "GRAPH_FEATURE_NAMES",
    "EDAGraphPipeline",
    "load_case_subject",
    "iter_case_windows",
]

__version__ = "1.0.0"
