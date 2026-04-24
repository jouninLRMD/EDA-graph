"""EDA-Graph: Graph Signal Processing of Electrodermal Activity for Emotional States Detection.

Reference
---------
Mercado-Diaz, L. R., Veeranki, Y. R., Marmolejo-Ramos, F., & Posada-Quintero, H. F. (2024).
EDA-Graph: Graph Signal Processing of Electrodermal Activity for Emotional States Detection.
IEEE Journal of Biomedical and Health Informatics. https://doi.org/10.1109/JBHI.2024.3405975

Three building blocks are exposed:

1. EDA-graph construction (Sections II-B and II-C).
2. Extraction of the 58 graph / node / edge-level features (Table II).
3. Statistical analysis (Section II-E) and Leave-One-Subject-Out
   classification (Section II-F) of the extracted features.
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
from .stats import anderson_darling_by_class, kruskal_wallis, dunn_posthoc
from .experiments import run_loso_classification

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
    "anderson_darling_by_class",
    "kruskal_wallis",
    "dunn_posthoc",
    "run_loso_classification",
]

__version__ = "1.0.0"
