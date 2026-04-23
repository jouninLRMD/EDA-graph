# EDA-Graph

**Graph Signal Processing of Electrodermal Activity for Emotional States Detection.**

This repository is the reference implementation of the paper

> Mercado-Diaz, L. R., Veeranki, Y. R., Marmolejo-Ramos, F., &
> Posada-Quintero, H. F. (2024). *EDA-Graph: Graph Signal Processing of
> Electrodermal Activity for Emotional States Detection.*
> **IEEE Journal of Biomedical and Health Informatics.**
> [doi:10.1109/JBHI.2024.3405975](https://doi.org/10.1109/JBHI.2024.3405975)
> • [IEEE Xplore](https://ieeexplore.ieee.org/document/10539177)

EDA-Graph turns a 1-D electrodermal activity signal into a network by
quantising the amplitude and linking time-adjacent samples through a
Euclidean *k*-nearest-neighbour rule. From that graph we extract 59
multi-scale features – degree/centrality, spectrum of the adjacency and
Laplacian, clique statistics, etc. – which outperform the four most widely
used traditional EDA features (*mean_SCL*, *nsSCR*, *TVSymp*, *LF*) when
classifying the five emotional states annotated in the
[CASE dataset](https://www.nature.com/articles/s41597-019-0209-0).

![Step-by-step graph construction](Paper_Figures/Fig_2_step_by_step_graph.jpg)

---

## 1. Quickstart

```bash
# 1. clone the repository and install the Python package
git clone https://github.com/jouninlrmd/eda-graph
cd eda-graph
pip install -r requirements.txt
pip install -e .

# 2. run the unit tests (no external data required)
pytest tests/ -v

# 3. extract features from your CASE dataset
python scripts/extract_features.py \
    --data-root /path/to/CASE/interpolated \
    --output-graph EDA_graph_features.csv \
    --output-traditional EDA_Traditional_Features.csv \
    --n-jobs -1

# 4. reproduce the Leave-One-Subject-Out classification experiment
python scripts/run_classification.py \
    --features EDA_graph_features.csv \
    --k-best 5 \
    --output results/graph_classification.csv

# 5. reproduce the statistical analysis (Anderson-Darling, Kruskal-Wallis, Dunn + FDR)
python scripts/run_statistics.py \
    --features EDA_graph_features.csv \
    --output-dir results/stats
```

All scripts are also exposed as console entry points once the package is
installed: `edagraph-extract`, `edagraph-classify`, `edagraph-stats`.

---

## 2. Dataset

The experiments target the public **CASE** dataset (Sharma et al., 2019).
Download `CASE_full.zip` from [figshare](https://springernature.figshare.com/articles/dataset/CASE_Dataset-full/8869157)
and extract it. The expected folder layout is

```
CASE/
├── interpolated/                 <-- preferred (1000 Hz annotations + physio)
│   ├── annotations/
│   │   ├── sub_1.csv  ...  sub_30.csv
│   │   └── ...
│   └── physiological/
│       ├── sub_1.csv  ...  sub_30.csv
│       └── ...
└── raw/                          <-- also supported
    ├── annotations/
    └── physiological/
```

Each `annotations/sub_X.csv` contains the columns
`jstime, valence, arousal, video` and each `physiological/sub_X.csv`
contains `daqtime, ecg, bvp, gsr, rsp, skt, emg_zygo, emg_coru, emg_trap, video`.
The EDA signal is taken from the `gsr` column, sampled at **1000 Hz**.

### Categorical class mapping

| Video id | Class id | Label      |
|----------|----------|------------|
| 10, 11   | 0        | Neutral (N)|
| 1, 2     | 1        | Amused (A) |
| 3, 4     | 2        | Bored (B)  |
| 5, 6     | 3        | Relaxed (R)|
| 7, 8     | 4        | Scared (S) |

Windows whose majority video-id is not in the table (i.e. transitions
between clips) are discarded automatically.

---

## 3. Methodology

### 3.1 Signal preprocessing (`edagraph.preprocessing`) - Section II-B

Applied exactly in the order used in the paper:

1. **Standard decimation** from `fs_raw = 1000 Hz` to `fs = 8 Hz`.
2. **4th-order zero-phase Butterworth low-pass** at `lowpass_hz = 1 Hz`,
   applied *after* decimation to eliminate high-frequency noise while
   retaining tonic baseline shifts and phasic drivers below 1 Hz.
3. **1-second median filter** (`median_filter_sec = 1.0`, i.e. 8 samples
   at 8 Hz) to further smooth the signal.
4. **Sliding windowing**: `window_sec = 60 s` with **50 % overlap**
   (`window_step_sec = 30 s`). A window is retained only when at least
   `majority_ratio = 0.8` of its samples share the same emotional
   label; the others are discarded.

### 3.2 Graph construction (`edagraph.graph.build_eda_graph`) - Section II-C

**Step 1 - Quantisation.** Each sample is rounded to the closest
multiple of the quantisation step `Q` in microsiemens:

$$ x_{\text{quantized}} = Q \cdot \operatorname{round}\!\bigl(x_{\text{original}} / Q\bigr). $$

The paper reports `Q = 0.05 μS` as optimal after sweeping Q from 0.001
to 0.99 μS - this is the default in `config.yaml`.

**Step 2 - Node definition.** Nodes are the *unique* values of
`x_quantized`, keeping only the first occurrence of each new value:

$$ X_{\text{nodes}} = x_{\text{quantized}}\bigl[x_{\text{quantized}} \ne x_{\text{quantized,shifted}}\bigr]. $$

**Step 3 - Distance calculation.** The Euclidean distance between every
pair of nodes (one-dimensional, `M = 1`):

$$ D_{ij} = \sqrt{\sum_{k=1}^{M}\! (x_{ik} - x_{jk})^2} = |x_i - x_j|. $$

**Step 4 - Nearest neighbours.** Each node is connected to its **`K = 8`**
closest nodes, matched to the 8 Hz sampling rate. For other sampling
rates (e.g. 10 Hz), `knn_k` should be set accordingly.

**Step 5 - Adjacency.** Edge weight `w_ij = 1 / D_ij` (equation 5 of the
paper). The adjacency matrix is symmetric, weighted, and has zeros on
the diagonal.

### 3.3 Features (`edagraph.features`) - Section II-D and Table II

Features are grouped by level (see ``edagraph.features.FEATURE_LEVELS``):

**Graph-level** (40) - Total Triangle Number, Graph Energy, Transitivity,
Cliques Counts (`graph_clique_num`), Number of Cliques, Graph Is
Chordal, Center (`P_center`), Diameter, Radius, Periphery, Average
Clustering (both global and on the largest component), Weisfeiler-
Lehman Kernel, Graph / Laplacian Spectrum Std, Avg Triangle
Participation, and the 24 GS/LS Mean/Min/Max/Median/Skewness/Kurtosis
of Magnitude *and* Phase.

**Node-level** (14) - Total Degree / Closeness / Betweenness / Eigenvector
/ Load / Harmonic Centrality, Total PageRank, Total Hubs, Number of
Nodes, Maximum / Minimum / Median Degree, Closeness Centrality
(`P_closeness_centrality`), Eccentricity.

**Edge-level** (5) - Total Flow Centrality, Total Log Flow Centrality,
Number of Edges, Assortativity, Spearman Correlation.

The extracted dictionary uses the exact same column order as
``EDA_graph_features.csv`` shipped with the paper (59 columns - the
``avg_clustering_coefficient`` global + the ``P_`` version count as one
feature family in Table II, giving the paper's total of 58).

**Traditional features (4 columns):**

| Name      | Description                                                 |
|-----------|-------------------------------------------------------------|
| `mean_SCL`| mean Skin Conductance Level (microsiemens, tonic component) |
| `nsSCR`   | non-specific Skin Conductance Responses per minute          |
| `TVSymp`  | Time-Varying Sympathetic index: PSD of phasic EDA in [0.045, 0.25] Hz |
| `LF`      | low-frequency power ~ `TVSymp * 1e3` (mS$^2$)               |

Phasic/tonic decomposition uses [NeuroKit2](https://neurokit2.readthedocs.io/)
(`cvxEDA` method). A pure-SciPy fallback is used if NeuroKit2 is not
installed.

### 3.4 Classification (`edagraph.experiments`) - Section II-F and Table IV

* Leave-One-Subject-Out cross-validation via `GroupKFold`.
* **Eight classifiers** with an inner grid-search matching Table IV:
  Gaussian NB, KNN, Random Forest, AdaBoost, Gradient Boosting,
  Decision Tree, SVM (RBF), Bagging Ensemble.
* Optional `SelectKBest(f_classif, k=K)` upstream (`--k-best 5` in the
  paper) - ANOVA F-value feature ranking (Section II-G).
* Metrics reported: accuracy, balanced accuracy, macro-F1, weighted-F1
  and the full `classification_report` (paper's primary metrics:
  balanced accuracy and F1).

### 3.5 Statistical analysis (`edagraph.stats`) - Section II-E

* **Anderson-Darling** normality test per feature × class.
* **Kruskal-Wallis** one-way ANOVA on ranks; features with
  `p < 0.05` advance to post-hoc.
* **Dunn** pair-wise post-hoc with **Holm-Bonferroni** correction.
  The paper's adjusted significance threshold is ``alpha = 0.005``;
  a Benjamini-Hochberg FDR correction can additionally be enabled via
  ``--apply-fdr`` if desired.

---

## 4. Repository layout

```
eda-graph/
├── edagraph/                     Core Python package
│   ├── config.py                 Dataclass of pipeline hyper-parameters
│   ├── preprocessing.py          Filtering, decimation, windowing, labelling
│   ├── quantization.py           Amplitude quantisation and node coords
│   ├── graph.py                  KD-tree-based k-NN graph construction
│   ├── features/
│   │   ├── graph_features.py     59 EDA-graph features
│   │   └── traditional.py        4 traditional EDA features
│   ├── dataset.py                CASE dataset loader
│   ├── pipeline.py               EDAGraphPipeline (joblib-parallel)
│   ├── experiments.py            LOSO classification with 7 classifiers
│   ├── stats.py                  Anderson-Darling / Kruskal / Dunn / FDR
│   └── visualize.py              Plotting helpers
├── scripts/
│   ├── extract_features.py       CLI: folder -> features CSV
│   ├── run_classification.py     CLI: features CSV -> LOSO results
│   └── run_statistics.py         CLI: features CSV -> stat tests
├── tests/test_pipeline.py        Smoke tests on a synthetic EDA signal
├── config.yaml                   Default pipeline parameters
├── requirements.txt / setup.py   Install recipe
├── EMOTION_CLASSIFICATION_LAST.ipynb   Original notebook (kept for reference)
├── Feature_Analysis.py           Legacy statistics script
├── EDA_graph_features.csv        Example feature file produced by the paper
├── EDA_Traditional_Features.csv  Example traditional-feature file
└── Paper_Figures/                Figures used in the paper and below
```

---

## 5. Python API

```python
from edagraph import Config, EDAGraphPipeline, build_eda_graph, extract_graph_features

# End-to-end for a full dataset (paper defaults: 60 s windows @ 50 % overlap, Q = 0.05 uS, K = 8)
cfg = Config()
pipe = EDAGraphPipeline(cfg=cfg, n_jobs=-1)
df = pipe.extract_all("CASE/interpolated")          # joblib-parallel over subjects
df.to_csv("EDA_graph_features.csv", index=False)

# Or on a single 60 s window of your own EDA signal
import numpy as np
window = np.load("my_eda_60s.npy")                  # 480 samples @ 8 Hz
graph = build_eda_graph(window, cfg)
features = extract_graph_features(graph)
```

### Running the LOSO experiment in Python

```python
import pandas as pd
from edagraph.experiments import run_loso_classification

df = pd.read_csv("EDA_graph_features.csv")
feats = [c for c in df.columns if c not in {"subject", "class", "valence", "arousal"}]
summary, reports = run_loso_classification(df, feature_cols=feats, k_best=5, n_jobs=-1)
print(summary)
```

---

## 6. Performance

Single-core, 60 s windows sampled at 8 Hz (`Q = 0.05 μS`, `K = 8`):

| Stage                                   | Time per window |
|-----------------------------------------|-----------------|
| Decimation + low-pass + median filter   | ~0.1 ms         |
| Quantisation + k-NN graph (cKDTree)     | ~0.3 ms         |
| 58 graph features                       | ~12 ms          |
| **Total (single window)**               | **~13 ms**      |

For the full CASE dataset (~16 447 windows across 30 subjects) the
pipeline runs in **~3–4 min on a laptop (8 cores)** versus roughly
**~2 hours** for the original un-optimised code. Speed-ups come from

* vectorised preprocessing (`scipy.signal.filtfilt` + `decimate`),
* `scipy.spatial.cKDTree` for $k$-NN (was $O(N^2)$ nested loop),
* one-shot eigendecomposition shared by every spectral feature,
* `joblib.Parallel` across subjects.

---

## 7. Reproducing the paper's key figures

| Figure in the paper | Reproduce with                                             |
|---------------------|------------------------------------------------------------|
| Fig. 1 – class grid | `Paper_Figures/Fig_1_discretization.jpg` (static)           |
| Fig. 2 – pipeline   | `edagraph.visualize.plot_eda_graph` on a sample window      |
| Fig. 3 – A vs R     | idem, overlayed for Amused and Relaxed windows              |
| Fig. 4 – optimal Q  | `python scripts/q_sweep.py --data-root CASE/interpolated --q-values 0.005,0.01,0.02,0.03,0.05,0.1,0.2,0.5` |
| Fig. 5 – trad boxes | `scripts/run_statistics.py --features EDA_Traditional_Features.csv` |
| Fig. 6 – graph boxes| `scripts/run_statistics.py --features EDA_graph_features.csv` |

![Fig. 4 – Optimal Q](Paper_Figures/Fig_4_Selection_of_optimal_Q.jpg)

---

## 8. Citing

```bibtex
@article{mercado2024edagraph,
  title   = {EDA-Graph: Graph Signal Processing of Electrodermal Activity for Emotional States Detection},
  author  = {Mercado-Diaz, Luis R. and Veeranki, Yedukondala Rao and Marmolejo-Ramos, Fernando and Posada-Quintero, Hugo F.},
  journal = {IEEE Journal of Biomedical and Health Informatics},
  year    = {2024},
  doi     = {10.1109/JBHI.2024.3405975}
}
```

## 9. Authors and institution

* Luis Roberto Mercado-Diaz, PhD
* Yedukondala Rao Veeranki, PhD
* Fernando Marmolejo-Ramos, PhD
* Hugo F. Posada-Quintero, PhD

Developed at the **Posada-Quintero Laboratory**, University of Connecticut.

## 10. License

Released under the **MIT License**.
