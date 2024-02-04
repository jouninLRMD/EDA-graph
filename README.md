
## README Documentation

### Project Overview
This project focuses on classifying emotional states from physiological and graph-based features extracted from subjects. It leverages a combination of traditional physiological measures and advanced graph-based analytics to predict emotions, aiming to advance the understanding of emotional states through valence and arousal dimensions.

### Files Overview

- **EDA_graph_features.csv**: Contains graph-based features with metrics like centrality measures and spectrum features, alongside emotional state measurements (valence and arousal).

- **EDA_Traditional_Features.csv**: Features traditional physiological measurements such as skin conductance and heart rate variability, with emotional state measurements.

- **EMOTION_CLASSIFICATION_LAST.ipynb**: A comprehensive Jupyter Notebook for emotion classification that includes data preprocessing, analysis, visualization, and machine learning model implementation.

- **Feature_Analysis.py**: A Python script for feature analysis, likely including data processing, analysis routines, and visualization functions to support emotion classification efforts.

### Environment Setup

To set up the Python environment necessary for running this project, please follow these steps:

1. Ensure you have Anaconda or Miniconda installed on your system.
2. Download the `env.yaml` file from the repository.
3. Create the environment from the `env.yaml` file by running:
   ```bash
   conda env create -f env.yaml
   ```
4. Activate the environment:
   ```bash
   conda activate pytorch
   ```

This environment includes all necessary dependencies, including PyTorch, CUDA libraries for GPU acceleration, and various Python libraries for data analysis, visualization, and machine learning.

### Requirements and Installation

- Python environment setup as described above.
- Additional libraries or tools if not covered by the `env.yaml` file.

### Usage Instructions

1. **Running the Notebook**:
   - Ensure the `pytorch` environment is activated.
   - Start Jupyter Notebook or JupyterLab.
   - Open `EMOTION_CLASSIFICATION_LAST.ipynb`.
   - Run all cells to perform the emotion classification analysis.

2. **Using the Feature Analysis Script**:
   - With the `pytorch` environment activated, run:
     ```bash
     python Feature_Analysis.py
     ```
   - Follow any on-screen instructions to complete the feature analysis.

### Authors and Institution

- **Authors**: Luis Roberto Mercado Diaz, PhD. Yedukondala Rao Veeranki, Ph.D.Fernando Marmolejo-Ramos, Professor Ph.D.Hugo F. Posada-Quintero
- **Institution**: This Project was developed on Posada-Quintero Laboratory in the University of Connecticut

### Contributing

We welcome contributions! If you have suggestions for improvements, please:
- Fork the repository.
- Create a new branch for your feature or fix.
- Submit a pull request with a clear description of your changes.

### License

This project is released under the MIT License. By using or contributing to this project, you agree to abide by its terms.

### Methodology Overview

Our methodology leverages a novel Graph Signal Processing (GSP) approach to transform EDA signals into graphical representations, capturing the intricate dynamics of emotional states. We have quantized EDA signals into discrete levels, defining nodes and constructing edges through Euclidean distances. The extracted multi-scale EDA-graph features, encompassing graph-level, node-level, and edge-level characteristics, are pivotal for distinguishing various emotional states. Statistical analysis underscores the significance of these features, enhancing the precision of emotion classification.

### Emotional State Detection Visualizations

To visualize the emotional state detection process and results, refer to the figures included in our repository:

![Emotional State Valence-Arousal](figures/Fig.1_discretization.jpg)
*Figure 1: Discretization of emotional states in the valence-arousal plane.*

![EDA Signal Processing Steps](figures/Fig.2_step_by_step_graph.jpg)
*Figure 2: Step-by-step visualization of EDA signal processing into graph representations.*

![Graphs for Amused vs. Relaxed States](figures/Fig.3_Graphs_representation.jpg)
*Figure 3: Comparative EDA-graphs for 'Amused' and 'Relaxed' emotional states.*

![Box Plots of Traditional EDA Features](figures/Fig.4_comparison_box_plots_traditionals_lab_features_reviewed_significants.jpg)
*Figure 4: Box plot comparisons of traditional EDA features across different emotional states.*

![Graph-Based Feature Comparison](figures/Fig.5_comparison_euclidean_8nn_box_plots_graphs_significants.jpg)
*Figure 5: Comparison of significant graph-based EDA features for emotion classification.*
