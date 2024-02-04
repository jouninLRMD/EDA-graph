
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
