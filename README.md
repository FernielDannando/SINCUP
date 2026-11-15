# SINCUP — Unified Machine Learning Platform

**An interactive Python application for clustering, classification, regression, and exploratory data analysis.**

SINCUP is a machine learning application built with Python, Streamlit, and scikit-learn. It provides a unified interface for exploring datasets, training machine learning models, evaluating their performance, and visualizing results.

The project focuses on making machine learning workflows configurable, interactive, and easier to interpret.

## Key Features

### 1. Data Import and Preprocessing

- Import CSV, Excel, and JSON datasets.
- Select numerical features for analysis.
- Handle missing numerical values using mean or median imputation.
- Apply Standard, MinMax, or Robust scaling.
- Sample datasets for experimentation.

### 2. Clustering

Supported algorithms:

- K-Means
- Gaussian Mixture Models (GMM)
- Agglomerative Clustering
- Spectral Clustering
- DBSCAN
- HDBSCAN (optional dependency)

Additional capabilities:

- Automatic cluster-count selection using Silhouette Score for supported algorithms.
- Automatic outlier detection using Isolation Forest or robust Z-scores.
- Manual exclusion and reinclusion of observations.
- Interactive 2D and 3D cluster visualizations.
- Dimensionality reduction using PCA and t-SNE.
- Convex Hull visualizations.
- Cluster visibility controls.
- Export of individual clusters to CSV.

### 3. Classification

Supported models:

- K-Nearest Neighbors (KNN)
- Logistic Regression
- Decision Tree Classifier

Evaluation features:

- Stratified cross-validation.
- Accuracy, macro precision, and macro recall.
- Confusion matrix visualization.

### 4. Regression

Supported models:

- Linear Regression
- Ridge Regression
- Lasso Regression
- K-Nearest Neighbors Regression

Evaluation features:

- K-Fold cross-validation.
- R², RMSE, and MAE metrics.
- Optional regression analysis within individual clusters.

### 5. Model Interpretation and Visualization

- Global and per-cluster correlation matrices.
- Cluster-level descriptive statistics.
- Feature differentiation analysis.
- Logistic regression coefficients for cluster interpretation.
- Interactive Plotly visualizations.

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Application logic |
| Streamlit | Interactive web interface |
| scikit-learn | Machine learning algorithms |
| Pandas | Data manipulation |
| NumPy | Numerical operations |
| Plotly | Interactive visualizations |
| SciPy | Convex Hull computation |
| HDBSCAN | Optional density-based clustering |

## Getting Started

### Prerequisites

- Python 3.10 or later is recommended.
- pip package manager.

### Installation

Clone the repository:

```bash
git clone https://github.com/FernielDannando/SINCUP.git
cd SINCUP
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install the core dependencies:

```bash
python -m pip install streamlit scikit-learn pandas numpy plotly scipy openpyxl
```

HDBSCAN can be installed separately if needed:

```bash
python -m pip install hdbscan
```

Run the application:

```bash
streamlit run Clusters.py
```

The application will provide a local URL that can be opened in your browser.

**Note:** Installation and execution instructions should be validated against the current repository environment.

## Current Architecture

The current implementation uses a Streamlit application containing data preprocessing, model execution, evaluation, visualization, and user interface logic.

The architecture is being reviewed for improved modularity, testability, and maintainability.

## Planned Improvements

- Separate machine learning algorithms from the user interface.
- Introduce automated unit and integration tests.
- Improve dataset validation and error handling.
- Strengthen model evaluation and reproducibility.
- Add dependency management and continuous integration.
- Improve performance when processing larger datasets.
- Expand technical documentation.

## Project Motivation

SINCUP was created to explore how different machine learning techniques can be integrated into a single interactive environment.

Rather than limiting the application to one algorithm, the project provides a configurable workflow for comparing approaches and interpreting their results.

## Development Status

**Active development / technical improvement phase**

The application includes working implementations of multiple machine learning workflows. Additional engineering work is planned to improve reliability, architecture, testing, and maintainability.

## Ownership

This repository contains a project maintained by its owner. No open-source license is currently declared in this README.

All rights are reserved unless a separate license or applicable agreement states otherwise.
