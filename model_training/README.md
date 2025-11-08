# Network Intrusion Detection System

A comprehensive machine learning-based network intrusion detection system that combines two major cybersecurity datasets (UNSW-NB15 and CIC-IDS-2017) to perform binary classification of network traffic as benign or malicious.

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Datasets](#datasets)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Pipeline Workflow](#pipeline-workflow)
- [Model Performance](#model-performance)
- [Results](#results)
- [Technologies Used](#technologies-used)
- [Contributing](#contributing)
- [License](#license)

## Overview

This project implements an end-to-end machine learning pipeline for network intrusion detection. It processes and merges two prominent cybersecurity datasets, performs comprehensive exploratory data analysis, applies hybrid feature selection techniques, and trains multiple supervised and unsupervised learning models to detect network attacks.

### Key Objectives

- Merge and preprocess heterogeneous network traffic datasets
- Perform comprehensive exploratory data analysis with visualizations
- Apply hybrid feature selection combining correlation, Random Forest importance, and mutual information
- Train and evaluate binary classification models (Decision Tree, Logistic Regression)
- Perform unsupervised clustering analysis using K-Means
- Achieve high accuracy in distinguishing benign traffic from network attacks

## Project Structure

```
Network-Intrusion/
├── Data_Preprocessing/
│   ├── Datasets_merging_process/
│   │   ├── DownloadDataset.py          # Kaggle dataset downloader
│   │   ├── dataset_cic_ids.py          # CIC-IDS-2017 preprocessing
│   │   ├── dataset_unsw.py             # UNSW-NB15 preprocessing
│   │   └── merge_both.py               # Dataset merger
│   └── DataPreprocessing_Binaryclass.py # Binary classification preprocessing
│
├── EDA/
│   ├── Script/
│   │   ├── merged_dataset_eda_with_plots.py    # Comprehensive EDA with visualizations
│   │   ├── merged_dataset_eda.py               # Statistical EDA
│   │   ├── cic_ids_2017_eda_complete_output.py # CIC-IDS-2017 specific analysis
│   │   └── unsw_nb15_focused_analysis.py       # UNSW-NB15 specific analysis
│   └── Outputs/                        # Generated EDA plots and reports
│
├── supervised_learning/
│   └── Binary Classification/
│       ├── training_binary_DecisionTree.py     # Decision Tree classifier
│       └── training_binary_Logistic_Regression.py # Logistic Regression classifier
│
├── unsupervised_learning/
│   └── Clustering/
│       ├── kmean.py                    # K-Means clustering analysis
│       └── results_kmean/              # Clustering results and visualizations
│
├── data/
│   ├── network-intrusion-dataset/      # Raw datasets (UNSW-NB15, CIC-IDS-2017)
│   ├── merged_data/                    # Merged datasets
│   └── processed_data_binary/          # Preprocessed training/test data
│
├── models_checkpoint/
│   └── binary/                         # Trained model checkpoints
│
├── results_binary/
│   ├── DecisionTree/                   # Decision Tree results
│   └── LogisticRegression/             # Logistic Regression results
│
├── requirements.txt                    # Python dependencies
├── pyproject.toml                      # Project configuration
└── README.md                           # Project documentation
```

## Datasets

### UNSW-NB15

- **Source**: University of New South Wales
- **Description**: Modern network traffic dataset containing normal activities and synthetic attack behaviors
- **Features**: 13 selected features including protocol, state, duration, bytes, packets, TCP window sizes, and segment sizes
- **Dataset ID**: 0

### CIC-IDS-2017

- **Source**: Canadian Institute for Cybersecurity
- **Description**: Comprehensive intrusion detection dataset with multiple attack types
- **Features**: 14 selected features aligned with UNSW-NB15 schema
- **Attack Types**: DDoS, DoS, PortScan, Brute Force, Web Attacks, Infiltration, Botnet, and more
- **Dataset ID**: 1

### Merged Dataset

- **Combined Features**: 14 canonical features harmonized across both datasets
- **Label**: Binary classification (0 = BENIGN, 1 = ATTACK)
- **Preprocessing**: Duplicate removal, missing value imputation, feature engineering

## Features

### Data Processing

- **Automated Dataset Download**: Kaggle API integration for seamless dataset acquisition
- **Duplicate Removal**: Intelligent deduplication based on feature signatures
- **Missing Value Handling**: Median imputation for numerical features
- **Feature Engineering**: 7 derived features including:
  - Average packet size
  - Packet ratio (forward/backward)
  - Byte ratio
  - Request-response packet ratio
  - Window-to-payload ratio
  - Bytes per second (throughput)
  - Packets per second (packet rate)

### Feature Selection

Hybrid approach combining three methods:

1. **Correlation Filter**: Spearman correlation with target (threshold: 0.10)
2. **Random Forest Importance**: Tree-based feature importance (threshold: 0.01)
3. **Mutual Information**: Information gain analysis (threshold: 0.01)
4. **Redundancy Removal**: Eliminates highly correlated features (correlation > 0.95)

### Class Imbalance Handling

- **Training Set**: Downsampling majority class to 2:1 ratio (Benign:Attack)
- **Test Set**: Maintains original distribution for realistic evaluation
- **Model Training**: Class weights applied for balanced learning

### Exploratory Data Analysis

Comprehensive visualizations including:

- Attack type distribution analysis
- Protocol and connection state analysis
- Port usage patterns and security analysis
- Traffic volume and duration distributions
- Feature correlation heatmaps
- Cybersecurity-specific insights

## Installation

### Prerequisites

- Python 3.13 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone https://github.com/NathanVuSwinburne/Network-Intrusion.git
cd Network-Intrusion
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

Or using uv:
```bash
uv pip install -r requirements.txt
```

### Dependencies

Core libraries:
- pandas >= 1.5.0
- numpy >= 1.21.0
- scikit-learn >= 1.0.0
- matplotlib >= 3.5.0
- seaborn >= 0.11.0
- xgboost >= 1.7.0
- imbalanced-learn >= 0.10.0
- kagglehub >= 0.2.0

## Usage

### 1. Download Datasets

```bash
python Data_Preprocessing/Datasets_merging_process/DownloadDataset.py
```

This downloads UNSW-NB15 and CIC-IDS-2017 datasets from Kaggle.

### 2. Process Individual Datasets(CIC-IDS-2017 First)

```bash
# Process CIC-IDS-2017
python Data_Preprocessing/Datasets_merging_process/dataset_cic_ids.py

# Process UNSW-NB15
python Data_Preprocessing/Datasets_merging_process/dataset_unsw.py

```

### 3. Merge Datasets

```bash
python Data_Preprocessing/Datasets_merging_process/merge_both.py
```

### 4. Exploratory Data Analysis
```bash
python EDA/Script/cic_ids_2017_eda_complete_output.py
```
```bash
python EDA/Script/unsw_nb15_focused_analysis.py
```
```bash
python EDA/Script/comprehensive_eda_individual_plots.py
```

Generates comprehensive visualizations in `EDA/Outputs/`.

### 5. Preprocess for Binary Classification

```bash
python Data_Preprocessing/DataPreprocessing_Binaryclass.py
```

Performs:
- Feature engineering
- Hybrid feature selection
- Train/test split (80/20)
- Class balancing
- Standardization
- Saves processed data to `data/processed_data_binary/`

### 6. Train Supervised Models

```bash
# Decision Tree
python supervised_learning/Binary\ Classification/training_binary_DecisionTree.py

# Logistic Regression
python supervised_learning/Binary\ Classification/training_binary_Logistic_Regression.py
```

Models are saved to `models_checkpoint/binary/` with encoders.

### 7. Unsupervised Clustering Analysis

```bash
python unsupervised_learning/Clustering/kmean.py
```

Performs K-Means clustering with PCA visualization and saves results to `unsupervised_learning/Clustering/results_kmean/`.

## Pipeline Workflow

```
┌─────────────────────────────────────────────────────────────┐
│                    1. Data Acquisition                      │
│  DownloadDataset.py → UNSW-NB15 + CIC-IDS-2017 datasets    │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│                 2. Dataset Processing                       │
│  dataset_unsw.py + dataset_cic_ids.py → Standardization    │
│  • Duplicate removal • Feature alignment • Label mapping    │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│                   3. Dataset Merging                        │
│  merge_both.py → merged_datasets.csv                        │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│          4. Exploratory Data Analysis (EDA)                 │
│  merged_dataset_eda_with_plots.py → Visualizations          │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│              5. Feature Engineering & Selection             │
│  DataPreprocessing_Binaryclass.py                           │
│  • 7 engineered features • Hybrid selection                 │
│  • Class balancing • Scaling                                │
└─────────────────────┬───────────────────────────────────────┘
                      │
        ┌─────────────┴─────────────┐
        │                           │
┌───────▼──────────┐      ┌─────────▼────────────┐
│  6a. Supervised  │      │  6b. Unsupervised    │
│     Learning     │      │      Learning        │
│  • Decision Tree │      │  • K-Means Clustering│
│  • Log Regression│      │  • PCA Visualization │
└──────────────────┘      └──────────────────────┘
```

## Model Performance

### Binary Classification Models

Models are trained with:
- **Class Weights**: Balanced to handle imbalanced data
- **Evaluation Metrics**: Accuracy, Precision, Recall, F1-Score
- **Confusion Matrix**: Visual performance analysis
- **Test Set**: Real-world distribution maintained

### Clustering Analysis

- **Algorithm**: K-Means with optimal K selection
- **Dimensionality Reduction**: PCA (2D for visualization, 10D for clustering)
- **Outlier Removal**: Isolation Forest (1% contamination)
- **Evaluation Metrics**: Silhouette Score, Davies-Bouldin Index
- **Cluster Interpretation**: Benign vs Attack composition analysis

## Results

### Output Directories

- **EDA/Outputs/**: Exploratory data analysis plots
  - Attack distribution analysis
  - Port analysis
  - Traffic analysis
  - Correlation heatmaps
  - Cybersecurity insights

- **results_binary/**: Supervised learning results
  - Confusion matrices
  - Classification reports
  - Prediction results (CSV)
  - Class mappings

- **unsupervised_learning/Clustering/results_kmean/**: Clustering results
  - PCA visualizations
  - Elbow plots
  - Cluster distribution analysis
  - Clustering metrics report

- **models_checkpoint/binary/**: Trained model files
  - Model weights (pickle format)
  - Label encoders
  - Protocol and state encoders

## Technologies Used

### Machine Learning

- **scikit-learn**: Classification, clustering, preprocessing, feature selection

### Data Processing

- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computations
- **scipy**: Statistical functions

### Visualization

- **matplotlib**: Base plotting library
- **seaborn**: Statistical visualizations
- **plotly**: Interactive plots (optional)

### Dataset Management

- **kagglehub**: Automated dataset downloading from Kaggle

## Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/YourFeature`)
3. Commit your changes (`git commit -m 'Add YourFeature'`)
4. Push to the branch (`git push origin feature/YourFeature`)
5. Open a Pull Request

### Areas for Contribution

- Additional classification algorithms (Random Forest, SVM, Neural Networks)
- Multi-class attack type classification
- Real-time intrusion detection system
- Model deployment and API development
- Performance optimization
- Additional dataset integration

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- **UNSW-NB15 Dataset**: University of New South Wales, Canberra
- **CIC-IDS-2017 Dataset**: Canadian Institute for Cybersecurity
- **Kaggle**: Dataset hosting and API access

## Contact

For questions, issues, or collaboration opportunities, please open an issue on GitHub.

---

**Project Status**: Active Development

**Last Updated**: October 2025
