# Machine Learning for Call Quality Estimation: Auto-Labeling using Hybrid Strategy combining Active and Semi-Supervised Learning

This repository contains the official code implementation for the paper, "Machine Learning for Call Quality Estimation: Auto-Labeling using Hybrid Strategy combining Active and Semi-Supervised Learning".

## Abstract

Manually annotating large-scale call records for quality estimation is expensive and impractical, yet essential for optimizing call routing in telecommunication systems. This paper proposes a novel, hybrid auto-labeling framework that integrates Active Learning (AL) and Semi-Supervised Learning (SSL) to address this challenge. The framework uses domain-specific metrics like packet loss rate, round trip time, and jitter to train a model. This model then employs confidence-based pseudo-labeling and pruning to iteratively expand the training dataset while ensuring the reliability of the new labels. Experimental results show this approach significantly improves labeling efficiency and provides a scalable solution for real-world telecommunication environments.

## Framework Overview

The code implements a hybrid auto-labeling framework that combines principles from Active Learning and Semi-Supervised Learning.

1.  **Initial Training**: A model is trained on a small, manually-labeled dataset.
2.  **Pseudo-Labeling**: The model predicts labels for a large pool of unlabeled data. Predictions with a confidence score above a certain threshold are accepted as "pseudo-labels."
3.  **Iterative Retraining**: The model is retrained on an expanded dataset that includes both the original labels and the new high-confidence pseudo-labels.
4.  **Loop**: This process repeats, allowing the model to improve iteratively as the labeled dataset grows.

## Key Features

*   **Multiple Model Support**: Easily compare various classification models (XGBoost, RandomForest, Logistic Regression, etc.).
*   **Flexible Confidence Threshold**: Supports both fixed and dynamic confidence thresholds for pseudo-labeling.
*   **Evaluation Suite**: Includes utilities for generating classification reports, confusion matrices, ROC curves, and visualizing embeddings.

## Codebase Structure

The core logic is located in the `semi_supervisedlearning/` directory:

*   `main.py`: The main entry point for running the training and evaluation pipeline. Parses command-line arguments.
*   `multi_model_runner.py`: Orchestrates the end-to-end process, iterating through different models and managing the active learning cycles.
*   `active_learning.py`: Implements the core active learning cycle, including pseudo-labeling based on confidence thresholds.
*   `data_loader.py`: Handles loading the dataset, preprocessing, and feature engineering.
*   `config.py`: A centralized configuration file for model parameters, confidence thresholds, and directory paths.
*   `evaluation_utils.py`: Provides helper functions for model evaluation, including plotting confusion matrices, ROC curves, and visualizing data embeddings with PCA/t-SNE.
*   `checkpoint_manager.py`: Manages saving model checkpoints during training.

## Installation

1.  **Clone the repository:**
    ```powershell
    git clone https://github.com/duydinhthai27/routing_ML_team2.git
    cd routing_ML_team2
    ```

2.  **Create and activate a virtual environment:**
    (A Python version `< 3.10` is recommended, `3.9` is a safe choice).
    ```powershell
    # Using venv
    python -m venv .venv
    .\.venv\Scripts\Activate.ps1

    # Or using Conda
    # conda create -n voice_team2 python=3.9
    # conda activate voice_team2
    ```

3.  **Install the required packages:**
    ```powershell
    pip install -r requirements.txt
    ```

## Usage

The main script `semi_supervisedlearning/main.py` is used to run the training and evaluation pipeline.

**To run with a fixed confidence threshold (defined in `config.py`):**
```powershell
python .\semi_supervisedlearning\main.py
```

**To run with a dynamic confidence threshold:**
```powershell
python .\semi_supervisedlearning\main.py --use-dynamic-threshold
```

The script will train all models defined in the `MODEL_REGISTRY` in `config.py` and output evaluation metrics, plots, and model checkpoints.

## Key Results

*   **Top Performers**: Tree-based ensemble models, particularly **XGBoost** and **Random Forest**, achieved the highest performance, with F1-scores of nearly 0.99 and 0.97, respectively.
*   **Threshold Importance**: The confidence threshold is a critical hyperparameter. The optimal balance between adding new samples and maintaining label quality was found in the 0.85 to 0.90 range.
*   **Dynamic Thresholding**: The dynamic threshold strategy proved highly beneficial for models like AdaBoost, significantly improving its F1-score, while having a marginal impact on already robust models like XGBoost.
