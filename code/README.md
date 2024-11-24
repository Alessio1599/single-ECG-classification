# Codes
## Folder structure
```
├── code
│   ├── README.md
│   ├── data_exploratory.py
│   ├── eval.py
│   ├── hyperparameter tuning
│   │   ├── CNN_hyperparameter_optimization.py
│   │   ├── README.md
│   │   ├── RNN_hyperparameter_optimization.py
│   │   └── sweep_conf
│   │       ├── cnn_sweep_config_bayes.yaml
│   │       ├── rnn_sweep_config_bayes.yaml
│   │       └── sweep-grid.yaml
│   ├── main.py
│   ├── models
│   │   ├── CNN.py
│   │   ├── RNN.py
│   │   ├── cnn
│   │   │   └── best_cnn_model_v1.keras
│   │   └── rnn
│   │       └── best_RNN_model_v1.keras
│   ├── training
│   │   ├── CNN_train.py
│   │   └── RNN_train.py
│   └── utils
│       ├── utils.py
│       └── utils_DL.py
```

> In zero.py I reimplemented the things without using functions!!

## Changes that I made
- At the beginnning I had a utils folder, now I have all the utility functions inside util.py

## Contents

- **`data_exploratory.py`**:  
  Script for initial data exploration and visualization. It helps in understanding the dataset's structure and characteristics.

- **`eval.py`**:  
  Script for evaluating the performance of trained models. This includes metrics computation and visualization of results.

- **`hyperparameter_tuning`**:  
  Directory containing scripts and configuration files for optimizing model hyperparameters.
  - **`CNN_hyperparameter_optimization.py`**:  
    Script dedicated to optimizing hyperparameters for CNN architectures.
  - **`RNN_hyperparameter_optimization.py`**:  
    Script dedicated to optimizing hyperparameters for RNN architectures.
  - **`README.md`**:  
    Detailed description of the hyperparameter tuning procedures and methodologies used.
  - **`sweep_conf`**:  
    Configuration files for hyperparameter sweeps.
    - **`cnn_sweep_config_bayes.yaml`**:  
      Bayesian configuration file for sweeping CNN hyperparameters.
    - **`rnn_sweep_config_bayes.yaml`**:  
      Bayesian configuration file for sweeping RNN hyperparameters.
    - **`sweep-grid.yaml`**:  
      Grid search configuration for hyperparameter tuning across different models.

- **`models/`**:  
  Contains model definitions and saved models.
  - **`CNN.py`**:  
    Script defining the architecture of the CNN.
  - **`RNN.py`**:  
    Script defining the architecture of the RNN.
  - **`cnn/`**:  
    Directory containing the trained CNN model file.
    - **`best_cnn_model_v1.keras`**:  
      Saved model file for the best performing CNN version.
  - **`rnn/`**:  
    Directory containing the trained RNN model file.
    - **`best_RNN_model_v1.keras`**:  
      Saved model file for the best performing RNN version.

- **`training/`**:  
  Scripts used for training the models.
  - **`CNN_train.py`**:  
    Script for training the CNN model.
  - **`RNN_train.py`**:  
    Script for training the RNN model.

- **`utils/`**:  
  Utility scripts used throughout the project. These contain helper functions and utilities for data processing and model training.
  - **`utils.py`**:  
    General utility functions for data handling and preprocessing.
  - **`utils_DL.py`**:  
    Functions specifically designed for deep learning tasks, including data augmentation and preprocessing methods.

## How to Use

- Start by running `data_exploratory.py` to understand your dataset.
- Use `CNN_train.py` or `RNN_train.py` to train your models.
- Optimize hyperparameters using scripts in the `hyperparameter_tuning` folder.
- Evaluate the trained models with `eval.py`.

## Requirements

Ensure that all required libraries and dependencies are installed as specified in the main project README (if available).