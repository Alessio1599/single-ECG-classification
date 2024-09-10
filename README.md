# ECG Classification
This repository contains code for training and optimizing deep learning models (CNN and RNN) to classify ECG signals into different arrhythmia categories using the [MIT-BIH Arrhythmia Dataset](https://www.physionet.org/physiobank/database/mitdb/) . The project implements hyperparameter optimization using Weights & Biases (Wandb) for better model tuning.

Arrhythmia is a unique type of heart disease which produces inefficient and irregular heartbeat. This is a cardiac disease which is diagnosed through electrocardiogram (ECG) procedure. 

## Table of contents
- [ECG Classification](#ecg-classification)
  - [Table of contents](#table-of-contents)
  - [Project structure](#project-structure)
    - [Description of Key Directories and Files](#description-of-key-directories-and-files)
  - [Dataset](#dataset)
  - [Installation](#installation)
  - [Models](#models)
  - [Hyperparameter Tuning](#hyperparameter-tuning)
  - [Links](#links)
  - [References](#references)

## Project structure
The project directory is organized as follows:
```
single-ECG-Classification/
├── README.md
├── data_exploratory.py
├── eval.py
├── hyperparameter tuning
│   ├── CNN_hyperparameter_optimization.py
│   ├── README.md
│   ├── RNN_hyperparameter_optimization.py
│   └── sweep_conf
│       ├── cnn_sweep_config_bayes.yaml
│       ├── rnn_sweep_config_bayes.yaml
│       └── sweep-grid.yaml
├── main.py
├── models
│   ├── CNN.py
│   ├── RNN.py
│   ├── cnn
│   │   └── best_cnn_model_v1.keras
│   └── rnn
│       └── best_RNN_model_v1.keras
├── requirements.txt
├── training
│   ├── CNN_train.py
│   └── RNN_train.py
└── utils
    ├── utils.py
    └── utils_DL.py
```

### Description of Key Directories and Files
- hyperparameter tuning/: Includes scripts and configuration files for hyperparameter optimization:
  - CNN_hyperparameter_optimization.py and RNN_hyperparameter_optimization.py for running the hyperparameter optimization.
  - sweep_conf/: Contains YAML configuration files for defining sweep parameters. 
    - cnn_sweep_config_bayes.yaml and rnn_sweep_config_bayes.yaml for CNN and RNN respectively.
    - sweep-grid.yaml for grid search configurations.
- models/: Contains model definitions and saved models:
  - CNN.py and RNN.py: Scripts defining the CNN and RNN architectures.
    - cnn/ and rnn/: Directories containing the trained model files.
    - best_cnn_model_v1.keras and best_RNN_model_v1.keras are the saved models for CNN and RNN respectively.
- training/: Scripts used for training the models:
  - CNN_train.py and RNN_train.py for training CNN and RNN models respectively.
- utils/: Utility scripts used throughout the project:
  - utils.py and utils_DL.py: Contain helper functions and utilities for data processing and model training.

## Dataset 
[The MIT-BIH Arrhythmia Dataset](https://www.physionet.org/physiobank/database/mitdb/) is used for training and testing the models. It includes the following classes:
- 0: "Normal Beats",
- 1: "Supraventricular Ectopy Beats",
- 2: "Ventricular Ectopy Beats",
- 3: "Fusion Beats",
- 4: "Unclassifiable Beats"

## Installation

To set up the environment, install the required dependencies using:

```sh
pip install -r requirements.txt
```

## Models

## Hyperparameter Tuning

To optimize the hyperparameters for the models, follow these steps:

1. **Setup Sweeps Configuration:**
   - Edit the sweep configuration files located in `hyperparameter tuning/sweep_conf/`.
   - Use the following configuration files for the respective models:
     - CNN: `cnn_sweep_config_bayes.yaml`
     - RNN: `rnn_sweep_config_bayes.yaml`

2. **Run Sweeps:**
   - Execute the sweep runs using the following scripts:
     - For CNN: `python hyperparameter tuning/CNN_hyperparameter_optimization.py`
     - For RNN: `python hyperparameter tuning/RNN_hyperparameter_optimization.py`
   - Perform at least 40-50 runs for each model to ensure a thorough search of the hyperparameter space.

## Links
- [Kaggle repository](https://www.kaggle.com/code/alessio1999/single-ecg-classification)
  
## References
1. [Classification on imbalanced data, Tensorflow](https://www.tensorflow.org/tutorials/structured_data/imbalanced_data#class_weights)
2. [Demystifying Neural Networks: Anomaly Detection with AutoEncoder](https://medium.com/@weidagang/demystifying-anomaly-detection-with-autoencoder-neural-networks-1e235840d879)
3. [Tune hyperparameter using sweeps](https://docs.wandb.ai/guides/sweeps)
