# ECG Classification
This repository contains code for training and optimizing deep learning models (CNN and RNN) to classify ECG signals into different arrhythmia categories using the [MIT-BIH Arrhythmia Dataset](https://www.physionet.org/physiobank/database/mitdb/) . The project implements hyperparameter optimization using Weights & Biases (Wandb) for better model tuning.

Arrhythmia is a unique type of heart disease which produces inefficient and irregular heartbeat. This is a cardiac disease which is diagnosed through electrocardiogram (ECG) procedure. 

## Table of contents
- [ECG Classification](#ecg-classification)
  - [Table of contents](#table-of-contents)
  - [Project structure](#project-structure)
  - [Dataset](#dataset)
  - [Installation](#installation)
  - [Models](#models)
  - [Hyperparameter Tuning](#hyperparameter-tuning)
    - [Documentation](#documentation)
  - [Links](#links)
  - [References](#references)

## Project structure
The project directory is organized as follows:
```
single-ECG-Classification/
├── CNN_hyperparameter_optimization.py  # Script for CNN hyperparameter optimization
├── CNN_train.py                        # Script for training CNN model
├── README.md                           # This README file
├── RNN_train.py                        # Script for training RNN model
├── RNN_train_2.py                      # Alternative RNN training script
├── css/
│   └── markdown-style.css              # Custom CSS for Markdown rendering
├── data/
│   ├── mitbih_test.csv                 # Test data from MIT-BIH Arrhythmia Dataset
│   └── mitbih_train.csv                # Train data from MIT-BIH Arrhythmia Dataset
├── data_exploratory.py                 # Script for data exploration
├── main_anomaly.py                     # Script for anomaly detection
├── models/
│   ├── CNN.py                          # CNN model definition
│   └── RNN.py                          # RNN model definition
├── requirements.txt                    # Required Python packages
├── sweep/
│   ├── cnn_sweep_config_bayes.yaml     # CNN sweep configuration (Bayesian)
│   ├── rnn_sweep_config_bayes.yaml     # RNN sweep configuration (Bayesian)
│   └── sweep-grid.yaml                 # General sweep configuration (grid search)
├── test/
│   └── test_utils.py                   # Unit tests for utility functions
├── utils/
│   ├── utils.py                        # Utility functions
│   └── utils_DL.py                     # Deep learning utility functions
└── wandb/
    ├── main_wandb.py                   # Wandb integration script
    └── main_wandb2.py                  # Alternative Wandb integration script
```

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

3. **Analyze Results:**
   - Use the visualization tools provided in the `wandb` directory to analyze the results of the sweeps.
   - Refer to the `main_wandb.py` and `main_wandb2.py` scripts for generating visualizations and reports.


### Documentation
Refer to the `docs` directory for detailed documentation, including the project proposal and final presentation.

## Links
- [Kaggle repository](https://www.kaggle.com/code/alessio1999/single-ecg-classification)
  
## References
1. [Classification on imbalanced data, Tensorflow](https://www.tensorflow.org/tutorials/structured_data/imbalanced_data#class_weights)
2. [Demystifying Neural Networks: Anomaly Detection with AutoEncoder](https://medium.com/@weidagang/demystifying-anomaly-detection-with-autoencoder-neural-networks-1e235840d879)
3. [Tune hyperparameter using sweeps](https://docs.wandb.ai/guides/sweeps)
