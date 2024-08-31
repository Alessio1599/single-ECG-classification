# ECG Classification
Arrhythmia is a unique type of heart disease which produces inefficient and irregular heartbeat. This is a cardiac disease which is diagnosed through electrocardiogram (ECG) procedure. 

# Project Overview

This repository contains code for training and optimizing deep learning models (CNN and RNN) to classify ECG signals into different arrhythmia categories using the [MIT-BIH Arrhythmia Dataset](https://www.physionet.org/physiobank/database/mitdb/) . The project implements hyperparameter optimization using Weights & Biases (Wandb) for better model tuning.

## Directory structure

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


## Dataset 
[The MIT-BIH Arrhythmia Dataset](https://www.physionet.org/physiobank/database/mitdb/) is used for training and testing the models. It includes the following classes:
- 0: "Normal Beats",
- 1: "Supraventricular Ectopy Beats",
- 2: "Ventricular Ectopy Beats",
- 3: "Fusion Beats",
- 4: "Unclassifiable Beats"

## Models

  
## References
1. [Classification on imbalanced data, Tensorflow](https://www.tensorflow.org/tutorials/structured_data/imbalanced_data#class_weights)
2. [Demystifying Neural Networks: Anomaly Detection with AutoEncoder](https://medium.com/@weidagang/demystifying-anomaly-detection-with-autoencoder-neural-networks-1e235840d879)
