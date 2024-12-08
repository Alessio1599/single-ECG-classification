# ECG Classification
This repository contains code for training and optimizing deep learning models (CNN and RNN) to classify ECG signals into different arrhythmia categories using the [MIT-BIH Arrhythmia Dataset](https://www.physionet.org/physiobank/database/mitdb/) . The project implements hyperparameter optimization using Weights & Biases (Wandb) for better model tuning.

Arrhythmia is a unique type of heart disease which produces inefficient and irregular heartbeat. This is a cardiac disease which is diagnosed through electrocardiogram (ECG) procedure. 

## Table of contents
- [ECG Classification](#ecg-classification)
  - [Table of contents](#table-of-contents)
  - [Project structure](#project-structure)
  - [Dataset](#dataset)
  - [Installation](#installation)

## Project structure
The project directory is organized as follows:
```
single-ECG-Classification/
├── README.md
├── configs
│   ├── README.md
│   ├── config_cnn.yaml
│   ├── sweep-grid.yaml
│   ├── sweep_cnn_config_bayes.yaml
│   └── sweep_rnn_config_bayes.yaml
├── img
│   └── class_distribution_train.png
├── models
│   ├── CNN.py
│   ├── RNN.py
│   ├── __pycache__
│   │   ├── CNN.cpython-311.pyc
│   │   └── RNN.cpython-311.pyc
│   ├── cnn
│   │   ├── best_cnn_model_v1.keras
│   │   ├── best_cnn_model_v1_script.keras
│   │   ├── cnn_best_model.keras
│   │   ├── cnn_model.keras
│   │   └── model.keras
│   └── rnn
│       └── best_RNN_model_v1.keras
├── requirements.txt
├── results
│   ├── LSTM_classification_report.csv
│   ├── README.md
│   ├── cnn
│   │   ├── cnn_classification_report.csv
│   │   ├── cnn_classification_report.png
│   │   ├── cnn_confusion_matrix.png
│   │   └── cnn_confusion_matrix_barplot.png
│   ├── comparison
│   │   └── f1-score_comparison_models.png
│   ├── hyperparameter_tuning
│   │   ├── W&B Chart 07_08_2024, 20_25_52.png
│   │   ├── W&B Chart 29_10_2024, 12_52_19_.png
│   │   └── cnn
│   │       └── W&B Chart 08_12_2024, 23_15_24.png
│   ├── improved_rnn
│   │   ├── README.md
│   │   ├── rnn_classification_report.csv
│   │   ├── rnn_classification_report.png
│   │   └── rnn_confusion_matrix.png
│   ├── reports
│   └── rnn
│       ├── rnn_classification_report.csv
│       ├── rnn_classification_report.png
│       └── rnn_confusion_matrix.png
└── scripts
    ├── README.md
    ├── __pycache__
    │   └── util.cpython-311.pyc
    ├── data_preprocessing.py
    ├── eval.py
    ├── hyperparameter tuning
    │   ├── CNN_hyperparameter_optimization.py
    │   ├── README.md
    │   └── RNN_hyperparameter_optimization.py
    ├── train_cnn.py
    ├── train_rnn.py
    └── util.py
```

## Dataset 
[The MIT-BIH Arrhythmia Dataset](https://www.physionet.org/physiobank/database/mitdb/) is used for training and testing the models. It includes the following classes:
- 0: "Normal Beats",
- 1: "Supraventricular Ectopy Beats",
- 2: "Ventricular Ectopy Beats",
- 3: "Fusion Beats",
- 4: "Unclassifiable Beats"

## Installation

Clone the repository
```sh
git clone git@github.com:alessio1599/single-ECG-classification.git
```

To set up the environment, install the required dependencies using:

```sh
pip install -r requirements.txt
```