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
└── requirements.txt
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