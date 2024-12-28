# ECG Classification
This repository contains code for training and optimizing deep learning models (CNN and RNN) to classify ECG signals into different arrhythmia categories using the [MIT-BIH Arrhythmia Dataset](https://www.physionet.org/physiobank/database/mitdb/) . The project implements hyperparameter optimization using Weights & Biases (Wandb) for better model tuning.

Arrhythmia is a unique type of heart disease which produces inefficient and irregular heartbeat. This is a cardiac disease which is diagnosed through electrocardiogram (ECG) procedure. 

## Table of contents
- [ECG Classification](#ecg-classification)
  - [Table of contents](#table-of-contents)
  - [Dataset](#dataset)
  - [Installation](#installation)


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