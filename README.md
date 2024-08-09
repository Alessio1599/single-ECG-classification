# 1. ECG Classification
Arrhythmia is a unique type of heart disease which produces inefficient and irregular heartbeat. This is a cardiac disease which is diagnosed through electrocardiogram (ECG) procedure. 

## Table of content
- [1. ECG Classification](#1-ecg-classification)
  - [Table of content](#table-of-content)
  - [ 1.1. Comments ](#-11-comments-)
  - [ 1.2. Problems ](#-12-problems-)
  - [1.3. Other things](#13-other-things)
  - [1.4. Directory structure](#14-directory-structure)
  - [1.5. Dataset](#15-dataset)
  - [1.6. Models](#16-models)
  - [1.7. References](#17-references)

## <font color="red"> 1.1. Comments </font>
In order to increase the performances of the model I can:
- add particular layers (Batch Normalization), see the literature and propose a new model (maybe taking inspiration from other problems)
- Since this dataset is highly umbalanced, so I can try to use a weighted loss function, or try to search other options online
- Metrics: Precision, Recall, F1 score

> Pay attention to the verdion of Tensorflow and Keras, maybe the versions are different in GoogleColab and so you have to do certain modifications...

## <font color="red"> 1.2. Problems </font>
I've tried to pass those metrics in the compile, in different way but nothing
- I've added the metrics in the ... function and I had problems
```pythonss
METRICS = [
      keras.metrics.Precision(name='precision'),
      keras.metrics.Recall(name='recall'),
      keras.metrics.AUC(name='auc'),
      keras.metrics.AUC(name='prc', curve='PR'), # precision-recall curve
]
```
## 1.3. Other things
[Property ‘model’ of ‘WandbCallback’ object has no setter](https://community.wandb.ai/t/property-model-of-wandbcallback-object-has-no-setter/6137)
 [Weights & Biases Keras Callbacks](https://docs.wandb.ai/guides/integrations/keras#the-weights--biases-keras-callbacks)
- I don't have to use WandbCallback() but WandbMetricsLogger 

**The Weights & Biases Keras Callbacks** 

We have added three new callbacks for Keras and TensorFlow users, available from wandb v0.13.4. For the legacy WandbCallback scroll down.

- WandbMetricsLogger : Use this callback for Experiment Tracking. It will log your training and validation metrics along with system metrics to Weights and Biases.
- WandbModelCheckpoint : Use this callback to log your model checkpoints to Weight and Biases Artifacts.
- WandbEvalCallback: This base callback will log model predictions to Weights and Biases Tables for interactive visualization.

## 1.4. Directory structure
```
ECG-Classification/
├── archive/
│   ├── mitbih_test.csv
│   └── mitbih_train.csv
├── .gitignore
├── LICENSE
├── main.py
├── README.md
└── utils.py
```
## 1.5. Dataset 
[The MIT-BIH Arrhythmia Dataset](https://www.physionet.org/physiobank/database/mitdb/) 


Classes: ['N': 0, 'S': 1, 'V': 2, 'F': 3, 'Q': 4]
- 0: "Normal Beats",
- 1: "Supraventricular Ectopy Beats",
- 2: "Ventricular Ectopy Beats",
- 3: "Fusion Beats",
- 4: "Unclassifiable Beats"

## 1.6. Models

**Hyperparameters**
- Learning_rate = 0.0001 - 0.001
- Number of hidden layers
- Regularization L1, L2? 
  
## 1.7. References
1. [Classification on imbalanced data, Tensorflow](https://www.tensorflow.org/tutorials/structured_data/imbalanced_data#class_weights)
2. [Demystifying Neural Networks: Anomaly Detection with AutoEncoder](https://medium.com/@weidagang/demystifying-anomaly-detection-with-autoencoder-neural-networks-1e235840d879)
