# 1. ECG Classification
Arrhythmia is a unique type of heart disease which produces inefficient and irregular heartbeat. This is a cardiac disease which is diagnosed through electrocardiogram (ECG) procedure. 

## Comments
- I can try to perform transfer learning. I saw a good paper, 1D ECG signals, maybe I can use ResNet

## 1.4. Directory structure

```
single-ECG-Classification/
├── CNN_hyperparameter_optimization.py
├── CNN_train.py
├── README.md
├── RNN_train.py
├── RNN_train_2.py
├── css
│   └── markdown-style.css
├── data
│   ├── mitbih_test.csv
│   ├── mitbih_train.csv
│   ├── ptbdb_abnormal.csv
│   └── ptbdb_normal.csv
├── data_exploratory.py
├── main_anomaly.py
├── models
│   ├── CNN.py
│   └── RNN.py
├── requirements.txt
├── sweep
│   ├── cnn_sweep_config_bayes.yaml
│   ├── rnn_sweep_config_bayes.yaml
│   └── sweep-grid.yaml
├── test
│   └── test_utils.py
├── utils
│   ├── utils.py
│   └── utils_DL.py
└── wandb
    ├── main_wandb.py
    └── main_wandb2.py
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
