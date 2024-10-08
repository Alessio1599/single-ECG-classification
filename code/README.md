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


## Contents

- **data_exploratory.py**: Script for initial data exploration and visualization.
- **eval.py**: Script for evaluating the performance of trained models.
- **hyperparameter_tuning**: Directory containing scripts and configuration files for hyperparameter tuning.
  - **CNN_hyperparameter_optimization.py**: Script for optimizing CNN hyperparameters.
  - **RNN_hyperparameter_optimization.py**: Script for optimizing RNN hyperparameters.
  - **README.md**: Detailed description of hyperparameter tuning procedures.
  - **sweep_conf**: Configuration files for hyperparameter sweeps.
    - **cnn_sweep_config_bayes.yaml**: Bayesian sweep configuration for CNN.
    - **rnn_sweep_config_bayes.yaml**: Bayesian sweep configuration for RNN.
    - **sweep-grid.yaml**: Grid search configuration for hyperparameter tuning.
- models/: Contains model definitions and saved models:
  - CNN.py and RNN.py: Scripts defining the CNN and RNN architectures.
    - cnn/ and rnn/: Directories containing the trained model files.
    - best_cnn_model_v1.keras and best_RNN_model_v1.keras are the saved models for CNN and RNN respectively.
- training/: Scripts used for training the models:
  - CNN_train.py and RNN_train.py for training CNN and RNN models respectively.
- utils/: Utility scripts used throughout the project: 
  - utils.py and utils_DL.py: Contain helper functions and utilities for data processing and model training.