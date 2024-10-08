# Codes
## Folder structure
```
├── code
│   ├── data_exploratory.py
│   ├── eval.py
│   ├── hyperparameter_tuning
│   │   ├── CNN_hyperparameter_optimization.py
│   │   ├── README.md
│   │   ├── RNN_hyperparameter_optimization.py
│   │   └── sweep_conf
│   │       ├── cnn_sweep_config_bayes.yaml
│   │       ├── rnn_sweep_config_bayes.yaml
│   │       └── sweep-grid.yaml
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
- utils/: Utility scripts used throughout the project: utils.py and utils_DL.py: Contain helper functions and utilities for data processing and model training.