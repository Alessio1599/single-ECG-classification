
```
project-root/
│
├── data/
│   ├── raw/             # Raw ECG data
│   ├── processed/       # Processed data, ready for training
│   ├── interim/         # Intermediate data transformations
│   └── external/        # Data from external sources
│
├── notebooks/           # Jupyter notebooks for EDA, experiments, etc.
│
├── scripts/
│   ├── data_preprocessing.py  # Data preprocessing scripts
│   ├── train_cnn.py           # Script for training CNN model
│   ├── train_lstm.py          # Script for training LSTM model
│   ├── hyperparameter_tuning.py # Script for hyperparameter tuning with W&B
│   └── utils.py               # Utility functions
│
├── models/              # Trained models and checkpoints
│   ├── cnn/
│   ├── lstm/
│   └── best_model/      # Best performing model
│
├── results/             # Evaluation results, plots, metrics
│   ├── figures/
│   ├── logs/
│   └── reports/
│
├── configs/             # Configuration files for experiments
│   ├── cnn_config.yaml
│   ├── lstm_config.yaml
│   └── sweep_config.yaml
│
├── environment.yml      # Conda environment file
├── requirements.txt     # Python dependencies
├── README.md            # Project overview and instructions
└── .gitignore           # Git ignore file
```