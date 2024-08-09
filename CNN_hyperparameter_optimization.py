""" 
HYPERPARAMETER OPTIMIZATION -> choose the best hyperparameters for the model
Then I will train the model with the best hyperparameters on the whole training set(training+validation) and evaluate it on the test set.

I've created also train.py and eval.py.  Then I deleted them. and I've created utils_DL.py.
Because the main was too long and I wanted to separate the training and evaluation process.

- I should improve the loss or the accuracy of the model. I should consider certain metrics to evaluate the model.

TO-DO:
- I will move all the prints in the jupyter notebook
- Check the best practices for the hypepramaeter optimization
- Adding (block=False) I cannot see the plots at all

Babies Comment:
- I can use a lower number of epoch for the hyperparameter optimization. I will use a high number of epochs for the final training.
- Pay attention to the commas!!! 

"""
import matplotlib.pyplot as plt
import numpy as np
#from seaborn import countplot
import tensorflow as tf
from tensorflow import keras
import wandb

from utils.utils import load_data, preprocess_for_hyperparameter, class_weights ,plot_history, show_confusion_matrix
#from utils.utils_DL import train_model, evaluate_model
from models.CNN import build_CNN

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# %% DATA PREPROCESSING

## Load the data
x_train, y_train, x_test, y_test= load_data()

# Preprocess the data for hyperparameter optimization
x_train, y_train, x_val, y_val, x_test, y_test, class_labels, class_names = preprocess_for_hyperparameter(x_train, y_train, x_test, y_test)

# Get the class weights
class_weights_dict = class_weights(y_train)
# %% MODEL TRAINING

# Default hyperparameters
defaults = dict(
    layer_1_size = 32,
    layer_2_size = 64,
    layer_3_size = 32,
    layer_FC_size=100,
    dropout_rate = 0.2,
    learning_rate = 0.001,
    batch_size = 128,
    epochs = 30,
)

wandb.login()
# Add requirement for wandb core
wandb.require("core") #wandb.require("service")  # Ensures correct Wandb version

wandb.init(config=defaults) #wandb.init(project="single-ECG-classification", entity="neuroeng")
config = wandb.config

input_shape = (x_train.shape[1], 1) # (187, 1)
output_shape = len(np.unique(y_train))
model = build_CNN(
    input_shape, 
    output_shape, 
    layer_1_size=config.layer_1_size, 
    layer_2_size=config.layer_2_size,
    layer_3_size=config.layer_3_size,
    layer_FC_size=config.layer_FC_size,
    dropout_rate=config.dropout_rate
)

optimizer = tf.keras.optimizers.Adam(learning_rate=config.learning_rate)

model.compile(
    optimizer=optimizer,
    loss='sparse_categorical_crossentropy', # When the labels are integers
    metrics= ['accuracy']
)

# Implement a learning rate scheduler
lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(  # Reduce learning rate when a metric has stopped improving.
    monitor='val_loss', 
    factor=0.5, 
    patience=2, # Number of epochs with no improvement after which learning rate will be reduced. #2
    min_lr=1e-6,
    verbose=1
)

patience = 5
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=patience,
    restore_best_weights=True
)

# # Custom WandbMetricsLogger (Idk if it is necessary, if it is useful or not)
class WandbMetricsLogger(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        wandb.log(logs)

#from wandb.integration.keras import WandbMetricsLogger #WandbModelCheckpoint
history = model.fit(
    x_train,
    y_train,
    batch_size=config.batch_size,
    validation_data=(x_val, y_val),
    epochs=config.epochs,
    callbacks=[early_stopping, WandbMetricsLogger()], 
    class_weight=class_weights_dict
)

plot_history(history, metric='accuracy')

"""Command line 
wandb sweep config.yaml
Since it is inside a folder wandb sweep sweep/sweep-bayes.yaml
wandb agent neuroeng/single-ECG-classification/c0bmfji5 --count 4
"""