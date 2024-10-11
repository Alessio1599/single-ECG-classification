import numpy as np
import tensorflow as tf

import sys
import os
code_dir = os.path.dirname(__file__)
sys.path.append(code_dir)

from util import load_data, class_weights, train_model, evaluate_model
from models.CNN import build_CNN

base_dir = os.path.dirname(code_dir)
x_train, y_train, x_test, y_test, class_labels, class_names = load_data(base_dir)

# Get the class weights
class_weights_dict = class_weights(y_train)

# Import best hyperparameters, best_hyperparameters is a dictionary
best_hyperparameters = {
    'layer_1_size': 64,
    'layer_2_size': 64,
    'layer_3_size': 64,
    'layer_FC_size': 64,
    'dropout_rate': 0.4,
    'learning_rate': 4e-4,
    'batch_size': 32,
    'epochs': 15 # I've setted to 1 just to compare the result with the other models #20 # I've setted to 5 just to see the performance of the model
}

input_shape = (x_train.shape[1], 1) # (187, 1)
output_shape = len(np.unique(y_train))

model = build_CNN(
    input_shape=input_shape,
    output_shape=output_shape,
    layer_1_size=best_hyperparameters['layer_1_size'],
    layer_2_size=best_hyperparameters['layer_2_size'],
    layer_3_size=best_hyperparameters['layer_3_size'],
    layer_FC_size=best_hyperparameters['layer_FC_size'],
    dropout_rate=best_hyperparameters['dropout_rate']
)

model.summary()

model, history = train_model(x_train, y_train, x_test, y_test, model, class_weights_dict)

evaluate_model(x_test, y_test, model, history, class_names)
