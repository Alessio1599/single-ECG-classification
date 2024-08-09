""" 
Modified version of the main.py file, creation of two models
Here I will do the anomaly detection model and the classification model
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.preprocessing import StandardScaler
#from seaborn import countplot
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from keras import activations

from utils.utils import load_data, plot_history, show_confusion_matrix
from models.CNN import build_CNN_for

x_train, y_train, x_test, y_test = load_data()

# %% DATA PREPROCESSING

class_labels = np.unique(y_train)

class_names = {
    0: "Normal Beats",
    1: "Supraventricular Ectopy Beats",
    2: "Ventricular Ectopy Beats",
    3: "Fusion Beats",
    4: "Unclassifiable Beats"
}

## Normalization
scaler = StandardScaler().fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# %% MODEL TRAINING

# In order to train a model for the anomaly detection I have to merge the classes 1,2,3,4 into a single class, for example 1
# I have to change the labels of the test set as well
y_train_anomaly = y_train.copy()
y_train_anomaly[y_train_anomaly != 0] = 1

y_test_anomaly = y_test.copy()
y_test_anomaly[y_test_anomaly != 0] = 1

class_labels_anomaly = np.unique(y_train_anomaly)
class_names_anomaly = {
    0: "Normal Beats",
    1: "Supraventricular Ectopy Beats"
}

#model = f_build_CNN((x_train.shape[1], 1), len(class_labels))
anomaly_detection = build_CNN_for((x_train.shape[1],1), len(class_labels_anomaly), [32,32])

anomaly_detection.summary()

learning_rate = 0.0001 #0.001
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
batch_size = 256 # 64, 128

anomaly_detection.compile(
    optimizer= optimizer,
    loss='sparse_categorical_crossentropy', # When the labels are integers
    metrics=['accuracy']
)

epochs = 30 #50
patience = 5
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=patience,
    restore_best_weights=True)

history = anomaly_detection.fit(
    x_train,
    y_train_anomaly,
    batch_size=batch_size,
    validation_data=(x_test, y_test_anomaly),
    epochs=epochs,
    callbacks=[early_stopping]
)

plot_history(history,metric='accuracy')

# Evaluation of the anomaly_detection
test_loss, test_accuracy = anomaly_detection.evaluate(x_test, y_test_anomaly)
print('Test loss: ', test_loss)
print('Test accuracy: ', test_accuracy)

## Confusion Matrix
y_test_conf_pred = anomaly_detection.predict(x_test) # Predicted probabilities
y_test_pred = np.argsort(y_test_conf_pred,axis=1)[:,-1] # Predicted classes, the one with the highest probability
conf_matrix = confusion_matrix(y_test, y_test_pred, normalize='true')
show_confusion_matrix(conf_matrix, class_names_anomaly)