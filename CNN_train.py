import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import tensorflow as tf
from utils.utils import load_data, class_weights, show_confusion_matrix, plot_history, plot_roc_multiclass
from models.CNN import build_CNN

x_train, y_train, x_test, y_test, class_labels, class_names = load_data()

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
    'epochs': 1 # I've setted to 1 just to compare the result with the other models #20 # I've setted to 5 just to see the performance of the model
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

optimizer = tf.keras.optimizers.Adam(learning_rate=best_hyperparameters['learning_rate'])

model.compile(
    optimizer=optimizer,
    loss='sparse_categorical_crossentropy', # When the labels are integers
    metrics= ['accuracy']
)

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    'best_model.keras',
    monitor='val_accuracy',
    save_best_only=True
)

patience = 5
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=patience,
    restore_best_weights=True
)

# Train the model
history = model.fit(
    x_train,
    y_train,
    batch_size=best_hyperparameters['batch_size'],
    epochs=best_hyperparameters['epochs'],
    validation_data=(x_test, y_test),
    callbacks=[early_stopping, checkpoint],
    class_weight=class_weights_dict
)

plot_history(history, metric='accuracy')

# %% eval.py

## Confusion Matrix
y_test_conf_pred = model.predict(x_test) # Predicted probabilities
y_test_pred = np.argsort(y_test_conf_pred,axis=1)[:,-1] # Predicted classes, the one with the highest probability
conf_matrix = confusion_matrix(y_test, y_test_pred, normalize='true')
show_confusion_matrix(conf_matrix, class_names)

# ROC curve for multiclass classification
num_classes = len(np.unique(y_test))
y_test_conf_pred_probs = model.predict(x_test)  # Get probability scores for all classes
plot_roc_multiclass("Model ROC", y_test, y_test_conf_pred_probs, num_classes)

## Metrics
from sklearn.metrics import classification_report
CNN_report = classification_report(
    y_test, 
    y_test_pred,
    labels=[0,1,2,3,4],
    target_names=['Normal Beats',"Supraventricular Ectopy Beats","Ventricular Ectopy Beats","Fusion Beats","Unclassifiable Beats"],
    output_dict=True)

import seaborn as sns
import pandas as pd

plt.figure(figsize=(8,8))
# Convert classification report to DataFrame
clf_report_df = pd.DataFrame(CNN_report).iloc[:-1, :].T # .iloc[:-1, :] to exclude support
ax = sns.heatmap(clf_report_df, annot=True, cmap='Blues')
ax.set_yticklabels(ax.get_yticklabels(),fontsize=12, rotation=0)
plt.title("Classification Report")

# %% Code to save the model 
model.save('CNN_model.h5')

# Load the model
from tensorflow.keras.models import load_model
model = load_model('CNN_model.h5')