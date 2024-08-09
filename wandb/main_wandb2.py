""" 
This is functioning, I changed WandCallback to WandbMetricsLogger()
"""
import pandas as pd
import numpy as np
import tensorflow as tf
import wandb
from wandb.integration.keras import WandbMetricsLogger
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from absl import app, flags
from utils.utils import get_data, plot_history, show_confusion_matrix
from models.CNN import f_build_CNN
import sys

# Define hyperparameters as flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('epochs', 5, 'Number of epochs to train the model')
flags.DEFINE_integer('batch_size', 250, 'Batch size for training')
flags.DEFINE_float('learning_rate', 0.001, 'Learning rate for the optimizer')

def main(argv):
    argv = FLAGS(sys.argv)
    
    # Load data
    train_df, test_df = get_data()

    # Separate ECG signals and labels
    x_train = train_df.iloc[:, :-1].values  # Numpy array
    y_train = train_df.iloc[:, -1].values   # Numpy array
    x_test = test_df.iloc[:, :-1].values 
    y_test = test_df.iloc[:, -1].values 

    class_labels = np.unique(y_train)

    class_names = {
        0: "Normal Beats",
        1: "Supraventricular Ectopy Beats",
        2: "Ventricular Ectopy Beats",
        3: "Fusion Beats",
        4: "Unclassifiable Beats"
    }

    # Normalization
    scaler = StandardScaler().fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)

    # Log into Wandb
    wandb.login()

    # Initialize wandb
    wandb.init(
        project="ECG-CNN",
        config={
            "epochs": FLAGS.epochs,
            "batch_size": FLAGS.batch_size,
            "architecture": "CNN",
            "dataset": "MIT-BIH",
            "optimizer": "Adam",
            "learning_rate": FLAGS.learning_rate
        }
    )

    # Build model
    model = f_build_CNN((x_train.shape[1], 1), len(class_labels))
    model.summary()

    learning_rate = wandb.config.learning_rate
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    batch_size = wandb.config.batch_size

    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',  # When the labels are integers
        metrics=['accuracy']
    )

    # Early stopping callback
    patience = 5
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=patience,
        restore_best_weights=True
    )

    # Train the model
    history = model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        validation_data=(x_test, y_test),
        epochs=wandb.config.epochs,
        callbacks=[early_stopping, WandbMetricsLogger()], #WandbMetricsLogger()
    )

    # Plot training history
    plot_history(history, metric='accuracy')

    # Evaluation of the model
    test_loss, test_accuracy = model.evaluate(x_test, y_test)
    print('Test loss: ', test_loss)
    print('Test accuracy: ', test_accuracy)

    # Confusion Matrix
    y_test_conf_pred = model.predict(x_test)  # Predicted probabilities
    y_test_pred = np.argmax(y_test_conf_pred, axis=1)  # Predicted classes, the one with the highest probability
    conf_matrix = confusion_matrix(y_test, y_test_pred, normalize='true')
    show_confusion_matrix(conf_matrix, class_names)

    wandb.finish()

if __name__ == '__main__':
    app.run(main)
