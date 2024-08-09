""" 
The WandCallback give problems, infact in main_wandb3.py I changed it to WandbMetricsLogger()
It is really strange because time ago I used WandCallback and it worked perfectly.
I used it to not save the model locally and to log the metrics to wandb.
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from keras import activations
from absl import app, flags
import wandb
from wandb.integration.keras import WandbCallback # from wandb.integration.keras import WandbCallback
import sys

from utils.utils import get_data, plot_history, show_confusion_matrix
from models.CNN import f_build_CNN

# Define hyperparameters as flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('epochs', 15, 'Number of epochs to train the model')
flags.DEFINE_integer('batch_size', 250, 'Batch size for training')
flags.DEFINE_float('learning_rate', 0.001, 'Learning rate for the optimizer')

def main(argv):  # Main function to encapsulate the training logic
    # Parse flags
    FLAGS(argv) #argv = FLAGS(sys.argv) # FLAGS(argv)

    # Get the training and testing data
    train_df, test_df = get_data()

    # %% DATA PREPROCESSING
    # Normalization
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

    # %% MODEL TRAINING

    wandb.login()
    
    import tempfile
    # use a temporary directory for any wandb file usage
    wandb_dir = tempfile.mkdtemp()
    
    wandb.init(
        project="ECG-CNN",
        config={
            "epochs": FLAGS.epochs,
            "batch_size": FLAGS.batch_size,
            "architecture": "CNN",
            "dataset": "MIT-BIH",
            "optimizer": "Adam",
            "learning_rate": FLAGS.learning_rate
        },
        dir=wandb_dir,  # Use the temp directory for any wandb files
        settings=wandb.Settings(_disable_stats=True)  # Disable local logging stats #(start_method='fork')# Use 'fork' to avoid issues with multiprocessing
    )

    # Build and compile the model
    model = f_build_CNN((x_train.shape[1], 1), len(class_labels))
    model.summary()

    optimizer = tf.keras.optimizers.Adam(learning_rate=wandb.config.learning_rate)
    batch_size = wandb.config.batch_size

    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',  # When the labels are integers
        metrics=['accuracy']
    )

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )
    
    # Wandb callback for automatic logging without saving models locally
    wandb_callback = WandbCallback(
    save_model=False,  # Do not save the model locally
    log_weights=False, # Disable logging weights if not needed
    log_gradients=False, # Disable logging gradients if not needed
    save_graph=False,  # Do not save model graph locally
    #dir=wandb_dir      # Ensure that any logs are directed to the temp directory
    )
    
    # Train the model
    history = model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        validation_data=(x_test, y_test),
        epochs=wandb.config.epochs,
        callbacks=[early_stopping, wandb_callback] 
    )

    # Plot history
    plot_history(history, metric='accuracy')

    # Evaluation of the model
    test_loss, test_accuracy = model.evaluate(x_test, y_test)
    print('Test loss: ', test_loss)
    print('Test accuracy: ', test_accuracy)

    # Confusion Matrix
    y_test_conf_pred = model.predict(x_test)  # Predicted probabilities
    y_test_pred = np.argsort(y_test_conf_pred, axis=1)[:, -1]  # Predicted classes
    conf_matrix = confusion_matrix(y_test, y_test_pred, normalize='true')
    show_confusion_matrix(conf_matrix, class_names)
    
    # Log metrics to wandb
    wandb.log({
        "test_loss": test_loss,
        "test_accuracy": test_accuracy
    })

    wandb.finish()

if __name__ == '__main__':
    app.run(main)  # Run the main function
