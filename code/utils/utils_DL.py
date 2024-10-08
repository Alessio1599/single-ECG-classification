import numpy as np
import tensorflow as tf
from tensorflow.keras.metrics import Recall, Precision, AUC

# Add the following line to import the entire keras module
import tensorflow.keras as keras
from sklearn.metrics import confusion_matrix

from utils.utils import plot_history, show_confusion_matrix

def train_model(x_train, y_train, x_test, y_test, model, class_weights_dict=None):
    learning_rate = 0.001
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    batch_size = 256 # 64, 128

    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy', # When the labels are integers
        metrics=['accuracy']
                #keras.metrics.Recall(name='recall'), #Recall()
                #keras.metrics.Precision(name='precision'),
                #keras.metrics.AUC(name='auc')] 
    )

    epochs = 30 #50
    patience = 5
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=patience,
        restore_best_weights=True)

    history = model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        validation_data=(x_test, y_test),
        epochs=epochs,
        callbacks=[early_stopping]
        #class_weight=class_weights_dict
    )

    plot_history(history, metric='accuracy')
    return model, history


def evaluate_model(x_test, y_test, model, class_names):
    test_loss, test_accuracy, test_recall, test_precision, test_auc = model.evaluate(x_test, y_test)
    print('Test loss: ', test_loss)
    print('Test accuracy: ', test_accuracy)
    print('Test recall:', test_recall)
    print('Test precision:', test_precision)
    print('Test AUC:', test_auc)

    ## Confusion Matrix
    y_test_conf_pred = model.predict(x_test) # Predicted probabilities
    y_test_pred = np.argsort(y_test_conf_pred,axis=1)[:,-1] # Predicted classes, the one with the highest probability
    conf_matrix = confusion_matrix(y_test, y_test_pred, normalize='true')
    show_confusion_matrix(conf_matrix, class_names)

