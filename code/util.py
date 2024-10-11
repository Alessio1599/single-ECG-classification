import os
import numpy as np
import pandas as pd
from ipywidgets import interact, IntSlider, fixed
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight # compute_class_weight returns Array with class_weight_vect[i] the weight for i-th class.
import sklearn.metrics
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import label_binarize # label_binarize is used to convert labels to binary form

import tensorflow as tf
from sklearn.metrics import confusion_matrix


def load_data(base_dir):
    """
    Reads ECG data from CSV files and returns training and test sets, including class labels and names.

    Returns
    -------
    x_train : numpy array
        Training data
    y_train : numpy array
        Training labels
    x_test : numpy array
        Test data
    y_test : numpy array
        Test labels
    class_labels : numpy array
        Unique class labels
    class_names : dict
        Mapping from class indices to class names
    """
    
    # os.path.dirname(__file__) -> '/Users/alessioguarachi/Desktop/single-ECG-classification/code/utils'
    # os.path.dirname(os.path.dirname(__file__)) -> '/Users/alessioguarachi/Desktop/single-ECG-classification/code'
    # os.path.dirname(os.path.dirname(os.path.dirname(__file__))) -> '/Users/alessioguarachi/Desktop/single-ECG-classification'
    
    #base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

    data_dir = os.path.join(base_dir, 'data')

    train_df = pd.read_csv(os.path.join(data_dir, 'mitbih_train.csv'), header=None)
    test_df = pd.read_csv(os.path.join(data_dir, 'mitbih_test.csv'), header=None)

    x_train = train_df.iloc[:, :-1].values
    y_train = train_df.iloc[:, -1].values

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

    return x_train, y_train, x_test, y_test, class_labels, class_names



def load_data_RNN():
    """
    Reads ECG data from CSV files and returns the training and test sets.

    Returns
    -------
    train_data : numpy array
        Training data including labels
    test_data : numpy array
        Test data including labels
    """
    base_dir = os.path.dirname(os.path.dirname(__file__))
    data_dir = os.path.join(base_dir,'data') # 'ECG' to add of the utils.py file is in the ECG folde and not in the utils folder

    #file_paths = glob.glob(os.path.join(data_dir, '*.csv'))

    train_df = pd.read_csv(os.path.join(data_dir, 'mitbih_train.csv'), delimiter=',', header=None) #87554 rows × 188 columns
    test_df = pd.read_csv(os.path.join(data_dir, 'mitbih_test.csv'), delimiter=',', header=None) #21892 rows × 188 columns
    
    train_data = train_df.values
    test_data = test_df.values
    
    # train_data = pd.read_csv(os.path.join(data_dir, 'mitbih_train.csv'), header=None).values # in one row
    
    return train_data, test_data


def preprocess_for_hyperparameter(x_train, y_train, x_test, y_test):
    """Preprocess the data for hyperparameter optimization. 
    
    Parameters
    ----------
    x_train : numpy array, 
        training data
    y_train: numpy array, 
        training labels
    x_test: numpy array, 
        test data
    y_test: numpy array, 
        test labels
    
    Returns
    -------
    x_train : numpy array, 
        training data
    y_train : numpy array, 
        training labels
    x_test : numpy array, 
        test data
    y_test : numpy array, 
        test labels
    """

    # I want to specify the number of samples for the validation set
    val_size = 21892 # 20% of the total samples
    # random state is set to 42 for reproducibility
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=val_size, random_state=42, stratify=y_train) # stratify=y_train: This argument ensures that the training and validation datasets will have the same proportion of each class as y_train.
    
    # Normalization
    scaler = StandardScaler().fit(x_train)
    x_train = scaler.transform(x_train)
    x_val = scaler.transform(x_val)
    x_test = scaler.transform(x_test)
    return x_train, y_train, x_val, y_val, x_test, y_test
    

# Weighted Loss
def class_weights(y_train):
    """
    Calculates class weights to handle class imbalance.

    Parameters
    ----------
    y_train : numpy array
        Training labels

    Returns
    -------
    class_weights_dict : dict
        Mapping of class indices to weights
    """
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y = y_train
    )
    # print("Computed class weights:", class_weights) # array of 5 elements, not dictionary
    # class_weights[0]= 0.24, class_weights[1]= ... class_weights[4]= ...
    
    # Create a dictionary mapping class indices to weights
    class_weights_dict = dict(enumerate(class_weights)) # enumerate returns an iterator with index and value pairs like [(0, 0.24), (1, 0.5), (2, 0.75), (3, 1.0), (4, 1.25)] and then dictionary {0: 0.24, 1: 0.5, 2: 0.75, 3: 1.0, 4: 1.25}
    #print("Class weights:", class_weights_dict)
    return class_weights_dict


## Interactive exploration, can be used in Jupyter Notebook
def plot_interactive_idx(x, y):
    """
    Plots an ECG signal and its label interactively.

    Parameters
    ----------
    x : numpy array
        Data
    y : numpy array
        Labels
    """
    def plot_ECG(x, y, idx):
        """ 
        This function plots an ECG signal and its label.
        """
        plt.plot(x[idx])
        plt.title('ECG Signal, label = ' + str(y[idx]))
        plt.ylabel('Amplitude[mV]')
        plt.show()

    # Create a slider to explore the data
    start_idx = IntSlider(min=0, max=len(x)-1, step=1, description='Index') 
    interact(plot_ECG, x=fixed(x), y=fixed(y), idx=start_idx) 

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

def plot_history(history,metric=None):
    """
    Plots training and validation loss and an optional metric.

    Parameters
    ----------
    history : keras.callbacks.History
        History object returned by the `fit` method of a Keras model
    metric : str, optional
        Name of the metric to plot
    """
    fig, ax1 = plt.subplots(figsize=(6, 6)) #figsize=(10,8)

    epoch_count=len(history.history['loss'])

    line1,=ax1.plot(range(1,epoch_count+1),history.history['loss'],label='train_loss',color='orange')
    ax1.plot(range(1,epoch_count+1),history.history['val_loss'],label='val_loss',color = line1.get_color(), linestyle = '--')
    ax1.set_xlim([1,epoch_count])
    ax1.set_ylim([0, max(max(history.history['loss']),max(history.history['val_loss']))])
    ax1.set_ylabel('loss',color = line1.get_color())
    ax1.tick_params(axis='y', labelcolor=line1.get_color())
    ax1.set_xlabel('Epochs')
    _=ax1.legend(loc='lower left')

    if (metric!=None): 
        ax2 = ax1.twinx()
        line2,=ax2.plot(range(1,epoch_count+1),history.history[metric],label='train_'+metric)
        ax2.plot(range(1,epoch_count+1),history.history['val_'+metric],label='val_'+metric,color = line2.get_color(), linestyle = '--')
        ax2.set_ylim([0, max(max(history.history[metric]),max(history.history['val_'+metric]))])
        ax2.set_ylabel(metric,color=line2.get_color())
        ax2.tick_params(axis='y', labelcolor=line2.get_color())
        _=ax2.legend(loc='upper right')
    plt.show() #block=False

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
    
def show_confusion_matrix(conf_matrix, class_names, figsize=(10,10)):
    """
    Plots the confusion matrix.

    Parameters
    ----------
    conf_matrix : numpy array
        Confusion matrix
    class_names : list
        Class names
    figsize : tuple, optional
        Size of the figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    img = ax.matshow(conf_matrix)
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    plt.ylabel('Real')
    plt.xlabel('Predicted')
    #plt.colorbar()

    for i in range(len(class_names)):
        for j in range(len(class_names)):
            text = ax.text(j, i, '{0:.1%}'.format(conf_matrix[i, j]),
                           ha='center', va='center', color='w')
    plt.show() #block=False


def plot_roc_multiclass(name, labels, predictions, num_classes, **kwargs):
    """
    Plots the ROC curve for multiclass classification.

    Parameters
    ----------
    name : str
        Name for the plot title
    labels : numpy array
        True labels
    predictions : numpy array
        Predicted probabilities
    num_classes : int
        Number of classes
    kwargs : dict, optional
        Additional arguments for the plot function
    """
    labels_bin = label_binarize(labels, classes=range(num_classes))
    plt.figure(figsize=(12, 8))  # Create a new figure with a specific size

    # Define a color map for different classes
    colors = plt.cm.get_cmap('tab10', num_classes)  # Use colormap with distinct colors

    for i in range(num_classes):
        fp, tp, _ = sklearn.metrics.roc_curve(labels_bin[:, i], predictions[:, i])
        plt.plot(100*fp, 100*tp, label=f"Class {i}", color=colors(i), linestyle='-', linewidth=2, **kwargs)

    plt.xlabel('False positives [%]')
    plt.ylabel('True positives [%]')
    plt.xlim([-0.5, 100])
    plt.ylim([0, 100])
    plt.grid(True)
    ax = plt.gca()
    ax.set_aspect('equal')
    plt.legend(loc='best')  # Add a legend with best location
    plt.title(f'ROC Curve - {name}')  # Add a title to the plot
    plt.show(block=False)  # Show the plot