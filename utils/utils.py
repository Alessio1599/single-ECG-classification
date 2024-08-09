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


def load_data():
    """ 
    This function reads the ECG data from the CSV files and returns the training,and test sets, splitted into ECG signals and labels.
    Returns:
    x_train: numpy array, training data
    y_train: numpy array, training labels
    x_test: numpy array, test data
    y_test: numpy array, test labels
    """
    base_dir = os.path.dirname(os.path.dirname(__file__))
    data_dir = os.path.join(base_dir,'data') # 'ECG' to add of the utils.py file is in the ECG folde and not in the utils folder

    #file_paths = glob.glob(os.path.join(data_dir, '*.csv'))

    #nRowsRead = 1000 # specify 'None' if want to read whole file
    train_df = pd.read_csv(os.path.join(data_dir, 'mitbih_train.csv'), delimiter=',', header=None) #87554 rows × 188 columns
    test_df = pd.read_csv(os.path.join(data_dir, 'mitbih_test.csv'), delimiter=',', header=None) #21892 rows × 188 columns
    3
    ## Separate ECG signals and labels (training and test)
    x_train = train_df.iloc[:, :-1].values # Numpy array # (87554, 187)
    y_train = train_df.iloc[:, -1].values # Numpy array  # (87554,)

    x_test = test_df.iloc[:, :-1].values 
    y_test = test_df.iloc[:, -1].values 

    return x_train, y_train, x_test, y_test


def load_data_RNN():
    """ 
    This function reads the ECG data from the CSV files and returns the training,and test sets, splitted into ECG signals and labels.
    Returns:
    x_train: numpy array, training data
    y_train: numpy array, training labels
    x_test: numpy array, test data
    y_test: numpy array, test labels
    """
    base_dir = os.path.dirname(os.path.dirname(__file__))
    data_dir = os.path.join(base_dir,'data') # 'ECG' to add of the utils.py file is in the ECG folde and not in the utils folder

    #file_paths = glob.glob(os.path.join(data_dir, '*.csv'))

    #nRowsRead = 1000 # specify 'None' if want to read whole file
    train_df = pd.read_csv(os.path.join(data_dir, 'mitbih_train.csv'), delimiter=',', header=None) #87554 rows × 188 columns
    test_df = pd.read_csv(os.path.join(data_dir, 'mitbih_test.csv'), delimiter=',', header=None) #21892 rows × 188 columns
    3
    
    train_data = train_df.values
    test_data = test_df.values
    
    return train_data, test_data


def preprocess_for_hyperparameter(x_train, y_train, x_test, y_test):
    """ 
    Args:
    x_train: numpy array, training data
    y_train: numpy array, training labels
    x_test: numpy array, test data
    y_test: numpy array, test labels
    
    Returns:
    x_train: numpy array, training data
    y_train: numpy array, training labels
    x_test: numpy array, test data
    y_test: numpy array, test labels
    class_labels: numpy array, unique class labels
    class_names: dictionary, mapping class indices to class names
    """
    class_labels = np.unique(y_train) # Get the unique class labels 0, 1, 2, 3, 4

    class_names = {
        0: "Normal Beats",
        1: "Supraventricular Ectopy Beats",
        2: "Ventricular Ectopy Beats",
        3: "Fusion Beats",
        4: "Unclassifiable Beats"
    }
    
    # I want to specify the number of samples for the validation set
    val_size = 21892 # 20% of the total samples
    # random state is set to 42 for reproducibility
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=val_size, random_state=42, stratify=y_train) # stratify=y_train: This argument ensures that the training and validation datasets will have the same proportion of each class as y_train.
    
    # Normalization
    scaler = StandardScaler().fit(x_train)
    x_train = scaler.transform(x_train)
    x_val = scaler.transform(x_val)
    x_test = scaler.transform(x_test)
    return x_train, y_train, x_val, y_val, x_test, y_test, class_labels, class_names
    

# Weighted Loss
def class_weights(y_train):
    """ 
    Calculate class weights to handle imbalance.
    Arg:
    y_train: array, training labels
    Returns:
    class_weights_dict: dictionary, mapping class indices to weights
    """
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y = y_train
    )
    print("Computed class weights:", class_weights) # array of 5 elements, not dictionary
    # class_weights[0]= 0.24, class_weights[1]= ... class_weights[4]= ...
    
    # Create a dictionary mapping class indices to weights
    class_weights_dict = dict(enumerate(class_weights)) # enumerate returns an iterator with index and value pairs like [(0, 0.24), (1, 0.5), (2, 0.75), (3, 1.0), (4, 1.25)] and then dictionary {0: 0.24, 1: 0.5, 2: 0.75, 3: 1.0, 4: 1.25}
    print("Class weights:", class_weights_dict)
    return class_weights_dict


## Interactive exploration, can be used in Jupyter Notebook
def plot_interactive_idx(x, y):
    """ 
    This function plots an ECG signal and its label interactively.
    Can be used to explore the data (training and test) interactively.
    Can be used in Jupyter Notebook.
    
    Args:
    x: numpy array, training data
    y: numpy array, training labels
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


def plot_history(history,metric=None):
    """ 
    This function plots the training and validation loss and an optional metric.
    
    Args:
    history: history object, output of the fit method of a keras model
    metric: string, optional, metric to plot
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
    
def show_confusion_matrix(conf_matrix, class_names, figsize=(10,10)):
    """ 
    This function plots the confusion matrix.
    Args:
    conf_matrix: numpy array, confusion matrix
    class_names: list, class names
    figsize: tuple, size of the figure
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
    Plot ROC curve for multiclass classification with colored lines only.
    Args:
    name: str, name of ..
    labels: array, true labels
    predictions: array, predicted probabilities
    num_classes: int, number of classes
    kwargs: dict, additional arguments for the plot function
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