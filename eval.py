""" 
Comments:
I can upload the two models CNN and RNN
I can create a function to generate the report for the two models
And then I can compare the two models using barplots

Maybe I can try to use a subplots (to see confusion matrix, roc curve in a single plot for a single model) for the different plots
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import tensorflow as tf

from utils.utils import load_data, show_confusion_matrix, plot_roc_multiclass
from utils.utils import load_data
from tensorflow.keras.models import load_model

x_train, y_train, x_test, y_test, class_labels, class_names = load_data()

cnn_model = load_model('models/cnn/best_cnn_model_v1.keras') #CNN_model = load_model('models/CNN_model.keras')
rnn_model = load_model('models/rnn/best_rnn_model_v1.keras')

def evaluate_model(model, x_test, y_test, class_names):
    """
    Evaluates a model by generating and displaying:
    - Confusion matrix using the provided `show_confusion_matrix` function.
    - ROC curve for multiclass classification using the provided `plot_roc_multiclass` function.

    Parameters
    ----------
    model : tensorflow.keras.Model
        The trained Keras model to be evaluated.

    x_test : numpy.ndarray
        The test feature data.

    y_test : numpy.ndarray
        The true labels for the test data.

    class_names : list of str
        The names of the classes corresponding to the labels in `y_test`.
        
    Returns
    -------
    None
        This function does not return any value. It displays plots for the confusion matrix and ROC curve.
    """
    # Predict probabilities and classes
    y_test_conf_pred_probs = model.predict(x_test) # Predicted probabilities # Get probability scores for all classes
    y_test_pred = np.argsort(y_test_conf_pred_probs, axis=1)[:, -1]  # Predicted classes, the one with the highest probability

    # Confusion Matrix
    conf_matrix = confusion_matrix(y_test, y_test_pred, normalize='true')
    show_confusion_matrix(conf_matrix, class_names)

    # ROC Curve for multiclass classification
    num_classes = len(class_names)
    plot_roc_multiclass("Model ROC", y_test, y_test_conf_pred_probs, num_classes)

evaluate_model(cnn_model, x_test, y_test, class_names)
evaluate_model(rnn_model, x_test, y_test, class_names) # I have a problem with the RNN model



def generate_metrics_report(model, x_test, y_test, class_names):
    
    """
    Generates a classification report and macro average metrics for a given model.

    This function uses the provided model to predict the classes of the test data, calculates a 
    classification report, and extracts macro average metrics (Precision, Recall, F1-Score). 
    It provides a comprehensive evaluation of the model's performance across multiple classes.

    Parameters
    ----------
    model : tensorflow.keras.Model
        The trained Keras model to be evaluated. This model should have been compiled and trained 
        before using this function.

    x_test : numpy.ndarray
        The test feature data. The shape of this array should match the input shape expected by 
        the model.

    y_test : numpy.ndarray
        The true labels for the test data. These labels should correspond to the classes that the 
        model is expected to predict.

    class_names : list of str
        A list of class names that correspond to the labels in `y_test`. This list is used to 
        generate a human-readable classification report.

    Returns
    -------
    report_df : pandas.DataFrame
        A DataFrame containing the classification report. This includes metrics such as precision, 
        recall, and F1-score for each class, as well as macro and weighted averages.

    macro_average_metrics : dict
        A dictionary containing the macro average metrics for the model. The keys are:
        - 'Precision': The macro average precision score.
        - 'Recall': The macro average recall score.
        - 'F1-Score': The macro average F1-score.
    """
    
    # Predict
    y_test_conf_pred_probs = model.predict(x_test)
    y_test_pred = np.argsort(y_test_conf_pred_probs, axis=1)[:, -1]
    
    # Classification report
    report = classification_report(
        y_test, 
        y_test_pred,
        labels=list(range(len(class_names))),
        target_names=class_names,
        output_dict=True
    )
    
    # Convert classification report to DataFrame
    report_df = pd.DataFrame(report).transpose()
    
    # Calculate macro average metrics
    macro_average_metrics = {
        'Precision': report_df.loc['macro avg', 'precision'],
        'Recall': report_df.loc['macro avg', 'recall'],
        'F1-Score': report_df.loc['macro avg', 'f1-score'] # report_df.loc['macro avg', 'f1-score'] To access to the macro avg of the f1-score column
    }
    
    return report_df, macro_average_metrics


def classification_report_heatmap(report_df):
    """
    Plots a heatmap of the classification report.

    This function takes a DataFrame containing the classification report metrics (excluding the 'support' row) 
    and generates a heatmap to visualize the precision, recall, and F1-score for each class. The heatmap 
    helps to quickly assess the performance of the model across different classes.

    Parameters
    ----------
    report_df : pandas.DataFrame
        A DataFrame containing the classification report metrics. The DataFrame should include metrics such 
        as precision, recall, and F1-score for each class. It should be in a format where the index represents 
        class names and columns represent different metrics. The 'support' row should be excluded or not included.

    Returns
    -------
    None
        This function does not return any value. It displays the heatmap plot of the classification report.
    """
    plt.figure(figsize=(8,8))
    report_df = report_df.iloc[:-1, :].T # .iloc[:-1, :] to exclude support
    ax = sns.heatmap(report_df, annot=True, cmap='Blues')
    ax.set_yticklabels(ax.get_yticklabels(),fontsize=12, rotation=0)
    plt.title("Classification Report")
    plt.show()

# Generate metrics report
cnn_report_df, cnn_macro_avg = generate_metrics_report(cnn_model, x_test, y_test, class_names)
rnn_report_df, rnn_macro_avg = generate_metrics_report(rnn_model, x_test, y_test, class_names)

classification_report_heatmap(cnn_report_df)
classification_report_heatmap(rnn_report_df)

# Convert macro average metrics to DataFrame for plotting
metrics_df = pd.DataFrame({
    'Model': ['CNN', 'RNN'],
    'Precision': [cnn_macro_avg['Precision'], rnn_macro_avg['Precision']],
    'Recall': [cnn_macro_avg['Recall'], rnn_macro_avg['Recall']],
    'F1-Score': [cnn_macro_avg['F1-Score'], rnn_macro_avg['F1-Score']]
})

# Melt DataFrame to long format for easier plotting
metrics_melted = pd.melt(metrics_df, id_vars='Model', var_name='Metrics', value_name='Score')

sns.barplot(x='Metric', y='Score', hue='Model', data=metrics_melted, palette='viridis')
plt.title('Macro Average Metrics Comparison')
plt.show()