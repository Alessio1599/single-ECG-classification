""" 
I can upload the two models CNN and RNN
I can create a function to generate the report for the two models
And then I can compare the two models using barplots

Maybe I can try to use a subplots for the different plots
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix
import tensorflow as tf
from utils.utils import load_data, show_confusion_matrix, plot_roc_multiclass
from utils.utils import load_data
from tensorflow.keras.models import load_model


model = load_model('models/CNN_model.keras') #CNN_model = load_model('models/CNN_model.keras')
#RNN_model = load_model('models/RNN_model.keras')

x_train, y_train, x_test, y_test, class_labels, class_names = load_data()
num_classes = len(np.unique(y_test))

y_test_conf_pred_probs = model.predict(x_test) # Predicted probabilities # Get probability scores for all classes
y_test_pred = np.argsort(y_test_conf_pred_probs,axis=1)[:,-1] # Predicted classes, the one with the highest probability

## Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_test_pred, normalize='true')
show_confusion_matrix(conf_matrix, class_names)

# ROC curve for multiclass classification
plot_roc_multiclass("Model ROC", y_test, y_test_conf_pred_probs, num_classes)

## Metrics
from sklearn.metrics import classification_report
CNN_report = classification_report(
    y_test, 
    y_test_pred,
    labels=[0,1,2,3,4],
    target_names=['Normal Beats',"Supraventricular Ectopy Beats","Ventricular Ectopy Beats","Fusion Beats","Unclassifiable Beats"],
    output_dict=True)

# Convert classification report to DataFrame
report_df = pd.DataFrame(CNN_report).transpose()

# Calculate the average metrics
average_metrics = {
    'F1-Score': report_df['f1-score'].mean(),
    'Precision': report_df['precision'].mean(),
    'Recall': report_df['recall'].mean()
}

import seaborn as sns
import pandas as pd

plt.figure(figsize=(8,8))
# Convert classification report to DataFrame
clf_report_df = pd.DataFrame(CNN_report).iloc[:-1, :].T # .iloc[:-1, :] to exclude support
ax = sns.heatmap(clf_report_df, annot=True, cmap='Blues')
ax.set_yticklabels(ax.get_yticklabels(),fontsize=12, rotation=0)
plt.title("Classification Report")
plt.show()