""" 
I have changed get_data with load_data...
"""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from utils.utils import load_data, preprocess_for_hyperparameter, plot_interactive_idx

x_train, y_train, x_test, y_test, class_labels, class_names = load_data()

x_train, y_train, x_val, y_val, x_test, y_test = preprocess_for_hyperparameter(x_train, y_train, x_test, y_test)

# Interactive exploration of the training data
# plot_interactive_idx(x_train, y_train)  

# %% CLASS DISTRIBUTION

class_labels, class_counts = np.unique(y_train, return_counts=True) # unique values in the target column [0., 1., 2., 3., 4.]
print(f"Number of samples in each class: {dict(zip(class_labels, class_counts))}")
total = sum(class_counts)
print(f"Total number of samples in training set: {total}")
print(f"Examples:\n 0: {class_counts[0]} ({class_counts[0]/total*100:.2f}%)")
print(f" 1: {class_counts[1]} ({class_counts[1]/total*100:.2f}%)")
print(f" 2: {class_counts[2]} ({class_counts[2]/total*100:.2f}%)")
print(f" 3: {class_counts[3]} ({class_counts[3]/total*100:.2f}%)")
print(f" 4: {class_counts[4]} ({class_counts[4]/total*100:.2f}%)")
# I was counting the number of samples in each class, because I wanted to manually compute the weights, fortunately I found a function that does this for me
# I could do this using a for cycle..       

## Assign meaningful names to the class labels based on domain knowledge
class_names = {
    0: "Normal Beats",
    1: "Supraventricular Ectopy Beats",
    2: "Ventricular Ectopy Beats",
    3: "Fusion Beats",
    4: "Unclassifiable Beats"
}

## Print class labels with their assigned names
for label in class_labels:
    print(f"Class label {label}: {class_names[label]}")

# %% visualization of the distribution of the classes in the training, validation, and test sets

import seaborn as sns
def plot_distribution_sets(y_train, y_val, y_test): 
    # Combine train, validation, and test labels into a single DataFrame
    train_labels = pd.DataFrame({'Label': y_train, 'Dataset': 'Train'}) # 65662 rows × 2 columns
    val_labels = pd.DataFrame({'Label': y_val, 'Dataset': 'Validation'})
    test_labels = pd.DataFrame({'Label': y_test, 'Dataset': 'Test'})

    # Concatenate all DataFrames
    combined_labels = pd.concat([train_labels, val_labels, test_labels], ignore_index=True) # 109.446 rows × 2 columns

    # Plot the class distribution for all sets with normalization
    plt.figure(figsize=(12, 8))
    sns.histplot(
        data=combined_labels,
        x='Label',
        hue='Dataset',
        multiple='dodge',  # Separate bars for each dataset
        stat='percent',    # Normalize to percentages
        common_norm=False, # Normalize within each subset
        palette='viridis',
        shrink=0.8
    )
    plt.title('Normalized Label Distribution in Training, Validation, and Test Data')
    plt.xlabel('Label')
    plt.ylabel('Percentage')
    plt.legend(title='Dataset')
    plt.xticks(np.unique(y_train)) # Set x-ticks to class labels
    plt.grid(True)
    plt.show()

plot_distribution_sets(y_train, y_val, y_test)