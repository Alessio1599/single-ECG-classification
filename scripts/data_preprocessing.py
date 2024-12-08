import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

scripts_dir = os.path.dirname(__file__)
base_dir = os.path.dirname(scripts_dir)
data_dir = os.path.join(base_dir, 'data')
raw_data_dir = os.path.join(data_dir, 'raw')
processed_data_dir = os.path.join(data_dir, 'processed')

train_df = pd.read_csv(os.path.join(raw_data_dir, 'mitbih_train.csv'), header=None)
test_df = pd.read_csv(os.path.join(raw_data_dir, 'mitbih_test.csv'), header=None)

# Extract the features and labels
x_train = train_df.iloc[:, :-1].values # All the column except the last one 
y_train = train_df.iloc[:, -1].values # The last column

x_test = test_df.iloc[:, :-1].values
y_test = test_df.iloc[:, -1].values

# Split the data into training, validation and test sets
x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=0.5, random_state=42, stratify=y_test)


# Standardize the data
scaler = StandardScaler().fit(x_train)
x_train = scaler.transform(x_train)
x_val = scaler.transform(x_val)
x_test = scaler.transform(x_test)

# Save the processed data as .npy files
np.save(os.path.join(processed_data_dir, 'x_train.npy'), x_train)
np.save(os.path.join(processed_data_dir, 'y_train.npy'), y_train)
np.save(os.path.join(processed_data_dir, 'x_val.npy'), x_val)
np.save(os.path.join(processed_data_dir, 'y_val.npy'), y_val)
np.save(os.path.join(processed_data_dir, 'x_test.npy'), x_test)
np.save(os.path.join(processed_data_dir, 'y_test.npy'), y_test)

print("Processed data saved successfully.")