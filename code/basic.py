# Here I will start from the beginning, I will avoid to use functions that I defined!

import os

# Directories
code_dir = os.path.dirname(__file__)
base_dir = os.path.dirname(code_dir)
data_dir = os.path.join(base_dir, 'data')

# Load the data
import pandas as pd

train_df = pd.read_csv(os.path.join(data_dir, 'mitbih_train.csv'), header=None)
test_df = pd.read_csv(os.path.join(data_dir, 'mitbih_test.csv'), header=None)

print('Train dataset shape:',train_df.shape) #(87554, 188)
print('Test dataset shape:',test_df.shape) #(21892, 188)

# Extract the features and labels
x_train = train_df.iloc[:, :-1].values # All the column except the last one 
y_train = train_df.iloc[:, -1].values # The last column

x_test = test_df.iloc[:, :-1].values
y_test = test_df.iloc[:, -1].values

print('x_train:',x_train)
print('y_train:',y_train)
print('x_train shape:',x_train.shape)
print('y_train shape:',y_train.shape)



# Preprocess the data (normalization) In this case I will not split the data into train and validation, since I will directly train the models
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler().fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


# Computation of the class weights
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

class_weights = compute_class_weight(
    class_weight='balanced', 
    classes=np.unique(y_train), 
    y=y_train
)

print('Class weights:', class_weights)
# Create a dictionary mapping class indices to weights
class_weights_dict = dict(enumerate(class_weights))


# Build the model
from models.CNN import build_CNN
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# Import best hyperparameters
best_hyperparameters = {
    'layer_1_size': 64,
    'layer_2_size': 128,
    'layer_3_size': 64,
    'layer_FC_size': 96,
    'dropout_rate': 0.1,
    'learning_rate': 4e-4,
}

input_shape = (x_train.shape[1], 1)
output_shape = len(np.unique(y_train))
print('input_shape:', input_shape)
print('output_shape:', output_shape)

model = build_CNN(
    input_shape=input_shape,
    output_shape=output_shape,
    layer_1_size=best_hyperparameters['layer_1_size'],
    layer_2_size=best_hyperparameters['layer_2_size'],
    layer_3_size=best_hyperparameters['layer_3_size'],
    layer_FC_size=best_hyperparameters['layer_FC_size'],
    dropout_rate=best_hyperparameters['dropout_rate']
)

model.summary()


learning_rate = best_hyperparameters['learning_rate']
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
model.compile(
    optimizer=optimizer,
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

epochs = 40
patience = 7
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=patience,
    restore_best_weights=True
)

# Better exponential decay
def exp_decay(epoch, lr):
    if epoch <= 5:
        return lr
    else:
        return lr * np.exp(-0.1)

# I can check how is the decay
lr = 0.01
lr_values = []
for i in range(20):
    lr = exp_decay(i, lr)
    lr_values.append(lr)

plt.figure()
plt.plot(lr_values)
plt.show()

lr_exp_decay = tf.keras.callbacks.LearningRateScheduler(exp_decay, verbose=1)

checkpoint_path = os.path.join(code_dir, 'models', 'cnn', 'cnn_model.keras')
chekpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    save_best_only=True,
    monitor='val_loss', # 'val_accuracy'
    mode='min'
)

batch_size = 256
history = model.fit(
    x_train,
    y_train,
    validation_data=(x_test, y_test),
    batch_size= batch_size,
    epochs=epochs,
    class_weight=class_weights_dict,
    callbacks=[early_stopping, lr_exp_decay, chekpoint]
)


from util import plot_history
plot_history(history, metric="accuracy")
