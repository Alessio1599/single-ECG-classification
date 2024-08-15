""" 
This code is functioning!!!
"""
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from utils.utils import load_data, class_weights, plot_history
from models.RNN import build_RNN, build_deep_rnn

x_train, y_train, x_test, y_test, class_labels, class_names = load_data()
print('x_train shape: ',x_train.shape) # (87554, 187)
print('y_train shape: ',y_train.shape) # (87554,)

# Get the class weights
class_weights_dict = class_weights(y_train)

# Normalization
scaler = StandardScaler().fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# For RNN models, the input data typically needs to be in the shape (samples, timesteps, features)
# Reshape x_train and x_test to add a time dimension of 1
# Idk if this part is necessary
x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

input_shape = (x_train.shape[1], 1) # (187, 1)
output_shape = len(np.unique(y_train)) # 

model = build_RNN(input_shape= input_shape, output_shape=output_shape) 
#model = build_deep_rnn(timesteps=timesteps,feature_count=train_data.shape[1]-1,unit_count_per_rnn_layer=[128,128])

model.summary()

learning_rate = 1e-3
epochs = 10 
batch_size = 32

# Compile the model
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

model.compile(
    optimizer=optimizer,
    loss='sparse_categorical_crossentropy', # When the labels are integers
    metrics= ['accuracy']
)

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    'models/rnn/best_RNN_model_v1.keras',
    monitor='val_accuracy',
    save_best_only=True
)

patience = 5
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=patience,
    restore_best_weights=True
)

# Train the model
history = model.fit(
    x_train,
    y_train,
    batch_size=batch_size,
    epochs=epochs,
    validation_data=(x_test, y_test),
    callbacks=[early_stopping, checkpoint],
    class_weight=class_weights_dict
)

plot_history(history, metric='accuracy')