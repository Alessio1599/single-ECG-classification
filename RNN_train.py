""" 
Ok I have a problem... I have to have the data in the columns of train data the feature
In this case I have only one feature, the ECG signal

Now is functioning, before wasn't function because in output I had (187,5) instead of (,5)
Because when I defined the model, return_sequences was set to True, so the output was a sequence of 187 values

WHY I SHOULD USE THE TIMESERIES GENERATOR?

IDK IF THIS CODE IS FUNCTIONING I SHOULD TAKE A LOOK
"""
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from utils.utils import load_data_RNN, class_weights
from models.RNN import build_RNN, build_deep_rnn

train_data, test_data = load_data_RNN() 

# Get the class weights  
#class_weights_dict = class_weights(y_train)

# Scale input feature
scaler_x = StandardScaler()
scaler_x.fit(train_data[:,:-1]) # tutti gli attributi tranne l'ultimo, calcola media e SD
train_data[:,:-1]=scaler_x.transform(train_data[:,:-1]) #remove mean and divide by SD the training set
test_data[:,:-1]=scaler_x.transform(test_data[:,:-1])

# Scale target variable
scaler_y = StandardScaler() #Istanza di standard scalar che è in grado di scaling e descaling
scaler_y.fit(train_data[:,-1].reshape(-1,1)) # solo l'ultimo attributo, dati del traffico
train_data[:,-1]=scaler_y.transform(train_data[:,-1].reshape(-1,1)).squeeze()
test_data[:,-1]=scaler_y.transform(test_data[:,-1].reshape(-1,1)).squeeze()

print("train_data shape: ",train_data.shape) # (87554, 188)

from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
# I have to use data generator for RNN
timesteps=187 # 187 is the length of the time series, of a signal
batch_size=32 

train_data_gen = TimeseriesGenerator(
    data=train_data, # training dataframe
    targets=train_data[:,-1], # last column of the dataframe
    length=timesteps, # time steps of the time series
    batch_size=batch_size)

test_data_gen= TimeseriesGenerator(
    data=test_data,
    targets=test_data[:,-1],
    length=timesteps,
    batch_size=batch_size)

print('Number of train batches: ',len(train_data_gen)) # 2736
print('Number of test batches: ',len(test_data_gen)) # 684

x,y = train_data_gen[0]

print('Time sequences feature batch shape: ',x.shape) # (32, 187, 188)
print('Time sequences target batch shape: ',y.shape) # (32,)

print('First feature time sequence:\n',x[0])
print('First target value: ',y[0])

input_shape = (timesteps,train_data.shape[1])
output_shape = len(np.unique(train_data[:,-1]))
model = build_RNN(input_shape= input_shape, output_shape=output_shape) 
#model = build_deep_rnn(timesteps=timesteps,feature_count=train_data.shape[1]-1,unit_count_per_rnn_layer=[128,128])

model.summary()

learning_rate = 1e-3
epochs = 10
# Compile the model
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
model.compile(
    optimizer=optimizer,
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']    
)

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

history = model.fit(
    train_data_gen,
    validation_data=test_data_gen,
    epochs=epochs,
    callbacks=[early_stopping]
)
