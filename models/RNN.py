import tensorflow as tf
from tensorflow import keras
from keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, SimpleRNN, GRU, LSTM, Dense, Dropout, BatchNormalization


def build_RNN(input_shape, output_shape):
    """ 
    I could also add a Batch Normalization layer before the activation function.
    """
    input_layer = Input(input_shape)
    
    x =LSTM(128, return_sequences=True)(input_layer)
    x = Dropout(0.2)(x)
    x = LSTM(64)(x) # I cannot use return_sequences=True because the next layer is not a RNN layer!!!
    x = Dropout(0.2)(x)
    
    x = Dense(64, activation='relu')(x)
    
    output_layer = Dense(output_shape, activation='softmax')(x)
    
    model = Model(inputs=input_layer, outputs=output_layer) #activation='softmax'
    return model


def build_deep_rnn(timesteps,feature_count,unit_count_per_rnn_layer=[128,128]):
  model = keras.Sequential()
  model.add(layers.Input(shape=(timesteps,feature_count))) # layer di input

  for i in range(len(unit_count_per_rnn_layer)):
    model.add(layers.SimpleRNN(unit_count_per_rnn_layer[i],return_sequences=True if i<(len(unit_count_per_rnn_layer)-1) else False))
    # LSTM, GRU, SimpleRNN

  if unit_count_per_rnn_layer[-1]>1: # l'ultimo layer ricorrerente deve avere un valore solo
    model.add(layers.Dense(1))# in questo caso non specificando la funzione di attivazione, fa solo una combinazione lineare

  return model

