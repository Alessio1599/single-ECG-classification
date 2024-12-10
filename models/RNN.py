import tensorflow as tf
from tensorflow import keras
from keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, SimpleRNN, GRU, LSTM, Dense, Dropout, BatchNormalization


def build_RNN1(input_shape, output_shape):
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


# Previously used in rnn_wandb.ipynb
def build_model(input_shape, 
                output_shape,
                number_units,
                number_layers):
    input_layer = Input(input_shape)
    
    x = LSTM(number_units, return_sequences=True)(input_layer)
    for i in range(number_layers-1):
        x = LSTM(number_units, return_sequences=True)(x)
    
    x = LSTM(number_units)(x) # I cannot use return_sequences=True because the next layer is not a RNN layer!!!
    
    x = Dense(64, activation='relu')(x)
    
    output_layer = Dense(output_shape, activation='softmax')(x)
    model = Model(inputs=input_layer, outputs=output_layer)
    return model
  
# Can be used to tune hyperparameters in a model with fixed number of LSTM layers
def build_RNN2(input_shape, output_shape, layer_1_size, layer_2_size, layer_FC, dropout_rate):
    """
    Builds an RNN model with LSTM layers and a Dense output layer.
    """
    input_layer = Input(shape=input_shape)

    x = LSTM(layer_1_size, return_sequences=True)(input_layer)
    x = Dropout(dropout_rate)(x)
    x = LSTM(layer_2_size)(x)  # No return_sequences=True
    x = Dropout(dropout_rate)(x)

    x = Dense(layer_FC, activation='relu')(x)

    output_layer = Dense(output_shape, activation='softmax')(x)

    model = Model(inputs=input_layer, outputs=output_layer)
    return model
  

# Can be used if I want to choose between LSTM/GRU
def build_RNN3(input_shape,
                output_shape,
                num_units,
                num_layers,
                rnn_layer_type,
                fc_layer_size,
                dropout_rate):
    input_layer = Input(input_shape)

    # Select the RNN type based on the passed parameter (LSTM or GRU)
    rnn_layer = LSTM if rnn_layer_type == 'LSTM' else GRU

    # Add the first LSTM/GRU layer
    x = rnn_layer(num_units, return_sequences=True)(input_layer)

    # Add additional layers with dropout
    for i in range(num_layers - 2):
        x = rnn_layer(num_units, return_sequences=True)(x)
        x = Dropout(dropout_rate)(x)  # Dropout layer after each RNN layer

    x = rnn_layer(num_units)(x) ## Final RNN layer without return_sequences

    x = Dense(fc_layer_size, activation='relu')(x)

    output_layer = Dense(output_shape, activation='softmax')(x)

    model = Model(inputs=input_layer, outputs=output_layer)
    return model