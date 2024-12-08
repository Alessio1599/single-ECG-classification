import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization
from keras import activations

def build_CNN(
    input_shape, 
    output_shape, 
    layer_1_size=32, 
    layer_2_size=64, 
    layer_3_size=32,
    layer_FC_size=100, 
    dropout_rate=0.25): # I have to define firslty the mandatory parameters and then the optional ones
    """ 
    Builds a CNN model with 1D convolutional layers using the Functional API.
    
    Args:
    input_shape (tuple): Shape of the input data.
    output_shape (int): Number of classes for the output layer.
    layer_1_size (int): Number of filters in the first convolutional layer.
    layer_2_size (int): Number of filters in the second convolutional layer.
    layer_3_size (int): Number of filters in the third convolutional layer.
    layer_FC_size (int): Number of neurons in the fully connected layer.
    dropout_rate (float): Dropout rate for all dropout layers.

    Returns:
    model (Model): Keras Functional API model.
    """
    input_layer = Input(input_shape)
    
    # First Convolutional Layer
    x = Conv1D(filters=layer_1_size, kernel_size=5, padding='same')(input_layer)
    x = BatchNormalization(momentum=0.99, epsilon=0.001)(x)
    x = Dropout(dropout_rate)(x)
    x = activations.relu(x)
    x = MaxPooling1D(pool_size=2)(x)
    
    # Second convolutional layer
    x = Conv1D(filters=layer_2_size, kernel_size=3, padding='same')(x)
    x = BatchNormalization(momentum=0.99, epsilon=0.001)(x)
    x = activations.relu(x)  # Applying activation separately
    x = Dropout(dropout_rate)(x)
    x = MaxPooling1D(pool_size=2)(x)
    
    # Third convolutional layer
    x = Conv1D(filters=layer_3_size, kernel_size=3, padding='same')(x)
    x = BatchNormalization(momentum=0.99, epsilon=0.001)(x)
    x = activations.relu(x)
    x = Dropout(dropout_rate)(x)
    x = MaxPooling1D(pool_size=2)(x)
    
    # Flatten and Dense layers
    x = Flatten()(x)
    x = Dense(layer_FC_size, activation='relu')(x)
    x = Dropout(dropout_rate)(x)
    
    # Output layer
    output_layer = Dense(output_shape, activation='softmax')(x)
    
    # Creating the model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model

def build_CNN_for(
    input_shape, 
    output_shape, 
    filter_count_per_layer = [32,64,32]):
    """ 
    Args:
    input_shape: tuple, shape of the input data
    output_shape: int, number of classes
    filter_count_per_layer: list of integers, number of filters in each convolutional layer
    
    Returns:
    model (Model): Keras Functional API model.
    
    I've introduced a new parameter filter_count_per_layer which is a list of integers.
    In this way I can test different architectures without changing the code, different number of filters per layer.
    """
    input_layer = Input(input_shape)
    x = input_layer
    
    for filter_count in filter_count_per_layer:
        x = Conv1D(filters=filter_count, kernel_size=5, padding='same')(x)
        x = BatchNormalization(momentum=0.99, epsilon=0.001)(x)
        x = Dropout(0.25)(x)
        x = activations.relu(x)
        x = MaxPooling1D(pool_size=2)(x)
    
    # Flatten and Dense layers
    x = Flatten()(x)
    x = Dense(100, activation='relu')(x)
    x = Dropout(0.5)(x)
    
    # Output layer
    output_layer = Dense(output_shape, activation='softmax')(x)
    
    # Creating the model
    model = Model(inputs=input_layer, outputs=output_layer)
    return model 
    