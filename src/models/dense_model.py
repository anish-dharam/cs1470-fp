"""
Dense network architecture for processing tabular data (weather + price features).
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from src.utils.config import DENSE_UNITS, DROPOUT_RATE, NUM_WEATHER_FEATURES, NUM_PRICE_FEATURES


def create_dense_network(input_dim=None, units=DENSE_UNITS, dropout_rate=DROPOUT_RATE):
    """
    Create dense network for processing tabular data.
    
    Args:
        input_dim: Input dimension (weather + price features)
        units: List of units for each dense layer
        dropout_rate: Dropout rate for regularization
    
    Returns:
        Keras model that outputs feature vector
    """
    if input_dim is None:
        input_dim = NUM_WEATHER_FEATURES + NUM_PRICE_FEATURES
    
    inputs = keras.Input(shape=(input_dim,), name='tabular_input')
    
    x = inputs
    
    # Dense layers with dropout
    for i, num_units in enumerate(units):
        x = layers.Dense(num_units, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(dropout_rate)(x)
    
    # Final dense layer to reduce dimensions
    x = layers.Dense(64, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    
    # Create model
    model = keras.Model(inputs=inputs, outputs=x, name='dense_network')
    
    return model


