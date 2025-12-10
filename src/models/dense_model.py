"""
Dense network architecture for processing tabular data (weather + price features).
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers

from src.utils.config import (
    DENSE_DROPOUT,
    DENSE_UNITS,
    NUM_PRICE_FEATURES,
    NUM_WEATHER_FEATURES,
    WEIGHT_DECAY,
)


def create_dense_network(
    input_dim=None,
    units=DENSE_UNITS,
    dropout_rate=DENSE_DROPOUT,
    use_bn=True,
):
    """
    Create dense network for processing tabular data.
    
    Args:
        input_dim: Input dimension (weather + price features)
        units: List of units for each dense layer
        dropout_rate: Dropout rate for regularization
        use_bn: Whether to apply batch normalization
    
    Returns:
        Keras model that outputs feature vector
    """
    if input_dim is None:
        input_dim = NUM_WEATHER_FEATURES + NUM_PRICE_FEATURES
    
    inputs = keras.Input(shape=(input_dim,), name="tabular_input")
    x = inputs
    
    for num_units in units:
        x = layers.Dense(
            num_units,
            activation="relu",
            kernel_regularizer=regularizers.l2(WEIGHT_DECAY),
        )(x)
        if use_bn:
            x = layers.LayerNormalization()(x)
        x = layers.Dropout(dropout_rate)(x)
    
    x = layers.Dense(
        64,
        activation="relu",
        kernel_regularizer=regularizers.l2(WEIGHT_DECAY),
    )(x)
    if use_bn:
        x = layers.LayerNormalization()(x)
    
    model = keras.Model(inputs=inputs, outputs=x, name="dense_network")
    return model




