"""
Combined model that fuses CNN and dense network outputs for final prediction.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from src.models.cnn_model import create_cnn_backbone
from src.models.dense_model import create_dense_network
from src.utils.config import (
    IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS,
    NUM_WEATHER_FEATURES, NUM_PRICE_FEATURES,
    FUSION_UNITS, DROPOUT_RATE
)


def create_combined_model(image_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS),
                           tabular_dim=None,
                           fusion_units=FUSION_UNITS,
                           dropout_rate=DROPOUT_RATE):
    """
    Create combined model that fuses CNN and dense network.
    
    Args:
        image_shape: Shape of input images
        tabular_dim: Dimension of tabular input
        fusion_units: List of units for fusion layers
        dropout_rate: Dropout rate
    
    Returns:
        Combined Keras model ready for training
    """
    if tabular_dim is None:
        tabular_dim = NUM_WEATHER_FEATURES + NUM_PRICE_FEATURES
    
    # Create CNN backbone
    cnn_model = create_cnn_backbone(input_shape=image_shape)
    
    # Create dense network
    dense_model = create_dense_network(input_dim=tabular_dim)
    
    # Get outputs from both models
    cnn_output = cnn_model.output
    dense_output = dense_model.output
    
    # Concatenate feature vectors
    combined = layers.concatenate([cnn_output, dense_output])
    
    # Fusion layers
    x = combined
    for num_units in fusion_units:
        x = layers.Dense(num_units, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(dropout_rate)(x)
    
    # Final output layer (binary classification)
    output = layers.Dense(1, activation='sigmoid', name='price_direction')(x)
    
    # Create combined model
    model = keras.Model(
        inputs=[cnn_model.input, dense_model.input],
        outputs=output,
        name='combined_model'
    )
    
    return model


