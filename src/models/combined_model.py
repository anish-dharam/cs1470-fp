"""
Combined model that fuses CNN and dense network outputs for final prediction.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from src.models.cnn_model import create_cnn_backbone
from src.models.dense_model import create_dense_network
from src.utils.config import (
    FUSION_DROPOUT,
    FUSION_UNITS,
    IMAGE_CHANNELS,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    NUM_PRICE_FEATURES,
    NUM_WEATHER_FEATURES,
)


def create_combined_model(
    image_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS),
    tabular_dim=None,
    fusion_units=FUSION_UNITS,
    dropout_rate=FUSION_DROPOUT,
):
    """
    Create combined model that fuses CNN and dense network.
    """
    if tabular_dim is None:
        tabular_dim = NUM_WEATHER_FEATURES + NUM_PRICE_FEATURES
    
    cnn_model = create_cnn_backbone(input_shape=image_shape)
    dense_model = create_dense_network(input_dim=tabular_dim)
    
    combined = layers.concatenate([cnn_model.output, dense_model.output])
    
    x = combined
    for num_units in fusion_units:
        x = layers.Dense(num_units, activation="relu")(x)
        x = layers.LayerNormalization()(x)
        x = layers.Dropout(dropout_rate)(x)
    
    output = layers.Dense(1, activation="linear", name="price")(x)
    
    model = keras.Model(
        inputs=[cnn_model.input, dense_model.input],
        outputs=output,
        name="combined_model",
    )
    return model




