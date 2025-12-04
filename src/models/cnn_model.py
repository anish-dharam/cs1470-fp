"""
CNN architecture for processing satellite imagery.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from src.utils.config import CNN_FILTERS, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS


def create_cnn_backbone(input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS),
                        filters=CNN_FILTERS):
    """
    Create CNN backbone for processing satellite images.
    
    Args:
        input_shape: Shape of input images (height, width, channels)
        filters: List of filter sizes for each conv layer
    
    Returns:
        Keras model that outputs feature vector
    """
    inputs = keras.Input(shape=input_shape, name='image_input')
    
    x = inputs
    
    # Frist conv block
    x = layers.Conv2D(filters[0], (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    # Secnd conv block
    x = layers.Conv2D(filters[1], (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    # Thrid conv block
    x = layers.Conv2D(filters[2], (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    # Additonal conv layers for feature extraction
    x = layers.Conv2D(filters[2], (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    
    # Globl average pooling to get feature vector
    x = layers.GlobalAveragePooling2D()(x)
    
    # Dense layr to reduce dimensions
    x = layers.Dense(128, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    
    # Create model
    model = keras.Model(inputs=inputs, outputs=x, name='cnn_backbone')
    
    return model

