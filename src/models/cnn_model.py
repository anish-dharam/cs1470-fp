"""
CNN architecture for processing satellite imagery.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers

from src.utils.config import (
    CNN_DROPOUT,
    CNN_EXTRA_BLOCKS,
    CNN_FILTERS,
    CNN_KERNEL_SIZE,
    CNN_POOL,
    CNN_USE_BN,
    IMAGE_CHANNELS,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    WEIGHT_DECAY,
)


def _conv_block(x, filters, kernel_size, pool_type="max", use_bn=True, dropout_rate=0.0):
    x = layers.Conv2D(
        filters,
        kernel_size,
        padding="same",
        activation="relu",
        kernel_regularizer=regularizers.l2(WEIGHT_DECAY),
    )(x)
    if use_bn:
        x = layers.LayerNormalization()(x)
    if pool_type == "max":
        x = layers.MaxPooling2D((2, 2))(x)
    elif pool_type == "avg":
        x = layers.AveragePooling2D((2, 2))(x)
    if dropout_rate and dropout_rate > 0:
        x = layers.Dropout(dropout_rate)(x)
    return x


def create_cnn_backbone(
    input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS),
    filters=CNN_FILTERS,
    kernel_size=CNN_KERNEL_SIZE,
    pool_type=CNN_POOL,
    use_bn=CNN_USE_BN,
    dropout_rate=CNN_DROPOUT,
    extra_blocks=CNN_EXTRA_BLOCKS,
):
    """
    Create CNN backbone for processing satellite images.

    Args:
        input_shape: Shape of input images (height, width, channels)
        filters: List of filter sizes for each conv block
        kernel_size: Convolution kernel size
        pool_type: "max" or "avg"
        use_bn: Whether to apply batch normalization
        dropout_rate: Dropout after each block
        extra_blocks: Additional conv blocks with last filter size

    Returns:
        Keras model that outputs feature vector
    """
    inputs = keras.Input(shape=input_shape, name="image_input")

    # Lightweight augmentations to improve generalization on tiny dataset
    augmentation = keras.Sequential(
        [
            layers.RandomFlip("horizontal_and_vertical"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
        ],
        name="img_augment",
    )

    x = augmentation(inputs)

    for f in filters:
        x = _conv_block(
            x,
            filters=f,
            kernel_size=kernel_size,
            pool_type=pool_type,
            use_bn=use_bn,
            dropout_rate=dropout_rate,
        )

    # depth with highest filter size
    for _ in range(extra_blocks):
        x = layers.Conv2D(
            filters[-1],
            kernel_size,
            padding="same",
            activation="relu",
            kernel_regularizer=regularizers.l2(WEIGHT_DECAY),
        )(x)
        if use_bn:
            x = layers.LayerNormalization()(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(
        256,
        activation="relu",
        kernel_regularizer=regularizers.l2(WEIGHT_DECAY),
    )(x)
    if use_bn:
        x = layers.LayerNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)

    model = keras.Model(inputs=inputs, outputs=x, name="cnn_backbone")
    return model

