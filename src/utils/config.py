"""
Configuration parameters for the wheat futures forecasting model.
"""

# Image parameters
IMAGE_HEIGHT = 128
IMAGE_WIDTH = 128
IMAGE_CHANNELS = 3  # RGB

# Data parameters
NUM_WEATHER_FEATURES = 10  # cloud cover, sun elevation, sun azimuth, temp, precip, etc.
NUM_PRICE_FEATURES = 5  # price, volume, moving averages
FORECAST_HORIZON = 20  # predict 20 trading days ahead

# Training parameters
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 50
VALIDATION_SPLIT = 0.15
TEST_SPLIT = 0.15

# Model architecture
CNN_FILTERS = [32, 64, 128]
DENSE_UNITS = [64, 128, 64]
FUSION_UNITS = [128, 64]
DROPOUT_RATE = 0.3

# Paths
MODEL_SAVE_PATH = "models/wheat_futures_model.h5"
CHECKPOINT_PATH = "checkpoints/weights.{epoch:02d}-{val_loss:.2f}.h5"
RESULTS_PATH = "results/"



