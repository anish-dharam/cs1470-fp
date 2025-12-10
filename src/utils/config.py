"""
Configuration parameters for the wheat futures forecasting model.
"""

# Image parameters
IMAGE_HEIGHT = 128
IMAGE_WIDTH = 128
IMAGE_CHANNELS = 3  # RGB

# Data parameters
NUM_WEATHER_FEATURES = 0  # no separate weather tabular features in current dataset
NUM_PRICE_FEATURES = 8  # price, OHLC, volume, change pct, MA5, MA20
FORECAST_HORIZON = 20  # predict 20 trading days ahead

# Training parameters
BATCH_SIZE = 4
LEARNING_RATE = 0.0002
EPOCHS = 120
VALIDATION_SPLIT = 0.15
TEST_SPLIT = 0.15
WEIGHT_DECAY = 1e-4

# Model architecture (tunable hyperparameters)
CNN_FILTERS = [32, 64, 96]
CNN_KERNEL_SIZE = 3
CNN_USE_BN = True
CNN_DROPOUT = 0.25
CNN_POOL = "max"  # max or avg
CNN_EXTRA_BLOCKS = 0  # extra conv block after main stack

DENSE_UNITS = [128, 64]
DENSE_DROPOUT = 0.3
FUSION_UNITS = [128, 64]
FUSION_DROPOUT = 0.3

# Paths
MODEL_SAVE_PATH = "models/wheat_futures_model.h5"
TABULAR_SCALER_PATH = "models/tabular_scaler.pkl"
TARGET_SCALER_PATH = "models/target_scaler.pkl"
CHECKPOINT_PATH = "checkpoints/weights.{epoch:02d}-{val_loss:.2f}.h5"
RESULTS_PATH = "results/"




