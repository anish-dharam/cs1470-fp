"""
Configuration parameters for the wheat futures forecasting model.
"""

# Image parameters
IMAGE_HEIGHT = 128
IMAGE_WIDTH = 128
IMAGE_CHANNELS = 3  # RGB

# Data parameters
NUM_WEATHER_FEATURES = 0  # no separate weather tabular features in current dataset
NUM_PRICE_FEATURES = 7  # price, OHLC, change pct, MA5, MA20
FORECAST_HORIZON = 20  # predict 20 trading days ahead

# Training parameters
BATCH_SIZE = 16
LEARNING_RATE = 0.001
EPOCHS = 120
VALIDATION_SPLIT = 0.15
TEST_SPLIT = 0.15
WEIGHT_DECAY = 5e-5

# Model architecture (tunable hyperparameters)
CNN_FILTERS = [64, 128, 192]
CNN_KERNEL_SIZE = 3
CNN_USE_BN = True
CNN_DROPOUT = 0.2
CNN_POOL = "max"  # max or avg
CNN_EXTRA_BLOCKS = 0  # extra conv block after main stack

DENSE_UNITS = [256, 128]
DENSE_DROPOUT = 0.25
FUSION_UNITS = [256, 128, 64]
FUSION_DROPOUT = 0.25

# Paths
MODEL_SAVE_PATH = "models/wheat_futures_model.h5"
TABULAR_SCALER_PATH = "models/tabular_scaler.pkl"
TARGET_SCALER_PATH = "models/target_scaler.pkl"
CHECKPOINT_PATH = "checkpoints/weights.{epoch:02d}-{val_loss:.2f}.h5"
RESULTS_PATH = "results/"




