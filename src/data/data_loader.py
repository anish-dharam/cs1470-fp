"""
Data loading module for satellite imagery, weather, and futures price data.
Currently uses dummy data generation. Future integration with Earth Engine API.
"""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from data_preprocessing import clean_wheat_csv

from src.utils.config import (
    FORECAST_HORIZON,
    IMAGE_CHANNELS,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    NUM_PRICE_FEATURES,
    NUM_WEATHER_FEATURES,
)


def generate_dummy_satellite_data(num_samples, image_height=IMAGE_HEIGHT, 
                                   image_width=IMAGE_WIDTH, 
                                   image_channels=IMAGE_CHANNELS):
    """
    Generate dummy Landsat satellite imagery data.
    
    Args:
        num_samples: Number of images to generate
        image_height: Height of images
        image_width: Width of images
        image_channels: Number of channels (RGB)
    
    Returns:
        numpy array of shape (num_samples, image_height, image_width, image_channels)
    """
    # Generat dummy images with some realistic patterns
    # Simulate agricultural fields with varying greenness
    np.random.seed(42)
    images = []
    
    for i in range(num_samples):
        # Base image with some structure (simulating fields)
        img = np.random.rand(image_height, image_width, image_channels) * 0.3
        
        # Add some green channel bias (vegetation)
        green_bias = np.random.rand() * 0.4 + 0.2
        img[:, :, 1] += green_bias
        
        # Add some patterns to simulate field boundaries
        for j in range(0, image_height, 20):
            img[j:j+2, :, :] = 0.1  # Horizontal lines
        for k in range(0, image_width, 20):
            img[:, k:k+2, :] = 0.1  # Vertical lines
        
        # Add noise
        noise = np.random.normal(0, 0.05, (image_height, image_width, image_channels))
        img = np.clip(img + noise, 0, 1)
        
        images.append(img)
    
    return np.array(images)


def generate_dummy_weather_data(num_samples, num_features=NUM_WEATHER_FEATURES):
    """
    Generate dummy weather features.
    
    Args:
        num_samples: Number of samples
        num_features: Number of weather features
    
    Returns:
        numpy array of shape (num_samples, num_features)
    """
    np.random.seed(42)
    
    # Generate realistic weather data
    weather_data = np.zeros((num_samples, num_features))
    
    for i in range(num_samples):
        # Cloud cover (0-1)
        weather_data[i, 0] = np.random.beta(2, 5)
        
        # Sun elevation (degrees, 0-90)
        weather_data[i, 1] = np.random.uniform(20, 70)
        
        # Sun azimuth (degrees, 0-360)
        weather_data[i, 2] = np.random.uniform(0, 360)
        
        # Temperature (Celsius, -10 to 35)
        weather_data[i, 3] = np.random.normal(15, 10)
        
        # Precipitation (mm, 0-50)
        weather_data[i, 4] = np.random.exponential(5)
        
        # Humidity (0-100%)
        weather_data[i, 5] = np.random.uniform(30, 90)
        
        # Wind speed (m/s, 0-20)
        weather_data[i, 6] = np.random.exponential(3)
        
        # Pressure (hPa, 980-1020)
        weather_data[i, 7] = np.random.normal(1013, 10)
        
        # Solar radiation (W/m^2, 0-1000)
        weather_data[i, 8] = np.random.uniform(0, 800)
        
        # Visibility (km, 0-50)
        weather_data[i, 9] = np.random.uniform(5, 50)
    
    return weather_data


def generate_dummy_futures_data(num_samples, num_features=NUM_PRICE_FEATURES):
    """
    Generate dummy wheat futures price data.
    
    Args:
        num_samples: Number of samples
        num_features: Number of price features
    
    Returns:
        numpy array of shape (num_samples, num_features)
    """
    np.random.seed(42)
    
    # Generate realistic price data with some trend
    base_price = 600.0  # Base wheat futures price in USD/bushel
    prices = []
    
    current_price = base_price
    for i in range(num_samples):
        # Random walk with slight upward bias
        change = np.random.normal(0.1, 2.0)
        current_price = max(400, min(800, current_price + change))
        
        # Create features: price, volume, moving averages
        price_features = np.zeros(num_features)
        price_features[0] = current_price
        price_features[1] = np.random.uniform(1000, 10000)  # Volume
        price_features[2] = current_price + np.random.normal(0, 1)  # MA5
        price_features[3] = current_price + np.random.normal(0, 2)  # MA20
        price_features[4] = np.random.uniform(-0.05, 0.05)  # Price change %
        
        prices.append(price_features)
    
    return np.array(prices)

def generate_futures_data(csv_path, num_samples, num_features=5):
    """
    Load real wheat futures data from cleaned CSV and return features in the
    same format as generate_dummy_futures_data.

    Args:
        csv_path: Path to the raw CSV
        num_samples: Number of samples requested
        num_features: Number of features (must match model expectation)

    Returns:
        numpy array of shape (num_samples, num_features)
    """
    # get cleaned data
    df = clean_wheat_csv(csv_path)

    # generate values for 5 and 20 day averages
    df["MA5"] = df["Price"].rolling(window=5).mean()
    df["MA20"] = df["Price"].rolling(window=20).mean()

    df = df.dropna(subset=["MA5", "MA20"]).reset_index(drop=True)

    # keep relevant values
    features = df[["Price", "Volume", "MA5", "MA20", "Change %"]].to_numpy(dtype=np.float32)

    if len(features) >= num_samples:
        features = features[-num_samples:]
    else:
        repeats = int(np.ceil(num_samples / len(features)))
        features = np.tile(features, (repeats, 1))[:num_samples]

    assert features.shape == (num_samples, num_features)

    return features


def generate_target_labels(futures_data, forecast_horizon=FORECAST_HORIZON):
    """
    Generate binary target labels (price direction 20 days ahead).
    
    Args:
        futures_data: Array of futures price data
        forecast_horizon: Number of days ahead to predict
    
    Returns:
        numpy array of binary labels (0=down, 1=up)
    """
    labels = []
    
    for i in range(len(futures_data) - forecast_horizon):
        current_price = futures_data[i, 0]
        future_price = futures_data[i + forecast_horizon, 0]
        
        # Binary classification: 1 if price goes up, 0 if down
        label = 1 if future_price > current_price else 0
        labels.append(label)
    
    # Pad the last few samples with zeros (can't predict future)
    labels.extend([0] * forecast_horizon)
    
    return np.array(labels)


def fetch_earth_engine_data(region, start_date, end_date):
    """
    Placeholder function for future Earth Engine API integration.
    
    Args:
        region: Region of interest (bounding box)
        start_date: Start date for data collection
        end_date: End date for data collection
    
    Returns:
        Tuple of (satellite_images, weather_data, futures_data)
    """
    # TODO: Implement Earth Engine API integration
    # This will replace dummy data generation in the future
    pass


def load_data(num_samples=1000, test_size=0.15, val_size=0.15, random_state=42):
    """
    Main function to load and split data into train/val/test sets.
    
    Args:
        num_samples: Total number of samples to generate
        test_size: Proportion of data for testing
        val_size: Proportion of data for validation (from remaining after test split)
        random_state: Random seed
    
    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        where X is a dict with keys 'images' and 'tabular'
    """
    # Generat dummy data
    print("Generating dummy satellite data...")
    satellite_images = generate_dummy_satellite_data(num_samples)
    
    print("Generating dummy weather data...")
    weather_data = generate_dummy_weather_data(num_samples)
    
    print("Generating dummy futures data...")
    futures_data = generate_dummy_futures_data(num_samples)
    
    # Combin weather and price features
    tabular_data = np.concatenate([weather_data, futures_data], axis=1)
    
    # Generat target labels
    print("Generating target labels...")
    labels = generate_target_labels(futures_data)
    
    # Align data (remove last FORECAST_HORIZON samples that don't have labels)
    satellite_images = satellite_images[:-FORECAST_HORIZON]
    tabular_data = tabular_data[:-FORECAST_HORIZON]
    labels = labels[:-FORECAST_HORIZON]
    
    # Split into train/test first
    X_img_train, X_img_test, X_tab_train, X_tab_test, y_train, y_test = train_test_split(
        satellite_images, tabular_data, labels,
        test_size=test_size, random_state=random_state, stratify=labels
    )
    
    # Split train into train/val
    val_size_adjusted = val_size / (1 - test_size)
    X_img_train, X_img_val, X_tab_train, X_tab_val, y_train, y_val = train_test_split(
        X_img_train, X_tab_train, y_train,
        test_size=val_size_adjusted, random_state=random_state, stratify=y_train
    )
    
    # Package data
    X_train = {'images': X_img_train, 'tabular': X_tab_train}
    X_val = {'images': X_img_val, 'tabular': X_tab_val}
    X_test = {'images': X_img_test, 'tabular': X_tab_test}
    
    print(f"Data loaded: Train={len(y_train)}, Val={len(y_val)}, Test={len(y_test)}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test

