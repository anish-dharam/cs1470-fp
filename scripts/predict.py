"""
Prediction script for making forecasts on new data.
"""

import os
import sys
from pathlib import Path

import joblib
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data.data_loader import load_data
from src.models.combined_model import create_combined_model
from src.training.trainer import Trainer
from src.utils.config import (
    MODEL_SAVE_PATH,
    TABULAR_SCALER_PATH,
    TARGET_SCALER_PATH,
)


def load_model_for_prediction(model_path=MODEL_SAVE_PATH):
    """
    Load trained model for prediction.
    
    Args:
        model_path: Path to saved model
    
    Returns:
        Loaded model
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Please train the model first.")
    
    model = create_combined_model()
    trainer = Trainer(model)
    trainer.load_model(model_path)
    
    return trainer


def predict_on_new_data(model_path=MODEL_SAVE_PATH, num_samples=10):
    """
    Make predictions on held-out test data.
    
    Args:
        model_path: Path to saved model
        num_samples: Number of samples to predict on
    """
    print("=" * 60)
    print("Wheat Futures Forecasting - Prediction Script")
    print("=" * 60)
    
    # Load model
    print("\nLoading trained model...")
    trainer = load_model_for_prediction(model_path)
    
    print("\nLoading dataset (test split) with saved scalers...")
    if not Path(TABULAR_SCALER_PATH).exists() or not Path(TARGET_SCALER_PATH).exists():
        raise FileNotFoundError(
            "Saved scalers not found. Please run training first to generate "
            f"{TABULAR_SCALER_PATH} and {TARGET_SCALER_PATH}."
        )
    tabular_scaler = joblib.load(TABULAR_SCALER_PATH)
    target_scaler = joblib.load(TARGET_SCALER_PATH)

    _, _, X_test, _, _, y_test, _ = load_data(tabular_scaler=tabular_scaler)
    y_test_scaled = target_scaler.transform(y_test.reshape(-1, 1)).flatten()
    X_new = {
        'images': X_test['images'][:num_samples],
        'tabular': X_test['tabular'][:num_samples],
    }
    y_true = target_scaler.inverse_transform(
        y_test_scaled[:num_samples].reshape(-1, 1)
    ).flatten()
    
    # Make predictions
    print("Making predictions...")
    predictions = trainer.predict(X_new).flatten()
    predictions = target_scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
    
    # Display results
    print("\n" + "=" * 60)
    print("Predictions:")
    print("=" * 60)
    print(f"{'Sample':<10} {'Predicted':<15} {'Actual':<15} {'Error':<15}")
    print("-" * 60)
    
    for i in range(num_samples):
        pred = predictions[i]
        actual = y_true[i]
        error = pred - actual
        print(f"{i+1:<10} {pred:>10.2f}     {actual:>10.2f}     {error:>10.2f}")
    
    print("=" * 60)
    print(f"\nSummary:")
    mae = np.mean(np.abs(predictions - y_true))
    rmse = np.sqrt(np.mean((predictions - y_true) ** 2))
    print(f"  MAE: {mae:.2f}")
    print(f"  RMSE: {rmse:.2f}")
    print("=" * 60)


def main():
    """Main prediction function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Make predictions on new data')
    parser.add_argument('--model_path', type=str, default=MODEL_SAVE_PATH,
                       help='Path to trained model')
    parser.add_argument('--num_samples', type=int, default=10,
                       help='Number of samples to predict on')
    
    args = parser.parse_args()
    
    predict_on_new_data(args.model_path, args.num_samples)


if __name__ == "__main__":
    main()




