"""
Prediction script for making forecasts on new data.
"""

import os
import sys
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data.data_loader import generate_dummy_satellite_data, generate_dummy_weather_data, generate_dummy_futures_data
from src.data.data_preprocessing import preprocess_images, preprocess_tabular
from src.models.combined_model import create_combined_model
from src.training.trainer import Trainer
from src.utils.config import MODEL_SAVE_PATH, NUM_WEATHER_FEATURES, NUM_PRICE_FEATURES


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
    Make predictions on new dummy data (for demonstration).
    
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
    
    # Generate new dummy data
    print(f"\nGenerating {num_samples} new samples...")
    new_images = generate_dummy_satellite_data(num_samples)
    new_weather = generate_dummy_weather_data(num_samples)
    new_prices = generate_dummy_futures_data(num_samples)
    
    # Preprocess data
    print("Preprocessing data...")
    new_images = preprocess_images(new_images)
    new_tabular = np.concatenate([new_weather, new_prices], axis=1)
    
    # Note: In real usage, you'd load a fitted scaler from training
    # For now, we'll create a dummy scaler (not ideal, but works for demo)
    new_tabular, _ = preprocess_tabular(new_tabular, fit=True)
    
    # Prepare data dict
    X_new = {
        'images': new_images,
        'tabular': new_tabular
    }
    
    # Make predictions
    print("Making predictions...")
    predictions = trainer.predict(X_new)
    predictions_binary = (predictions >= 0.5).astype(int)
    
    # Display results
    print("\n" + "=" * 60)
    print("Predictions:")
    print("=" * 60)
    print(f"{'Sample':<10} {'Probability':<15} {'Prediction':<15} {'Direction':<15}")
    print("-" * 60)
    
    for i in range(num_samples):
        prob = predictions[i][0]
        pred = predictions_binary[i][0]
        direction = "UP" if pred == 1 else "DOWN"
        print(f"{i+1:<10} {prob:.4f}         {pred:<15} {direction:<15}")
    
    print("=" * 60)
    print(f"\nSummary:")
    print(f"  Samples predicted UP: {np.sum(predictions_binary == 1)}")
    print(f"  Samples predicted DOWN: {np.sum(predictions_binary == 0)}")
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



