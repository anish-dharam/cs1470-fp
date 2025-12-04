"""
Main training script for wheat futures forecasting model.
"""

import os
import sys

import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data.data_loader import load_data
from src.data.data_preprocessing import preprocess_images, preprocess_tabular
from src.models.combined_model import create_combined_model
from src.training.metrics import evaluate_model
from src.training.trainer import Trainer
from src.utils.config import (
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    MODEL_SAVE_PATH,
    NUM_PRICE_FEATURES,
    RESULTS_PATH,
)
from src.utils.visualization import (
    plot_pnl,
    plot_training_history,
    visualize_predictions,
)


def main():
    """Main training function."""
    print("=" * 60)
    print("Wheat Futures Forecasting - Training Script")
    print("=" * 60)
    
    # load and preproces data
    print("\n[1/5] Loading data...")
    X_train, X_val, X_test, y_train, y_val, y_test = load_data(
        num_samples=1000,
        test_size=0.15,
        val_size=0.15,
        random_state=42
    )
    
    # preprocess images
    print("\n[2/5] Preprocessing images...")
    X_train['images'] = preprocess_images(X_train['images'])
    X_val['images'] = preprocess_images(X_val['images'])
    X_test['images'] = preprocess_images(X_test['images'])
    
    # preprocess tabular data
    print("\n[3/5] Preprocessing tabular data...")
    X_train['tabular'], scaler = preprocess_tabular(X_train['tabular'], fit=True)
    X_val['tabular'], _ = preprocess_tabular(X_val['tabular'], scaler=scaler, fit=False)
    X_test['tabular'], _ = preprocess_tabular(X_test['tabular'], scaler=scaler, fit=False)
    
    # create model
    print("\n[4/5] Creating model...")
    model = create_combined_model()
    model.summary()
    
    # create trainer
    trainer = Trainer(model, learning_rate=LEARNING_RATE)
    
    # train model j
    print("\n[5/5] Training model...")
    history = trainer.train(
        X_train, y_train,
        X_val, y_val,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1
    )
    
    print("\nSaving model...")
    trainer.save_model(MODEL_SAVE_PATH)
    
    print("\nEvaluating on test set...")
    prices_test = X_test['tabular'][:, -NUM_PRICE_FEATURES:]  # Last features are price features
    
    metrics = evaluate_model(
        model,
        X_test,
        y_test,
        prices_test=prices_test,
        threshold=0.5
    )
    
    print(f"\nTest Accuracy: {metrics['accuracy']:.4f}")
    if 'final_pnl' in metrics:
        print(f"Final PNL: ${metrics['final_pnl']:.2f}")
    
    os.makedirs(RESULTS_PATH, exist_ok=True)
    
    # visualize results
    print("\nGenerating visualizations...")
    plot_training_history(history, save_path=os.path.join(RESULTS_PATH, 'training_history.png'))
    visualize_predictions(
        y_test, metrics['predictions'],
        save_path=os.path.join(RESULTS_PATH, 'predictions.png')
    )
    if 'pnl_history' in metrics:
        plot_pnl(metrics['pnl_history'], save_path=os.path.join(RESULTS_PATH, 'pnl.png'))
    
    print("\n" + "=" * 60)
    print("Training completed!")
    print(f"Model saved to: {MODEL_SAVE_PATH}")
    print(f"Results saved to: {RESULTS_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()

