"""
Main training script for wheat futures forecasting model.
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
from src.training.metrics import evaluate_regression
from src.training.trainer import Trainer
from src.utils.config import (
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    MODEL_SAVE_PATH,
    RESULTS_PATH,
    TABULAR_SCALER_PATH,
    TARGET_SCALER_PATH,
)
from src.utils.visualization import plot_training_history, visualize_predictions


def main():
    """Main training function."""
    print("=" * 60)
    print("Wheat Futures Forecasting - Training Script")
    print("=" * 60)
    
    # load data
    print("\n[1/4] Loading data...")
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = load_data()

    # scale targets for stability
    from sklearn.preprocessing import StandardScaler
    target_scaler = StandardScaler()
    y_train_scaled = target_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_val_scaled = target_scaler.transform(y_val.reshape(-1, 1)).flatten()
    y_test_scaled = target_scaler.transform(y_test.reshape(-1, 1)).flatten()
    Path(MODEL_SAVE_PATH).parent.mkdir(parents=True, exist_ok=True)
    Path(TABULAR_SCALER_PATH).parent.mkdir(parents=True, exist_ok=True)
    Path(TARGET_SCALER_PATH).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, TABULAR_SCALER_PATH)
    joblib.dump(target_scaler, TARGET_SCALER_PATH)
    
    # create model
    print("\n[2/4] Creating model...")
    model = create_combined_model(tabular_dim=X_train['tabular'].shape[1])
    model.summary()
    
    # create trainer
    trainer = Trainer(model, learning_rate=LEARNING_RATE)
    
    # train model 
    print("\n[3/4] Training model...")
    history = trainer.train(
        X_train, y_train_scaled,
        X_val, y_val_scaled,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1
    )
    
    print("\nSaving model...")
    trainer.save_model(MODEL_SAVE_PATH)
    
    # summarize train/val at best val_loss epoch
    best_idx = int(np.argmin(history.history['val_loss']))
    print("\nValidation summary (best epoch):")
    print(f"  Epoch: {best_idx + 1}")
    print(f"  Train Loss: {history.history['loss'][best_idx]:.4f}")
    print(f"  Train MAE : {history.history['mae'][best_idx]:.4f}")
    print(f"  Train RMSE: {history.history['rmse'][best_idx]:.4f}")
    print(f"  Val Loss  : {history.history['val_loss'][best_idx]:.4f}")
    print(f"  Val MAE   : {history.history['val_mae'][best_idx]:.4f}")
    print(f"  Val RMSE  : {history.history['val_rmse'][best_idx]:.4f}")
    
    print("\nEvaluating on test set (unscaled targets)...")
    metrics = evaluate_regression(model, X_test, y_test_scaled, target_scaler=target_scaler)
    
    print(f"  Test MAE : {metrics['mae']:.4f}")
    print(f"  Test RMSE: {metrics['rmse']:.4f}")
    print(f"  Test MAPE: {metrics['mape']:.2f}%")
    
    os.makedirs(RESULTS_PATH, exist_ok=True)
    
    # visualize results
    print("\nGenerating visualizations...")
    plot_training_history(history, save_path=os.path.join(RESULTS_PATH, 'training_history.png'))
    visualize_predictions(
        y_test, metrics['predictions'],
        save_path=os.path.join(RESULTS_PATH, 'predictions.png')
    )
    
    print("\n" + "=" * 60)
    print("Training completed!")
    print(f"Model saved to: {MODEL_SAVE_PATH}")
    print(f"Results saved to: {RESULTS_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()

