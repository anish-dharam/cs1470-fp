"""
Visualization utilities for training history, predictions, and PNL.
"""

import matplotlib.pyplot as plt
import numpy as np


def plot_training_history(history, save_path=None):
    """
    Plot training and validation loss plus a secondary regression metric.
    
    Args:
        history: Keras training history object
        save_path: Optional path to save the figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot loss
    axes[0].plot(history.history['loss'], label='Train Loss')
    axes[0].plot(history.history['val_loss'], label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Model Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    metric_key = 'rmse' if 'rmse' in history.history else 'mae' if 'mae' in history.history else None
    if metric_key:
        val_key = f'val_{metric_key}'
        axes[1].plot(history.history[metric_key], label=f'Train {metric_key.upper()}')
        if val_key in history.history:
            axes[1].plot(history.history[val_key], label=f'Val {metric_key.upper()}')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel(metric_key.upper())
        axes[1].set_title(f'Model {metric_key.upper()}')
        axes[1].legend()
        axes[1].grid(True)
    else:
        axes[1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    
    plt.close()


def visualize_predictions(y_true, y_pred, num_samples=10, save_path=None):
    """
    Visualize regression predictions vs actuals.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        num_samples: Number of samples to visualize
        save_path: Optional path to save the figure
    """
    indices = np.random.choice(len(y_true), min(num_samples, len(y_true)), replace=False)
    y_true_sel = y_true[indices]
    y_pred_sel = y_pred[indices]
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Bar chart of sample predictions
    x = range(len(indices))
    axes[0].bar([i - 0.2 for i in x], y_true_sel, width=0.4, label='True', alpha=0.7)
    axes[0].bar([i + 0.2 for i in x], y_pred_sel, width=0.4, label='Predicted', alpha=0.7)
    axes[0].set_xlabel('Sample Index')
    axes[0].set_ylabel('Price')
    axes[0].set_title('Predicted vs True Prices')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Scatter plot true vs predicted
    axes[1].scatter(y_true, y_pred, alpha=0.6, label='Predictions')
    axes[1].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', label='Ideal')
    axes[1].set_xlabel('True Price')
    axes[1].set_ylabel('Predicted Price')
    axes[1].set_title('True vs Predicted Scatter')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    
    plt.close()


def plot_pnl(pnl_history, save_path=None):
    """
    Visualize profit and loss over time.
    
    Args:
        pnl_history: List or array of cumulative PNL values
        save_path: Optional path to save the figure
    """
    plt.figure(figsize=(10, 6))
    plt.plot(pnl_history, linewidth=2)
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    plt.xlabel('Time Step')
    plt.ylabel('Cumulative PNL')
    plt.title('Profit and Loss Over Time')
    plt.grid(True, alpha=0.3)
    
    # Add final PNL as text
    final_pnl = pnl_history[-1] if len(pnl_history) > 0 else 0
    plt.text(0.02, 0.98, f'Final PNL: {final_pnl:.2f}', 
             transform=plt.gca().transAxes, 
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    
    plt.close()




