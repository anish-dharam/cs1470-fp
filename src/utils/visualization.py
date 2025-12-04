"""
Visualization utilities for training history, predictions, and PNL.
"""

import matplotlib.pyplot as plt
import numpy as np


def plot_training_history(history, save_path=None):
    """
    Plot training and validation loss and accuracy curves.
    
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
    
    # Plot accuracy
    axes[1].plot(history.history['accuracy'], label='Train Accuracy')
    axes[1].plot(history.history['val_accuracy'], label='Val Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Model Accuracy')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    
    plt.close()


def visualize_predictions(y_true, y_pred, num_samples=10, save_path=None):
    """
    Visualize sample predictions vs actual labels.
    
    Args:
        y_true: True binary labels
        y_pred: Predicted probabilities
        num_samples: Number of samples to visualize
        save_path: Optional path to save the figure
    """
    # Convert probabilities to binary predictions
    y_pred_binary = (y_pred > 0.5).astype(int)
    
    # Select random samples
    indices = np.random.choice(len(y_true), min(num_samples, len(y_true)), replace=False)
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot predictions
    x = range(len(indices))
    axes[0].bar([i - 0.2 for i in x], y_true[indices], width=0.4, label='True', alpha=0.7)
    axes[0].bar([i + 0.2 for i in x], y_pred_binary[indices], width=0.4, label='Predicted', alpha=0.7)
    axes[0].set_xlabel('Sample Index')
    axes[0].set_ylabel('Label (0=Down, 1=Up)')
    axes[0].set_title('Binary Predictions vs True Labels')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot probabilities
    axes[1].plot(x, y_pred[indices], 'o-', label='Predicted Probability', alpha=0.7)
    axes[1].axhline(y=0.5, color='r', linestyle='--', label='Threshold')
    axes[1].set_xlabel('Sample Index')
    axes[1].set_ylabel('Probability')
    axes[1].set_title('Prediction Probabilities')
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

