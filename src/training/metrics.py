"""
Evaluation metrics for the wheat futures forecasting model.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
)


def calculate_accuracy(y_true, y_pred, threshold=0.5):
    """
    Calculate classification accuracy.
    
    Args:
        y_true: True binary labels
        y_pred: Predicted probabilities
        threshold: Threshold for binary classification
    
    Returns:
        Accuracy score
    """
    y_pred_binary = (y_pred >= threshold).astype(int)
    return accuracy_score(y_true, y_pred_binary)


def calculate_pnl(y_true, y_pred, prices, threshold=0.5, initial_capital=10000):
    """
    Calculate profit and loss based on predictions.
    
    Args:
        y_true: True binary labels (0=down, 1=up)
        y_pred: Predicted probabilities
        prices: Array of futures prices
        threshold: Threshold for binary classification
        initial_capital: Initial capital for trading
    
    Returns:
        Cumulative PNL array and final PNL
    """
    y_pred_binary = (y_pred >= threshold).astype(int)
    
    # Simple trading strategy: buy if predicted up, sell if predicted down
    # Assume we can trade at the current price
    pnl = []
    capital = initial_capital
    position = 0  # 0 = no position, 1 = long, -1 = short
    entry_price = None
    
    for i in range(len(y_true)):
        current_price = prices[i, 0] if len(prices.shape) > 1 else prices[i]
        
        # Close previous position if exists
        if position != 0:
            if position == 1:  # Close long
                capital = capital * (current_price / entry_price)
            else:  # Close short
                capital = capital * (entry_price / current_price)
            position = 0
        
        # Open new position based on prediction
        if y_pred_binary[i] == 1:  # Predict up, go long
            entry_price = current_price
            position = 1
        else:  # Predict down, go short
            entry_price = current_price
            position = -1
        
        # Calculate PNL
        pnl.append(capital - initial_capital)
    
    # Close final position
    if position != 0 and len(prices) > 0:
        final_price = prices[-1, 0] if len(prices.shape) > 1 else prices[-1]
        if position == 1:
            capital = capital * (final_price / entry_price)
        else:
            capital = capital * (entry_price / final_price)
        pnl[-1] = capital - initial_capital
    
    final_pnl = pnl[-1] if len(pnl) > 0 else 0
    
    return np.array(pnl), final_pnl


def calculate_confusion_matrix(y_true, y_pred, threshold=0.5):
    """
    Calculate confusion matrix.
    
    Args:
        y_true: True binary labels
        y_pred: Predicted probabilities
        threshold: Threshold for binary classification
    
    Returns:
        Confusion matrix
    """
    y_pred_binary = (y_pred >= threshold).astype(int)
    return confusion_matrix(y_true, y_pred_binary)


def evaluate_model(model, X_test, y_test, prices_test=None, threshold=0.5):
    """
    Comprehensive model evaluation.
    
    Args:
        model: Trained Keras model
        X_test: Test data dict
        y_test: Test labels
        prices_test: Test prices for PNL calculation
        threshold: Classification threshold
    
    Returns:
        Dictionary of evaluation metrics
    """
    # Make predictions
    y_pred = model.predict([X_test['images'], X_test['tabular']], verbose=0)
    y_pred = y_pred.flatten()
    
    # Calculate metrics
    accuracy = calculate_accuracy(y_test, y_pred, threshold)
    cm = calculate_confusion_matrix(y_test, y_pred, threshold)
    
    metrics = {
        'accuracy': accuracy,
        'confusion_matrix': cm,
        'predictions': y_pred,
        'true_labels': y_test
    }
    
    # Calculate PNL if prices provided
    if prices_test is not None:
        pnl_history, final_pnl = calculate_pnl(y_test, y_pred, prices_test, threshold)
        metrics['pnl_history'] = pnl_history
        metrics['final_pnl'] = final_pnl
    
    # Classification report
    y_pred_binary = (y_pred >= threshold).astype(int)
    metrics['classification_report'] = classification_report(
        y_test, y_pred_binary, output_dict=True
    )
    
    return metrics


def evaluate_regression(model, X_test, y_test, target_scaler=None):
    """
    Evaluate regression performance (price prediction).

    Returns:
        dict with mae, rmse, mape, predictions
    """
    preds = model.predict([X_test['images'], X_test['tabular']], verbose=0).flatten()
    y_true = y_test

    if target_scaler is not None:
        preds = target_scaler.inverse_transform(preds.reshape(-1, 1)).flatten()
        y_true = target_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

    mae = mean_absolute_error(y_true, preds)
    rmse = mean_squared_error(y_true, preds, squared=False)
    mape = np.mean(np.abs((y_true - preds) / np.clip(np.abs(y_true), 1e-6, None))) * 100

    return {
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "predictions": preds,
        "true": y_true,
    }




