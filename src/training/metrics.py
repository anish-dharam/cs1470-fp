"""
Evaluation metrics for the wheat futures forecasting model.
"""

from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
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


def _best_exit_price(
    price_df: pd.DataFrame,
    anchor_date: datetime,
    horizon_days: int,
    trade_direction: int,  # 1 for long, -1 for short
    min_days: int = 10,
    max_days: int = 30,
) -> Optional[float]:
    """
    Find the best exit price, prioritizing exactly horizon_days (20 days).
    Uses High price for long positions and Low price for short positions from day 20.
    Falls back to flexible window [min_days, max_days] if day 20 is not available.
    
    For long positions: uses High price (best exit for long)
    For short positions: uses Low price (best exit for short)
    Falls back to Price column if High/Low not available.
    
    Args:
        price_df: Price dataframe
        anchor_date: Date to start from
        horizon_days: Target forecast horizon (e.g., 20 days)
        trade_direction: 1 for long, -1 for short
        min_days: Minimum days to look ahead for fallback (default 10)
        max_days: Maximum days to look ahead for fallback (default 30)
    
    Returns:
        Best exit price from day 20 (High for long, Low for short), or from closest date
        in [min_days, max_days] window if day 20 not available, or None if no prices found
    """
    target_date = anchor_date + timedelta(days=horizon_days)
    
    # First, try to get price from exactly day 20
    target_rows = price_df[price_df["Date"] == target_date]
    if not target_rows.empty:
        target_row = target_rows.iloc[0]
        
        # Select best price based on trade direction from day 20
        if trade_direction == 1:  # Long position - want highest price
            # Try High, then Price, then Open as fallback
            if "High" in price_df.columns and pd.notna(target_row.get("High")):
                return float(target_row["High"])
            elif "Price" in price_df.columns and pd.notna(target_row.get("Price")):
                return float(target_row["Price"])
            elif "Open" in price_df.columns and pd.notna(target_row.get("Open")):
                return float(target_row["Open"])
        else:  # Short position - want lowest price
            # Try Low, then Price, then Open as fallback
            if "Low" in price_df.columns and pd.notna(target_row.get("Low")):
                return float(target_row["Low"])
            elif "Price" in price_df.columns and pd.notna(target_row.get("Price")):
                return float(target_row["Price"])
            elif "Open" in price_df.columns and pd.notna(target_row.get("Open")):
                return float(target_row["Open"])
    
    # Fallback: if day 20 not available, use flexible window [min_days, max_days]
    min_target_date = anchor_date + timedelta(days=min_days)
    max_target_date = anchor_date + timedelta(days=max_days)
    
    future_rows = price_df[
        (price_df["Date"] >= min_target_date) & (price_df["Date"] <= max_target_date)
    ]
    if future_rows.empty:
        return None
    
    # Calculate distance from target horizon for each available date
    future_rows = future_rows.copy()
    future_rows["days_from_target"] = (
        (future_rows["Date"] - target_date).dt.total_seconds() / 86400
    ).abs()
    
    # Sort by distance from target horizon and pick the closest
    closest_row = future_rows.loc[future_rows["days_from_target"].idxmin()]
    
    # Select best price based on trade direction
    if trade_direction == 1:  # Long position - want highest price
        # Try High, then Price, then Open as fallback
        if "High" in price_df.columns and pd.notna(closest_row.get("High")):
            return float(closest_row["High"])
        elif "Price" in price_df.columns and pd.notna(closest_row.get("Price")):
            return float(closest_row["Price"])
        elif "Open" in price_df.columns and pd.notna(closest_row.get("Open")):
            return float(closest_row["Open"])
        else:
            return None
    else:  # Short position - want lowest price
        # Try Low, then Price, then Open as fallback
        if "Low" in price_df.columns and pd.notna(closest_row.get("Low")):
            return float(closest_row["Low"])
        elif "Price" in price_df.columns and pd.notna(closest_row.get("Price")):
            return float(closest_row["Price"])
        elif "Open" in price_df.columns and pd.notna(closest_row.get("Open")):
            return float(closest_row["Open"])
        else:
            return None


def calculate_regression_pnl(
    current_prices,
    predicted_prices,
    actual_future_prices,
    initial_capital=10000,
    image_dates=None,
    price_df=None,
    horizon_days=20,
    use_flexible_exit=False,
):
    """
    Calculate profit and loss based on regression price predictions.
    
    Strategy: If predicted price > current price, go long (buy).
              If predicted price < current price, go short (sell).
    
    Args:
        current_prices: Array of current prices at prediction time
        predicted_prices: Array of predicted future prices
        actual_future_prices: Array of actual future prices (used if use_flexible_exit=False)
        initial_capital: Initial capital for trading
        image_dates: Optional array of image dates (required if use_flexible_exit=True)
        price_df: Optional price dataframe (required if use_flexible_exit=True)
        horizon_days: Target forecast horizon (default 20)
        use_flexible_exit: If True, uses High/Low from exactly day 20 (High for long,
                          Low for short), falling back to flexible window [10, 30] days
                          if day 20 is not available
    
    Returns:
        Cumulative PNL array and final PNL
    """
    pnl = []
    capital = initial_capital
    
    for i in range(len(current_prices)):
        current_price = current_prices[i]
        predicted_price = predicted_prices[i]
        
        # Determine trade direction
        if predicted_price > current_price:
            trade_direction = 1  # Go long
        else:
            trade_direction = -1  # Go short
        
        # Get exit price
        if use_flexible_exit and image_dates is not None and price_df is not None:
            # Use High/Low from exactly day 20 (High for long, Low for short),
            exit_price = _best_exit_price(
                price_df, image_dates[i], horizon_days, trade_direction
            )
            if exit_price is None:
                # Fallback to provided actual future price
                exit_price = actual_future_prices[i]
        else:
            # Use provided actual future price
            exit_price = actual_future_prices[i]
        
        # Execute trade
        if trade_direction == 1:  # Long
            capital = capital * (exit_price / current_price)
        else:  # Short
            capital = capital * (current_price / exit_price)
        
        pnl.append(capital - initial_capital)
    
    final_pnl = pnl[-1] if len(pnl) > 0 else 0
    return np.array(pnl), final_pnl


def evaluate_regression(
    model,
    X_test,
    y_test,
    target_scaler=None,
    tabular_scaler=None,
    image_dates=None,
    price_df=None,
    horizon_days=20,
    use_flexible_exit=True,
):
    """
    Evaluate regression performance (price prediction).

    Args:
        model: Trained model
        X_test: Test data dict with 'images' and 'tabular' keys
        y_test: Test target values (may be scaled)
        target_scaler: Optional scaler for target values
        tabular_scaler: Optional scaler for tabular features (needed for PNL calculation)
        image_dates: Optional array of image dates (needed for flexible exit PNL)
        price_df: Optional price dataframe (needed for flexible exit PNL)
        horizon_days: Target forecast horizon (default 20)
        use_flexible_exit: If True, uses High/Low from exactly day 20 (High for long,
                          Low for short), falling back to flexible window [10, 30] days
                          if day 20 is not available (default True)

    Returns:
        dict with mae, rmse, mape, predictions, pnl_history, final_pnl, profit_pct
    """
    preds = model.predict([X_test['images'], X_test['tabular']], verbose=0).flatten()
    y_true = y_test

    if target_scaler is not None:
        preds = target_scaler.inverse_transform(preds.reshape(-1, 1)).flatten()
        y_true = target_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

    mae = mean_absolute_error(y_true, preds)
    rmse = np.sqrt(mean_squared_error(y_true, preds))
    mape = np.mean(np.abs((y_true - preds) / np.clip(np.abs(y_true), 1e-6, None))) * 100

    # Calculate PNL using current prices from tabular features
    # Price is the first column in PRICE_FEATURE_COLUMNS
    current_prices_scaled = X_test['tabular'][:, 0]
    
    # Unscale current prices if tabular scaler is provided
    if tabular_scaler is not None:
        # Create a dummy array with all features, unscale, then extract price column
        # We need to unscale just the price column (index 0)
        # Create a full feature array with zeros, set price column, unscale, extract
        dummy_features = np.zeros((len(current_prices_scaled), X_test['tabular'].shape[1]))
        dummy_features[:, 0] = current_prices_scaled
        dummy_unscaled = tabular_scaler.inverse_transform(dummy_features)
        current_prices = dummy_unscaled[:, 0]
    else:
        # Assume prices are already in original scale
        current_prices = current_prices_scaled
    
    # Determine if we can use flexible exit
    can_use_flexible = (
        use_flexible_exit
        and image_dates is not None
        and price_df is not None
        and len(image_dates) == len(current_prices)
    )
    
    # Calculate PNL
    pnl_history, final_pnl = calculate_regression_pnl(
        current_prices,
        preds,
        y_true,
        initial_capital=10000,
        image_dates=image_dates if can_use_flexible else None,
        price_df=price_df if can_use_flexible else None,
        horizon_days=horizon_days,
        use_flexible_exit=can_use_flexible,
    )
    profit_pct = (final_pnl / 10000) * 100

    return {
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "predictions": preds,
        "true": y_true,
        "pnl_history": pnl_history,
        "final_pnl": final_pnl,
        "profit_pct": profit_pct,
    }




