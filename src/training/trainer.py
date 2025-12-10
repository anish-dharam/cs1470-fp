"""
Training utilities for the combined model.
"""

import os

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from src.utils.config import (
    CHECKPOINT_PATH,
    EPOCHS,
    LEARNING_RATE,
    MODEL_SAVE_PATH,
    WEIGHT_DECAY,
)


class Trainer:
    """
    Trainer class for handling model training, validation, and saving.
    """
    
    def __init__(self, model, learning_rate=LEARNING_RATE):
        """
        Initialize trainer.
        
        Args:
            model: Keras model to train
            learning_rate: Learning rate for optimizer
        """
        self.model = model
        self.learning_rate = learning_rate
        self.history = None
        
        self._compile_model()
    
    def _compile_model(self):
        """Compile the model with optimizer and loss function (regression)."""
        optimizer = keras.optimizers.AdamW(
            learning_rate=self.learning_rate,
            weight_decay=WEIGHT_DECAY,
            clipnorm=1.0,
        )
        self.model.compile(
            optimizer=optimizer,
            loss=keras.losses.Huber(),
            metrics=[
                keras.metrics.MeanAbsoluteError(name="mae"),
                keras.metrics.RootMeanSquaredError(name="rmse"),
            ],
        )
    
    def train(self, X_train, y_train, X_val, y_val, 
              epochs=EPOCHS, batch_size=32, 
              callbacks=None, verbose=1):
        """
        Train the model.
        
        Args:
            X_train: Training data dict with 'images' and 'tabular' keys
            y_train: Training labels
            X_val: Validation data dict
            y_val: Validation labels
            epochs: Number of epochs
            batch_size: Batch size
            callbacks: Optional list of callbacks
            verbose: Verbosity level
        
        Returns:
            Training history
        """
        if callbacks is None:
            callbacks = self._create_default_callbacks()
        
        self.history = self.model.fit(
            [X_train['images'], X_train['tabular']],
            y_train,
            validation_data=([X_val['images'], X_val['tabular']], y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose
        )
        
        return self.history
    
    def _create_default_callbacks(self):
        """Create default training callbacks."""
        # Create directories if they don't exist
        os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
        os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
        
        callbacks = [
            # Early stopping to prevent overfitting
            EarlyStopping(
                monitor='val_loss',
                patience=8,
                restore_best_weights=True,
                verbose=1
            ),
            # Save model checkpoints
            ModelCheckpoint(
                CHECKPOINT_PATH,
                monitor='val_loss',
                save_best_only=True,
                save_weights_only=False,
                verbose=1
            ),
            # Reduce learning rate on plateau
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.3,
                patience=4,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        return callbacks
    
    def save_model(self, filepath=MODEL_SAVE_PATH):
        """
        Save the trained model.
        
        Args:
            filepath: Path to save the model
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath=MODEL_SAVE_PATH):
        """
        Load a saved model.
        
        Args:
            filepath: Path to load the model from
        """
        self.model = keras.models.load_model(filepath)
        self._compile_model()
        print(f"Model loaded from {filepath}")
    
    def predict(self, X):
        """
        Make predictions on new data.
        
        Args:
            X: Data dict with 'images' and 'tabular' keys
        
        Returns:
            Predictions (prices)
        """
        return self.model.predict([X['images'], X['tabular']], verbose=0)

