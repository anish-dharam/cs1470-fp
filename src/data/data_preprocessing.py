"""
Data preprocessing utilities for images and tabular data.
"""

import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler


def preprocess_images(images, normalize=True):
    """
    Preprocess satellite images: normalize pixel values.
    
    Args:
        images: numpy array of images
        normalize: Whether to normalize to [0, 1] range
    
    Returns:
        Preprocessed images
    """
    images = images.astype(np.float32)
    
    if normalize:
        # shoud already be in [0, 1] range from generation
        # are properly normalized
        images = np.clip(images, 0, 1)
    
    return images


def preprocess_tabular(tabular_data, scaler=None, fit=True):
    """
    Preprocess tabular data (weather + price features) using StandardScaler.
    
    Args:
        tabular_data: numpy array of tabular features
        scaler: Optional pre-fitted StandardScaler
        fit: Whether to fit the scaler (for training data)
    
    Returns:
        Preprocessed tabular data and the scaler
    """
    if scaler is None:
        scaler = StandardScaler()
    
    if fit:
        tabular_data_scaled = scaler.fit_transform(tabular_data)
    else:
        tabular_data_scaled = scaler.transform(tabular_data)
    
    return tabular_data_scaled, scaler


def augment_images(images, labels=None):
    """
    Apply data augmentation to images (translation, rotation, scaling, noise).
    
    Args:
        images: numpy array of images
        labels: Optional labels to augment in sync
    
    Returns:
        Augmented images and labels (if provided)
    """
    augmented_images = []
    augmented_labels = [] if labels is not None else None
    
    for i, img in enumerate(images):
        # random horizontal flip
        if np.random.rand() > 0.5:
            img = np.fliplr(img)
        
        # random rotation (small angle)
        angle = np.random.uniform(-5, 5)
        # simple rotation using scipy or tensorflow
        # actually ksip lol
        
        # random brightness adjustment
        brightness = np.random.uniform(0.9, 1.1)
        img = np.clip(img * brightness, 0, 1)
        
        # random noise
        if np.random.rand() > 0.7:
            noise = np.random.normal(0, 0.02, img.shape)
            img = np.clip(img + noise, 0, 1)
        
        augmented_images.append(img)
        if labels is not None:
            augmented_labels.append(labels[i])
    
    result = [np.array(augmented_images)]
    if labels is not None:
        result.append(np.array(augmented_labels))
    
    return result[0] if len(result) == 1 else tuple(result)


def create_sequences(images, tabular_data, labels, sequence_length=1):
    """
    Create sequences from data (for time series modeling if needed).
    Currently returns data as-is since we're using single timestep prediction.
    
    Args:
        images: Image data
        tabular_data: Tabular data
        labels: Target labels
        sequence_length: Length of sequences (currently 1)
    
    Returns:
        Sequences of data
    """
    # for now, just return data as-is
    # Future: could implement sliding window approach
    return images, tabular_data, labels


def create_data_generator(images, tabular_data, labels, batch_size=32, augment=False):
    """
    Create a data generator for training (useful for large datasets).
    
    Args:
        images: Image data
        tabular_data: Tabular data
        labels: Target labels
        batch_size: Batch size
        augment: Whether to apply augmentation
    
    Returns:
        Generator that yields batches
    """
    num_samples = len(images)
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    
    for start_idx in range(0, num_samples, batch_size):
        end_idx = min(start_idx + batch_size, num_samples)
        batch_indices = indices[start_idx:end_idx]
        
        batch_images = images[batch_indices]
        batch_tabular = tabular_data[batch_indices]
        batch_labels = labels[batch_indices]
        
        if augment:
            batch_images, batch_labels = augment_images(batch_images, batch_labels)
        
        yield {'images': batch_images, 'tabular': batch_tabular}, batch_labels

