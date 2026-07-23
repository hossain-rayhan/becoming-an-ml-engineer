"""
Shared Utilities for Project 1A

Contains data generation functions and helper utilities used across modules.
"""

import numpy as np


def generate_linear_data(n_samples=100, n_features=1, noise=0.1, seed=42):
    """
    Generate y = Xw + b + noise
    
    Args:
        n_samples: Number of data points
        n_features: Number of input features
        noise: Standard deviation of Gaussian noise
        seed: Random seed for reproducibility
    
    Returns:
        X: Input features (n_samples, n_features)
        y: Target values (n_samples,)
        true_w: True weights used to generate data
        true_b: True bias used to generate data
    """
    np.random.seed(seed)
    
    true_w = np.random.randn(n_features) * 2
    true_b = np.random.randn() * 0.5
    
    X = np.random.randn(n_samples, n_features)
    y = X @ true_w + true_b + np.random.randn(n_samples) * noise
    
    return X, y, true_w, true_b


def generate_binary_data(n_samples=200, seed=42):
    """
    Generate linearly separable binary classification data.
    
    Creates two clusters with a linear decision boundary.
    """
    np.random.seed(seed)
    
    n_per_class = n_samples // 2
    
    X0 = np.random.randn(n_per_class, 2) * 0.8 + np.array([-1, -1])
    y0 = np.zeros(n_per_class)
    
    X1 = np.random.randn(n_per_class, 2) * 0.8 + np.array([1, 1])
    y1 = np.ones(n_per_class)
    
    X = np.vstack([X0, X1])
    y = np.hstack([y0, y1])
    
    shuffle_idx = np.random.permutation(n_samples)
    X, y = X[shuffle_idx], y[shuffle_idx]
    
    return X, y


def normalize_features(X):
    """
    Standardize features to zero mean and unit variance.
    
    Args:
        X: Input features (n_samples, n_features)
    
    Returns:
        X_normalized: Normalized features
        mean: Mean of each feature
        std: Standard deviation of each feature
    """
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1  # Prevent division by zero
    X_normalized = (X - mean) / std
    return X_normalized, mean, std


def train_test_split(X, y, test_size=0.2, seed=42):
    """
    Split data into training and test sets.
    """
    np.random.seed(seed)
    n_samples = len(y)
    indices = np.random.permutation(n_samples)
    
    n_test = int(n_samples * test_size)
    test_idx = indices[:n_test]
    train_idx = indices[n_test:]
    
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def compute_accuracy(y_true, y_pred):
    """Compute classification accuracy."""
    return np.mean(y_true == y_pred)


def compute_mse(y_true, y_pred):
    """Compute mean squared error."""
    return np.mean((y_true - y_pred) ** 2)


def compute_r2_score(y_true, y_pred):
    """Compute R² score (coefficient of determination)."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)
