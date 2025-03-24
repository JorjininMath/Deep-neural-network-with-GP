"""
Utility functions for the Smooth Nested Simulation project.
"""

import numpy as np
import torch
from typing import Tuple, Optional, Callable
from .config import DEVICE

def generate_random_function(dimension: int) -> Callable:
    """
    Generate a random smooth function for testing.
    
    Parameters
    ----------
    dimension : int
        Input dimension
        
    Returns
    -------
    callable
        A random smooth function that takes a numpy array of shape (n_samples, dimension)
    """
    # Generate random coefficients for polynomial terms
    coeffs = np.random.randn(dimension)
    
    def random_function(X: np.ndarray) -> np.ndarray:
        """
        Evaluate the random function at given points.
        
        The function is a combination of polynomial and trigonometric terms
        to ensure smoothness.
        """
        result = np.zeros(len(X))
        
        # Add polynomial terms
        result += np.dot(X, coeffs)
        
        # Add trigonometric terms for smoothness
        for i in range(dimension):
            result += np.sin(X[:, i]) * coeffs[i]
            result += np.cos(X[:, i]) * coeffs[i]
        
        return result
    
    return random_function

def generate_nested_data(
    n_outer: int,
    n_inner: int,
    dimension: int,
    noise_level: float,
    random_state: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, Callable]:
    """
    Generate data for nested simulation.
    
    Parameters
    ----------
    n_outer : int
        Number of outer-level samples
    n_inner : int
        Number of inner-level samples per outer point
    dimension : int
        Input dimension
    noise_level : float
        Standard deviation of the noise
    random_state : int, optional
        Random seed for reproducibility
        
    Returns
    -------
    X : ndarray of shape (n_outer, dimension)
        Outer-level input points
    y : ndarray of shape (n_outer,)
        True conditional expectations
    true_function : callable
        The true function being approximated
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    # Generate outer-level points
    X = np.random.randn(n_outer, dimension)
    
    # Generate true function
    true_function = generate_random_function(dimension)
    
    # Compute true conditional expectations
    y = true_function(X)
    
    return X, y, true_function

def compute_mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute Mean Squared Error.
    """
    return np.mean((y_true - y_pred) ** 2)

def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute Root Mean Squared Error.
    """
    return np.sqrt(compute_mse(y_true, y_pred))

def compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute Mean Absolute Error.
    """
    return np.mean(np.abs(y_true - y_pred))

def compute_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute R² score.
    """
    u = ((y_true - y_pred) ** 2).sum()
    v = ((y_true - y_true.mean()) ** 2).sum()
    return 1 - u/v

class EarlyStopping:
    """Early stopping to prevent overfitting"""
    
    def __init__(self, patience: int = 7, min_delta: float = 0):
        """
        Parameters
        ----------
        patience : int
            How many epochs to wait before stopping when loss is not improving
        min_delta : float
            Minimum change in the monitored quantity to qualify as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    
    def __call__(self, val_loss: float) -> bool:
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
        
        return self.early_stop

def set_seed(seed: int):
    """
    Set random seed for reproducibility.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
