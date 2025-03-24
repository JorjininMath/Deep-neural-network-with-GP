"""
Kernel Ridge Regression implementation for smooth function approximation.
"""

import numpy as np
from scipy.spatial.distance import cdist
from typing import Dict, Union, Optional
import torch
from .config import KERNEL_TYPES, DEVICE

class KernelRidge:
    """
    Kernel Ridge Regression with various kernel functions and efficient computation
    for high-dimensional data.
    """
    
    def __init__(
        self,
        kernel_type: str = 'rbf',
        alpha: float = 1.0,
        kernel_params: Optional[Dict] = None
    ):
        """
        Initialize KernelRidge regression.
        
        Parameters
        ----------
        kernel_type : str
            Type of kernel ('rbf' or 'polynomial')
        alpha : float
            Ridge regression regularization parameter
        kernel_params : dict, optional
            Parameters for the kernel function
        """
        self.kernel_type = kernel_type
        self.alpha = alpha
        self.kernel_params = kernel_params or KERNEL_TYPES[kernel_type]['params']
        
        # Initialize parameters
        self.X_fit = None
        self.dual_coef_ = None
        
    def _rbf_kernel(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """
        Compute RBF (Gaussian) kernel between X and Y.
        
        K(x, y) = exp(-||x-y||^2 / (2 * length_scale^2))
        """
        length_scale = self.kernel_params['length_scale']
        dists = cdist(X / length_scale, Y / length_scale, metric='sqeuclidean')
        return np.exp(-0.5 * dists)
    
    def _polynomial_kernel(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """
        Compute polynomial kernel between X and Y.
        
        K(x, y) = (x·y + coef0)^degree
        """
        degree = self.kernel_params['degree']
        coef0 = self.kernel_params['coef0']
        return (np.dot(X, Y.T) + coef0) ** degree
    
    def _get_kernel(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Get the appropriate kernel matrix."""
        if self.kernel_type == 'rbf':
            return self._rbf_kernel(X, Y)
        elif self.kernel_type == 'polynomial':
            return self._polynomial_kernel(X, Y)
        else:
            raise ValueError(f"Unknown kernel type: {self.kernel_type}")
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit Kernel Ridge Regression model.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : array-like of shape (n_samples,)
            Target values
        """
        self.X_fit = X.copy()
        
        # Compute kernel matrix
        K = self._get_kernel(X, X)
        
        # Add ridge regularization
        K[np.diag_indices_from(K)] += self.alpha
        
        # Solve the dual problem
        self.dual_coef_ = np.linalg.solve(K, y)
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using the kernel ridge model.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples
            
        Returns
        -------
        y_pred : array-like of shape (n_samples,)
            Predicted target values
        """
        K = self._get_kernel(X, self.X_fit)
        return K.dot(self.dual_coef_)
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Returns the coefficient of determination R^2 of the prediction.
        """
        y_pred = self.predict(X)
        u = ((y - y_pred) ** 2).sum()
        v = ((y - y.mean()) ** 2).sum()
        return 1 - u/v
    
    def to_torch(self) -> 'TorchKernelRidge':
        """
        Convert the model to a PyTorch version for GPU acceleration.
        """
        return TorchKernelRidge.from_numpy(self)

class TorchKernelRidge:
    """
    PyTorch implementation of Kernel Ridge Regression for GPU acceleration.
    """
    
    @classmethod
    def from_numpy(cls, numpy_model: KernelRidge) -> 'TorchKernelRidge':
        """
        Create a TorchKernelRidge instance from a numpy KernelRidge model.
        """
        model = cls(
            kernel_type=numpy_model.kernel_type,
            alpha=numpy_model.alpha,
            kernel_params=numpy_model.kernel_params
        )
        
        if numpy_model.X_fit is not None:
            model.X_fit = torch.from_numpy(numpy_model.X_fit).float().to(DEVICE)
            model.dual_coef_ = torch.from_numpy(numpy_model.dual_coef_).float().to(DEVICE)
            
        return model
    
    def __init__(
        self,
        kernel_type: str = 'rbf',
        alpha: float = 1.0,
        kernel_params: Optional[Dict] = None
    ):
        """Initialize TorchKernelRidge with the same parameters as KernelRidge."""
        self.kernel_type = kernel_type
        self.alpha = alpha
        self.kernel_params = kernel_params or KERNEL_TYPES[kernel_type]['params']
        
        self.X_fit = None
        self.dual_coef_ = None
    
    def _rbf_kernel(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """Compute RBF kernel using PyTorch operations."""
        length_scale = self.kernel_params['length_scale']
        X = X / length_scale
        Y = Y / length_scale
        
        X_norm = (X ** 2).sum(1).view(-1, 1)
        Y_norm = (Y ** 2).sum(1).view(1, -1)
        K = torch.mm(X, Y.t())
        K.mul_(-2)
        K.add_(X_norm)
        K.add_(Y_norm)
        K.mul_(-0.5)
        return torch.exp(K)
    
    def _polynomial_kernel(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """Compute polynomial kernel using PyTorch operations."""
        degree = self.kernel_params['degree']
        coef0 = self.kernel_params['coef0']
        return (torch.mm(X, Y.t()) + coef0) ** degree
    
    def _get_kernel(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """Get the appropriate kernel matrix using PyTorch operations."""
        if self.kernel_type == 'rbf':
            return self._rbf_kernel(X, Y)
        elif self.kernel_type == 'polynomial':
            return self._polynomial_kernel(X, Y)
        else:
            raise ValueError(f"Unknown kernel type: {self.kernel_type}")
    
    def predict(self, X: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """
        Predict using the kernel ridge model with PyTorch.
        
        Parameters
        ----------
        X : array-like or torch.Tensor of shape (n_samples, n_features)
            Samples
            
        Returns
        -------
        y_pred : array-like or torch.Tensor of shape (n_samples,)
            Predicted target values
        """
        is_numpy = isinstance(X, np.ndarray)
        if is_numpy:
            X = torch.from_numpy(X).float().to(DEVICE)
        
        K = self._get_kernel(X, self.X_fit)
        y_pred = torch.mm(K, self.dual_coef_.view(-1, 1)).squeeze()
        
        if is_numpy:
            return y_pred.cpu().numpy()
        return y_pred
