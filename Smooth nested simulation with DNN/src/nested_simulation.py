"""
Implementation of the Smooth Nested Simulation algorithm.
"""

import numpy as np
import torch
from typing import Optional, Tuple, Dict, List
from tqdm import tqdm
import time
from .kernel_ridge import KernelRidge, TorchKernelRidge
from .utils import (
    generate_nested_data,
    compute_rmse,
    compute_r2,
    set_seed,
    EarlyStopping
)
from .config import (
    DEVICE,
    KERNEL_TYPES,
    NN_CONFIG,
    TRAIN_CONFIG,
    EXPERIMENT_CONFIG
)

class NestedSimulator:
    """
    Implementation of the Smooth Nested Simulation algorithm with
    kernel ridge regression and neural network integration.
    """
    
    def __init__(
        self,
        dimension: int,
        budget: int,
        kernel_type: str = 'rbf',
        noise_level: float = 0.1,
        alpha: float = 1.0,
        kernel_params: Optional[Dict] = None,
        use_gpu: bool = True,
        random_state: Optional[int] = None
    ):
        """
        Initialize the Nested Simulator.
        
        Parameters
        ----------
        dimension : int
            Input dimension
        budget : int
            Total simulation budget
        kernel_type : str
            Type of kernel for ridge regression
        noise_level : float
            Noise level for inner simulations
        alpha : float
            Ridge regression regularization parameter
        kernel_params : dict, optional
            Parameters for the kernel function
        use_gpu : bool
            Whether to use GPU acceleration
        random_state : int, optional
            Random seed for reproducibility
        """
        self.dimension = dimension
        self.budget = budget
        self.kernel_type = kernel_type
        self.noise_level = noise_level
        self.alpha = alpha
        self.kernel_params = kernel_params
        self.use_gpu = use_gpu and torch.cuda.is_available()
        
        if random_state is not None:
            set_seed(random_state)
        
        # Initialize models
        self.kernel_model = KernelRidge(
            kernel_type=kernel_type,
            alpha=alpha,
            kernel_params=kernel_params
        )
        
        # Calculate sample sizes
        self.n_outer = int(np.sqrt(budget))  # Optimal allocation
        self.n_inner = budget // self.n_outer
        
        # Initialize results storage
        self.results = {
            'rmse': [],
            'r2': [],
            'computation_time': [],
            'n_outer': self.n_outer,
            'n_inner': self.n_inner,
            'total_budget': budget
        }
    
    def _generate_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate nested simulation data.
        
        Returns
        -------
        X : ndarray of shape (n_outer, dimension)
            Outer-level input points
        y_true : ndarray of shape (n_outer,)
            True conditional expectations
        y_noisy : ndarray of shape (n_outer,)
            Noisy observations
        """
        # Generate outer and inner level data
        X, y_true, true_function = generate_nested_data(
            n_outer=self.n_outer,
            n_inner=self.n_inner,
            dimension=self.dimension,
            noise_level=self.noise_level
        )
        
        # Add noise to simulate inner-level uncertainty
        y_noisy = y_true + np.random.normal(
            0,
            self.noise_level,
            size=self.n_outer
        )
        
        return X, y_true, y_noisy
    
    def run_single_experiment(self) -> Dict:
        """
        Run a single nested simulation experiment.
        
        Returns
        -------
        dict
            Results of the experiment
        """
        start_time = time.time()
        
        # Generate data
        X, y_true, y_noisy = self._generate_data()
        
        # Fit kernel ridge regression
        if self.use_gpu:
            self.kernel_model = self.kernel_model.to_torch()
        
        self.kernel_model.fit(X, y_noisy)
        
        # Make predictions
        y_pred = self.kernel_model.predict(X)
        
        # Compute metrics
        rmse = compute_rmse(y_true, y_pred)
        r2 = compute_r2(y_true, y_pred)
        computation_time = time.time() - start_time
        
        return {
            'rmse': rmse,
            'r2': r2,
            'computation_time': computation_time
        }
    
    def run_experiments(self, n_experiments: int = 30) -> Dict:
        """
        Run multiple nested simulation experiments.
        
        Parameters
        ----------
        n_experiments : int
            Number of experiments to run
            
        Returns
        -------
        dict
            Aggregated results of all experiments
        """
        all_results = []
        
        for _ in tqdm(range(n_experiments), desc="Running experiments"):
            result = self.run_single_experiment()
            all_results.append(result)
        
        # Aggregate results
        rmse_values = [r['rmse'] for r in all_results]
        r2_values = [r['r2'] for r in all_results]
        time_values = [r['computation_time'] for r in all_results]
        
        self.results.update({
            'mean_rmse': np.mean(rmse_values),
            'std_rmse': np.std(rmse_values),
            'mean_r2': np.mean(r2_values),
            'std_r2': np.std(r2_values),
            'mean_time': np.mean(time_values),
            'std_time': np.std(time_values),
            'n_experiments': n_experiments
        })
        
        return self.results
    
    def get_convergence_rate(
        self,
        budget_multipliers: List[int],
        n_experiments: int = 30
    ) -> Dict:
        """
        Analyze convergence rate by running experiments with different budgets.
        
        Parameters
        ----------
        budget_multipliers : list of int
            List of budget multipliers to test
        n_experiments : int
            Number of experiments for each budget
            
        Returns
        -------
        dict
            Convergence analysis results
        """
        convergence_results = []
        
        original_budget = self.budget
        
        for multiplier in tqdm(budget_multipliers, desc="Analyzing convergence"):
            self.budget = original_budget * multiplier
            self.n_outer = int(np.sqrt(self.budget))
            self.n_inner = self.budget // self.n_outer
            
            results = self.run_experiments(n_experiments)
            convergence_results.append({
                'budget': self.budget,
                'mean_rmse': results['mean_rmse'],
                'std_rmse': results['std_rmse']
            })
        
        # Restore original budget
        self.budget = original_budget
        self.n_outer = int(np.sqrt(self.budget))
        self.n_inner = self.budget // self.n_outer
        
        return {
            'convergence_results': convergence_results,
            'budget_multipliers': budget_multipliers
        }
