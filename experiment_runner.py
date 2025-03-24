import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'  # Enable CPU fallback for unsupported MPS operations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from data_generator import DataGenerator
from network_estimator import NetworkEstimator
import os
from datetime import datetime
import json
from tqdm import tqdm
import torch
import time
from multiprocessing import Pool, cpu_count, Lock
import threading

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

class ExperimentRunner:
    '''
    Class to run multiple experiments with different configurations
    and collect results
    '''
    def __init__(self, base_dir='experiments'):
        self.base_dir = base_dir
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_dir = os.path.join(base_dir, f'experiment_{self.timestamp}')
        os.makedirs(self.experiment_dir, exist_ok=True)
        os.makedirs(os.path.join(self.experiment_dir, 'data'), exist_ok=True)
        
        # Define experiment configurations
        self.dimensions = [20, 100, 500]  # Test all dimensions
        
        # Sample sizes for each dimension
        # For d=20: 2^9 to 2^12
        # For d=100 and d=500: 2^10 to 2^13 (double the sample sizes)
        self.sample_sizes = {
            20: [2**k for k in range(9, 13)],     # 512, 1024, 2048, 4096
            100: [2**k for k in range(11, 15)],   # 2048, 4096, 8192, 16384
            500: [2**k for k in range(11, 15)]    # 2048, 4096, 8192, 16384
        }
        self.noise_levels = [0.1, 10.0]  # Low and high noise scenarios
        self.replications = 10  # m_i = 10 for all i
        self.macro_replications = 30  # Number of macro replications (R)
        self.num_processes = max(1, cpu_count() - 1)  # Use CPU cores - 1 for parallel processing
        
        # Training parameters
        self.epochs = 100          # Maximum training epochs
        self.batch_size = 32       # Batch size for training
        self.patience = 10         # Early stopping patience
        self.val_split = 0.2       # Validation split ratio
        self.prune_iterations = 10 # Number of pruning iterations
        self.retrain_epochs = 5    # Epochs for retraining after pruning
        
        # Create data generator
        self.data_generator = DataGenerator(save_dir=os.path.join(self.experiment_dir, 'data'))
        
        print("\nExperiment Configuration:")
        print("-------------------------")
        print(f"Dimensions: {self.dimensions}")
        print(f"Sample sizes:")
        for d in self.dimensions:
            print(f"  d={d}: {self.sample_sizes[d]}")
        print(f"Noise levels: {self.noise_levels}")
        print(f"Replications per point (m): {self.replications}")
        print(f"Macro replications (R): {self.macro_replications}")
        print(f"Number of parallel processes: {self.num_processes}")
        print("-------------------------\n")
        
    def generate_data_safe(self, n, d, m, xi, macro_rep):
        """Safely generate data while avoiding directory conflicts"""
        data_dir = os.path.join(self.experiment_dir, 'data')
        os.makedirs(data_dir, exist_ok=True)
        
        # Use macro_rep as unique identifier
        X, y = self.data_generator.generate_data(
            n=n,
            d=d,
            m=m,
            xi=xi,
            function='griewank',
            save=False  # Don't save to file
        )
        return X, y

    def run_single_experiment(self, n, d, xi, macro_rep):
        '''
        Run a single experiment with given configuration
        
        Parameters
        ----------
        n : int
            Number of design points
        d : int
            Input dimension
        xi : float
            Noise level
        macro_rep : int
            Current macro replication number
            
        Returns
        -------
        dict
            Experiment results
        '''
        start_time = time.time()  # Start timing
        
        print(f"\nRunning experiment with n={n}, d={d}, xi={xi}, macro_rep={macro_rep}")
        
        # Set random seed for reproducibility
        np.random.seed(macro_rep)
        torch.manual_seed(macro_rep)
        
        # Generate data
        X, y = self.generate_data_safe(n, d, self.replications, xi, macro_rep)
        
        # Calculate network parameters based on theory
        beta = 2  # Griewank function
        phi_n = n ** (-2 * beta / (2 * beta + 1))
        
        # Depth: number of hidden layers
        depth = max(1, int(n * phi_n))
        
        # Width: number of nodes per hidden layer
        width = int(2 * n * phi_n)
        
        # Calculate target sparsity
        target_nonzero = int(n * phi_n * np.log(n))
        total_params = d * width + width * width * (depth - 1) + width
        target_sparsity = max(0, min(1, 1 - target_nonzero / total_params))
        
        if macro_rep == 0:  # Only print architecture details for first macro replication
            print(f"\nNetwork Architecture:")
            print(f"phi_n: {phi_n:.6f}")
            print(f"Number of hidden layers (depth): {depth}")
            print(f"Nodes per hidden layer (width): {width}")
            print(f"Total parameters: {total_params}")
            print(f"Target non-zero parameters: {target_nonzero}")
            print(f"Target sparsity ratio: {target_sparsity:.4f}\n")
        
        # Create and train model with new features
        estimator = NetworkEstimator(
            input_dim=d,
            hidden_layers=[width] * depth,
            sparsity=target_sparsity,
            output_bound=2.0,
            batch_size=self.batch_size,
            patience=self.patience,
            prune_iterations=self.prune_iterations,
            retrain_epochs=self.retrain_epochs
        )
        
        # Train model with validation
        estimator.fit(
            X=X,
            y=y,
            m=self.replications,
            epochs=self.epochs,
            val_split=self.val_split
        )
        
        # Generate test data (without noise and replications)
        X_test, y_test = self.generate_data_safe(
            n=1000,  # Fixed test size
            d=d,
            m=1,
            xi=0,
            macro_rep=macro_rep
        )
        
        # Evaluate
        predictions = estimator.predict(X_test)
        rmse = np.sqrt(np.mean((predictions - y_test.reshape(-1, 1))**2))
        
        # Get model info
        model_info = estimator.get_model_info()
        
        result = {
            'n': n,
            'd': d,
            'xi': xi,
            'macro_rep': macro_rep,
            'rmse': rmse,
            'phi_n': phi_n,
            'depth': depth,
            'width': width,
            'total_params': model_info['total_parameters'],
            'nonzero_params': model_info['nonzero_parameters'],
            'actual_sparsity': model_info['sparsity_ratio'],
            'target_sparsity': target_sparsity,
            'computation_time': time.time() - start_time  # Add computation time
        }
        
        return result

    def run_parallel_macro_reps(self, args):
        n, d, xi, macro_reps = args
        results = []
        for r in macro_reps:
            result = self.run_single_experiment(n, d, xi, r)
            results.append(result)
        return results

    def run_experiments(self):
        '''
        Run all experiments and save results
        '''
        total_start_time = time.time()  # Start timing for entire program
        
        all_results = {
            'config': {
                'dimensions': [int(d) for d in self.dimensions],
                'sample_sizes': {str(d): [int(n) for n in sizes] for d, sizes in self.sample_sizes.items()},
                'noise_levels': [float(xi) for xi in self.noise_levels],
                'replications': int(self.replications),
                'macro_replications': int(self.macro_replications),
                'training_params': {
                    'epochs': self.epochs,
                    'batch_size': self.batch_size,
                    'patience': self.patience,
                    'val_split': self.val_split,
                    'prune_iterations': self.prune_iterations,
                    'retrain_epochs': self.retrain_epochs
                }
            },
            'experiments': [],
            'timing': {}  # Add timing information
        }
        
        # Create progress bars for each level
        dimension_pbar = tqdm(self.dimensions, desc='Dimensions')
        
        # Create process pool
        with Pool(processes=self.num_processes) as pool:
            for d in dimension_pbar:
                dimension_pbar.set_description(f'Dimension d={d}')
                
                sample_pbar = tqdm(self.sample_sizes[d], desc='Sample sizes', leave=False)
                for n in sample_pbar:
                    sample_pbar.set_description(f'Sample size n={n}')
                    
                    noise_pbar = tqdm(self.noise_levels, desc='Noise levels', leave=False)
                    for xi in noise_pbar:
                        noise_pbar.set_description(f'Noise level xi={xi}')
                        config_start_time = time.time()  # Start timing for this configuration
                        
                        # Calculate phi_n using n^(-2β/(2β + 1))
                        beta = 2  # Griewank function
                        phi_n = n ** (-2 * beta / (2 * beta + 1))
                        
                        # Calculate network architecture parameters
                        depth = max(1, int(n * phi_n))  # Removed 0.5 coefficient
                        width = int(2 * n * phi_n)
                        
                        # Calculate target sparsity
                        target_nonzero = int(n * phi_n * np.log(n))
                        total_params = d * width + width * width * (depth - 1) + width
                        target_sparsity = max(0, min(1, 1 - target_nonzero / total_params))
                        
                        # Store fixed information for this configuration
                        experiment_config = {
                            'settings': {
                                'n': int(n),
                                'd': int(d),
                                'xi': float(xi),
                                'phi_n': float(phi_n),
                                'depth': int(depth),
                                'width': int(width),
                                'target_sparsity': float(target_sparsity)
                            }
                        }
                        
                        # Split macro replications into batches
                        batch_size = max(1, self.macro_replications // self.num_processes)
                        macro_rep_batches = [
                            list(range(i, min(i + batch_size, self.macro_replications)))
                            for i in range(0, self.macro_replications, batch_size)
                        ]
                        
                        # Prepare parallel processing arguments
                        parallel_args = [(n, d, xi, batch) for batch in macro_rep_batches]
                        
                        # Run macro replications in parallel
                        all_results_parallel = []
                        for batch_results in pool.imap_unordered(self.run_parallel_macro_reps, parallel_args):
                            all_results_parallel.extend(batch_results)
                        
                        # Collect results
                        rmse_values = [r['rmse'] for r in all_results_parallel]
                        sparsity_values = [r['actual_sparsity'] for r in all_results_parallel]
                        nonzero_params_values = [r['nonzero_params'] for r in all_results_parallel]
                        computation_times = [r['computation_time'] for r in all_results_parallel]
                        
                        # Calculate statistics
                        mean_rmse = float(np.mean(rmse_values))
                        std_rmse = float(np.std(rmse_values))
                        mean_time = float(np.mean(computation_times))
                        std_time = float(np.std(computation_times))
                        
                        experiment_config['statistics'] = {
                            'mean_rmse': mean_rmse,
                            'std_rmse': std_rmse,
                            'rmse_ci_lower': float(mean_rmse - 1.96 * std_rmse / np.sqrt(self.macro_replications)),
                            'rmse_ci_upper': float(mean_rmse + 1.96 * std_rmse / np.sqrt(self.macro_replications)),
                            'mean_sparsity': float(np.mean(sparsity_values)),
                            'mean_nonzero_params': int(np.mean(nonzero_params_values)),
                            'mean_computation_time': mean_time,
                            'std_computation_time': std_time,
                            'total_configuration_time': float(time.time() - config_start_time)
                        }
                        
                        all_results['experiments'].append(experiment_config)
                        
                        # Print current experiment summary
                        print(f"\nExperiment (n={n}, d={d}, xi={xi}):")
                        print(f"  Mean RMSE: {mean_rmse:.6f} ± {std_rmse:.6f}")
                        print(f"  Mean Sparsity: {experiment_config['statistics']['mean_sparsity']:.4f}")
                        print(f"  Mean Non-zero Parameters: {experiment_config['statistics']['mean_nonzero_params']}")
                        print(f"  Mean Computation Time: {mean_time:.2f} ± {std_time:.2f} seconds")
                        print(f"  Total Configuration Time: {experiment_config['statistics']['total_configuration_time']:.2f} seconds")
                        
                        # Save intermediate results
                        self.save_results(all_results)
        
        # Add total program time
        all_results['timing'] = {
            'total_program_time': float(time.time() - total_start_time),
            'timestamp_start': self.timestamp,
            'timestamp_end': datetime.now().strftime("%Y%m%d_%H%M%S")
        }
        
        print(f"\nTotal Program Time: {all_results['timing']['total_program_time']:.2f} seconds")
        return all_results

    def save_results(self, results):
        '''
        Save experimental results to JSON file
        '''
        results_file = os.path.join(self.experiment_dir, 'results.json')
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=4, cls=NumpyEncoder)
        
        print(f"\nResults saved to {results_file}")

if __name__ == "__main__":
    # Run experiments
    runner = ExperimentRunner()
    results = runner.run_experiments()
    
    print("\nExperiments completed. Results saved in:", runner.experiment_dir) 