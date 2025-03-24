import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os

class ExperimentPlotter:
    """Class for plotting experiment results"""
    
    def __init__(self, config):
        self.config = config
        self.paths = config.get_experiment_paths()
        self.results = None
        
        # Set plot style
        plt.style.use('seaborn')
        sns.set_palette("husl")
    
    def load_results(self, results_file=None):
        """Load results from JSON file"""
        if results_file is None:
            results_file = self.paths['results_file']
        
        with open(results_file, 'r') as f:
            self.results = json.load(f)
    
    def plot_rmse_vs_n(self):
        """Plot RMSE vs sample size for each dimension"""
        if self.results is None:
            raise ValueError("No results loaded. Call load_results() first.")
        
        for d in self.config.dimensions:
            plt.figure(figsize=(10, 6))
            
            for xi in self.config.noise_levels:
                data = [(exp['settings']['n'], 
                        exp['statistics']['mean_rmse'],
                        exp['statistics']['std_rmse'])
                       for exp in self.results['experiments']
                       if exp['settings']['d'] == d and exp['settings']['xi'] == xi]
                
                n_values, means, stds = zip(*data)
                plt.errorbar(n_values, means, yerr=stds, 
                           label=f'ξ={xi}', marker='o', capsize=5)
            
            plt.xscale('log')
            plt.yscale('log')
            plt.xlabel('Number of Design Points (n)')
            plt.ylabel('RMSE')
            plt.title(f'RMSE vs Sample Size (d={d})')
            plt.legend()
            plt.grid(True)
            
            # Save figure
            plt.savefig(os.path.join(self.paths['figures_dir'], f'rmse_vs_n_d{d}.png'))
            plt.close()
    
    def plot_time_vs_n(self):
        """Plot computation time vs sample size for each dimension"""
        if self.results is None:
            raise ValueError("No results loaded. Call load_results() first.")
        
        for d in self.config.dimensions:
            plt.figure(figsize=(10, 6))
            
            for xi in self.config.noise_levels:
                data = [(exp['settings']['n'], 
                        exp['statistics']['mean_computation_time'],
                        exp['statistics']['std_computation_time'])
                       for exp in self.results['experiments']
                       if exp['settings']['d'] == d and exp['settings']['xi'] == xi]
                
                n_values, means, stds = zip(*data)
                plt.errorbar(n_values, means, yerr=stds, 
                           label=f'ξ={xi}', marker='o', capsize=5)
            
            plt.xscale('log')
            plt.yscale('log')
            plt.xlabel('Number of Design Points (n)')
            plt.ylabel('Computation Time (seconds)')
            plt.title(f'Computation Time vs Sample Size (d={d})')
            plt.legend()
            plt.grid(True)
            
            # Save figure
            plt.savefig(os.path.join(self.paths['figures_dir'], f'time_vs_n_d{d}.png'))
            plt.close()
    
    def plot_time_vs_params(self):
        """Plot computation time vs total parameters for each dimension"""
        if self.results is None:
            raise ValueError("No results loaded. Call load_results() first.")
        
        for d in self.config.dimensions:
            plt.figure(figsize=(10, 6))
            
            for xi in self.config.noise_levels:
                data = [(exp['settings']['n'], 
                        exp['statistics']['mean_computation_time'],
                        exp['statistics']['std_computation_time'],
                        exp['settings']['total_params'])
                       for exp in self.results['experiments']
                       if exp['settings']['d'] == d and exp['settings']['xi'] == xi]
                
                n_values, means, stds, params = zip(*data)
                plt.errorbar(params, means, yerr=stds, 
                           label=f'ξ={xi}', marker='o', capsize=5)
            
            plt.xscale('log')
            plt.yscale('log')
            plt.xlabel('Total Number of Parameters')
            plt.ylabel('Computation Time (seconds)')
            plt.title(f'Computation Time vs Parameters (d={d})')
            plt.legend()
            plt.grid(True)
            
            # Save figure
            plt.savefig(os.path.join(self.paths['figures_dir'], f'time_vs_params_d{d}.png'))
            plt.close()
    
    def plot_all(self):
        """Generate all plots"""
        self.plot_rmse_vs_n()
        self.plot_time_vs_n()
        self.plot_time_vs_params() 