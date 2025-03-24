import json
import matplotlib.pyplot as plt
import os
import argparse

def create_plots(results, output_dir):
    '''Create visualization plots from results'''
    # Use default matplotlib style
    plt.style.use('default')
    
    # Extract unique dimensions
    dimensions = results['config']['dimensions']
    
    print("\nGenerating plots...")
    print("-------------------")
    
    # Create RMSE vs n plot for each dimension
    for d in dimensions:
        plt.figure(figsize=(12, 8))
        
        # Filter experiments for current dimension
        d_experiments = [exp for exp in results['experiments'] if exp['settings']['d'] == d]
        
        for xi in results['config']['noise_levels']:
            # Get data for current noise level
            xi_data = [(exp['settings']['n'], exp['statistics']['mean_rmse'],
                       exp['statistics']['rmse_ci_lower'], exp['statistics']['rmse_ci_upper'])
                      for exp in d_experiments if exp['settings']['xi'] == xi]
            
            if xi_data:
                n_values, means, ci_lower, ci_upper = zip(*xi_data)
                
                plt.plot(n_values, means, marker='o', label=f'Noise level ξ={xi}')
                plt.fill_between(n_values, ci_lower, ci_upper, alpha=0.2)
        
        plt.xlabel('Number of Design Points (n)')
        plt.ylabel('Root Mean Square Error (RMSE)')
        plt.title(f'RMSE vs Sample Size (d={d})')
        plt.legend()
        plt.grid(True)
        plt.yscale('log')
        plt.xscale('log')
        
        filename = os.path.join(output_dir, f'rmse_d{d}.png')
        plt.savefig(filename)
        plt.close()
        print(f"- Dimension {d} RMSE plot saved as: {filename}")
        
        # Create computation time vs n plot
        plt.figure(figsize=(12, 8))
        
        for xi in results['config']['noise_levels']:
            # Get time data for current noise level
            time_data = [(exp['settings']['n'], exp['statistics']['mean_computation_time'],
                         exp['statistics']['mean_computation_time'] - exp['statistics']['std_computation_time'],
                         exp['statistics']['mean_computation_time'] + exp['statistics']['std_computation_time'])
                        for exp in d_experiments if exp['settings']['xi'] == xi]
            
            if time_data:
                n_values, means, lower, upper = zip(*time_data)
                
                plt.plot(n_values, means, marker='o', label=f'Noise level ξ={xi}')
                plt.fill_between(n_values, lower, upper, alpha=0.2)
        
        plt.xlabel('Number of Design Points (n)')
        plt.ylabel('Computation Time (seconds)')
        plt.title(f'Computation Time vs Sample Size (d={d})')
        plt.legend()
        plt.grid(True)
        plt.yscale('log')
        plt.xscale('log')
        
        filename = os.path.join(output_dir, f'time_vs_n_d{d}.png')
        plt.savefig(filename)
        plt.close()
        print(f"- Dimension {d} time vs n plot saved as: {filename}")
        
        # Create computation time vs parameters plot
        plt.figure(figsize=(12, 8))
        
        for xi in results['config']['noise_levels']:
            # Get time data vs parameters for current noise level
            param_time_data = [(exp['settings']['width'] * exp['settings']['width'] * (exp['settings']['depth'] - 1) + 
                               exp['settings']['d'] * exp['settings']['width'] + exp['settings']['width'],
                               exp['statistics']['mean_computation_time'],
                               exp['statistics']['mean_computation_time'] - exp['statistics']['std_computation_time'],
                               exp['statistics']['mean_computation_time'] + exp['statistics']['std_computation_time'])
                              for exp in d_experiments if exp['settings']['xi'] == xi]
            
            if param_time_data:
                params, means, lower, upper = zip(*param_time_data)
                
                plt.plot(params, means, marker='o', label=f'Noise level ξ={xi}')
                plt.fill_between(params, lower, upper, alpha=0.2)
        
        plt.xlabel('Total Number of Parameters')
        plt.ylabel('Computation Time (seconds)')
        plt.title(f'Computation Time vs Number of Parameters (d={d})')
        plt.legend()
        plt.grid(True)
        plt.yscale('log')
        plt.xscale('log')
        
        filename = os.path.join(output_dir, f'time_vs_params_d{d}.png')
        plt.savefig(filename)
        plt.close()
        print(f"- Dimension {d} time vs parameters plot saved as: {filename}")
    
    print("\nAll plots have been generated successfully.")

def print_summary(results):
    '''Print summary statistics from results'''
    print("\nExperiment Summary:")
    print("------------------")
    
    # Sort experiments by dimension, sample size, and noise level
    sorted_experiments = sorted(
        results['experiments'],
        key=lambda x: (x['settings']['d'], x['settings']['n'], x['settings']['xi'])
    )
    
    current_d = None
    for exp in sorted_experiments:
        settings = exp['settings']
        stats = exp['statistics']
        
        # Print dimension header if it changed
        if current_d != settings['d']:
            current_d = settings['d']
            print(f"\nDimension d = {settings['d']}:")
            print("-" * 20)
        
        print(f"\nn = {settings['n']}, ξ = {settings['xi']}")
        print(f"Network: depth = {settings['depth']}, width = {settings['width']}")
        print(f"Sparsity: target = {settings['target_sparsity']:.4f}, achieved = {stats['mean_sparsity']:.4f}")
        print(f"Non-zero parameters: {stats['mean_nonzero_params']}")
        print(f"RMSE: {stats['mean_rmse']:.6f} ± {stats['std_rmse']:.6f}")
        print(f"95% CI: [{stats['rmse_ci_lower']:.6f}, {stats['rmse_ci_upper']:.6f}]")

def main():
    parser = argparse.ArgumentParser(description='Create plots from experiment results')
    parser.add_argument('results_file', help='Path to the results JSON file')
    parser.add_argument('--output-dir', default='plots', help='Directory to save plots')
    parser.add_argument('--show-summary', action='store_true', help='Print summary statistics')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load results
    with open(args.results_file, 'r') as f:
        results = json.load(f)
    
    # Create plots
    create_plots(results, args.output_dir)
    print(f"\nPlots saved in: {args.output_dir}")
    
    # Print summary if requested
    if args.show_summary:
        print_summary(results)

if __name__ == "__main__":
    main() 