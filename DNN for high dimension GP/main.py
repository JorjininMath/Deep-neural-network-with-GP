from config import ExperimentConfig
from experiment_runner import ExperimentRunner
from plotter import ExperimentPlotter
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Run DNN experiments with GP')
    
    # Add arguments for customizing experiment settings
    parser.add_argument('--dimensions', nargs='+', type=int, default=[20, 100, 500],
                      help='Input dimensions to test')
    parser.add_argument('--macro_reps', type=int, default=30,
                      help='Number of macro replications')
    parser.add_argument('--noise_levels', nargs='+', type=float, default=[0.1, 10.0],
                      help='Noise levels to test')
    parser.add_argument('--epochs', type=int, default=100,
                      help='Maximum training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                      help='Training batch size')
    
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Create configuration with custom settings
    config = ExperimentConfig(
        dimensions=args.dimensions,
        macro_replications=args.macro_reps,
        noise_levels=args.noise_levels,
        epochs=args.epochs,
        batch_size=args.batch_size
    )
    
    # Create and run experiments
    runner = ExperimentRunner(config)
    results = runner.run_experiments()
    
    # Create plotter and generate figures
    plotter = ExperimentPlotter(config)
    plotter.load_results()
    plotter.plot_all()
    
    print(f"\nExperiments completed. Results and figures saved in: {config.experiment_dir}")

if __name__ == "__main__":
    main() 