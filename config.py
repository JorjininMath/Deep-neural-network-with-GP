import os
from datetime import datetime
from multiprocessing import cpu_count

class ExperimentConfig:
    """Configuration class for managing all experiment settings"""
    
    def __init__(self, **kwargs):
        # Set environment variable for MPS fallback
        os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
        
        # Base directory for experiments
        self.base_dir = kwargs.get('base_dir', 'experiments')
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_dir = os.path.join(self.base_dir, f'experiment_{self.timestamp}')
        
        # Experiment dimensions and sample sizes
        self.dimensions = kwargs.get('dimensions', [20, 100, 500])
        self.sample_sizes = kwargs.get('sample_sizes', {
            20: [2**k for k in range(9, 13)],      # 512, 1024, 2048, 4096
            100: [2**k for k in range(11, 15)],    # 2048, 4096, 8192, 16384
            500: [2**k for k in range(11, 15)]     # 2048, 4096, 8192, 16384
        })
        
        # Noise and replication settings
        self.noise_levels = kwargs.get('noise_levels', [0.1, 10.0])
        self.replications = kwargs.get('replications', 10)
        self.macro_replications = kwargs.get('macro_replications', 30)
        self.num_processes = kwargs.get('num_processes', max(1, cpu_count() - 1))
        
        # Training parameters
        self.epochs = kwargs.get('epochs', 100)
        self.batch_size = kwargs.get('batch_size', 32)
        self.patience = kwargs.get('patience', 10)
        self.val_split = kwargs.get('val_split', 0.2)
        self.prune_iterations = kwargs.get('prune_iterations', 10)
        self.retrain_epochs = kwargs.get('retrain_epochs', 5)
        
        # Create necessary directories
        self._create_directories()
        
        # Print configuration
        self.print_config()
    
    def _create_directories(self):
        """Create necessary directories for the experiment"""
        os.makedirs(self.experiment_dir, exist_ok=True)
        os.makedirs(os.path.join(self.experiment_dir, 'data'), exist_ok=True)
        os.makedirs(os.path.join(self.experiment_dir, 'figures'), exist_ok=True)
    
    def print_config(self):
        """Print current configuration settings"""
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
        print("\nTraining Parameters:")
        print(f"Epochs: {self.epochs}")
        print(f"Batch size: {self.batch_size}")
        print(f"Patience: {self.patience}")
        print(f"Validation split: {self.val_split}")
        print(f"Pruning iterations: {self.prune_iterations}")
        print(f"Retrain epochs: {self.retrain_epochs}")
        print("-------------------------\n")
    
    def get_experiment_paths(self):
        """Get dictionary of important paths"""
        return {
            'base_dir': self.base_dir,
            'experiment_dir': self.experiment_dir,
            'data_dir': os.path.join(self.experiment_dir, 'data'),
            'figures_dir': os.path.join(self.experiment_dir, 'figures'),
            'results_file': os.path.join(self.experiment_dir, 'results.json')
        } 