"""
Configuration parameters for the Smooth Nested Simulation project.
"""

import torch

# Device configuration
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Simulation parameters
DIMENSIONS = [20, 50, 100]  # Input dimensions to test
BUDGET_MULTIPLIERS = [1000, 5000, 10000]  # Total simulation budget multipliers
NOISE_LEVELS = [0.1, 1.0]  # Noise levels for the inner simulation

# Kernel parameters
KERNEL_TYPES = {
    'rbf': {
        'name': 'RBF',
        'params': {
            'length_scale': 1.0,
            'length_scale_bounds': (1e-1, 1e1)
        }
    },
    'polynomial': {
        'name': 'Polynomial',
        'params': {
            'degree': 3,
            'coef0': 1.0
        }
    }
}

# Neural Network parameters
NN_CONFIG = {
    'hidden_layers': [128, 64, 32],
    'activation': 'relu',
    'dropout_rate': 0.1,
    'learning_rate': 0.001,
    'batch_size': 32,
    'epochs': 100,
    'early_stopping_patience': 10
}

# Training parameters
TRAIN_CONFIG = {
    'validation_split': 0.2,
    'test_split': 0.1,
    'random_seed': 42
}

# Experiment settings
EXPERIMENT_CONFIG = {
    'n_replications': 30,  # Number of replications for each configuration
    'save_frequency': 5,   # Save results every n epochs
    'verbose': True       # Print detailed information during training
}

# File paths
PATHS = {
    'data': '../data',
    'results': '../results',
    'models': '../results/models',
    'figures': '../results/figures'
}

# Create required directories if they don't exist
import os
for path in PATHS.values():
    os.makedirs(path, exist_ok=True)
