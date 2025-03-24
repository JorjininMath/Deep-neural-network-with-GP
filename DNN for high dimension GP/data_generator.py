import numpy as np
import os
import json
from datetime import datetime

class DataGenerator:
    '''
    Data generator class for generating and storing experimental data
    
    Attributes
    ----------
    save_dir : str
        Directory for saving data
    '''
    def __init__(self, save_dir='data'):
        self.save_dir = save_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    def griewank_function(self, X):
        '''
        Griewank function implementation
        
        Parameters
        ----------
        X : numpy.ndarray
            Input data, shape: (n, d)
        
        Returns
        -------
        numpy.ndarray
            Function values
        '''
        sum_part = np.sum(X**2, axis=1) / 4000
        prod_part = np.prod(np.cos(X / np.sqrt(np.arange(1, X.shape[1]+1))), axis=1)
        return sum_part - prod_part + 1

    def add_noise(self, y, xi):
        '''
        Add heteroscedastic noise
        
        Parameters
        ----------
        y : numpy.ndarray
            True function values
        xi : float
            Noise level
        
        Returns
        -------
        numpy.ndarray
            Noisy observations
        '''
        noise = np.random.normal(0, np.sqrt(xi * np.abs(y)))
        return y + noise

    def generate_data(self, n, d, m, xi, function='griewank', save=True):
        '''
        Generate experimental data
        
        Parameters
        ----------
        n : int
            Number of design points
        d : int
            Input dimension
        m : int
            Number of replications per design point
        xi : float
            Noise level
        function : str
            Name of the function to use
        save : bool
            Whether to save the data
        
        Returns
        -------
        tuple
            (X, y) generated data
        '''
        # Generate design points
        X = 8 * np.random.rand(n, d) - 4  # [-4, 4]^d
        
        # Replicate m times
        X_repeated = np.repeat(X, m, axis=0)
        
        # Calculate true function values
        if function == 'griewank':
            y_true = self.griewank_function(X)
        else:
            raise ValueError(f"Unknown function: {function}")
        
        # Replicate true values m times and add noise
        y_true_repeated = np.repeat(y_true, m)
        y_noisy = self.add_noise(y_true_repeated, xi)
        
        if save:
            self.save_data(X_repeated, y_noisy, n, d, m, xi, function)
        
        return X_repeated, y_noisy

    def save_data(self, X, y, n, d, m, xi, function):
        '''
        Save data and metadata
        
        Parameters
        ----------
        X : numpy.ndarray
            Input data
        y : numpy.ndarray
            Output data
        n : int
            Number of design points
        d : int
            Input dimension
        m : int
            Number of replications
        xi : float
            Noise level
        function : str
            Name of the function used
        '''
        # Create timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create data directory
        data_dir = os.path.join(self.save_dir, f"{function}_{timestamp}")
        os.makedirs(data_dir)
        
        # Save data
        np.save(os.path.join(data_dir, 'X.npy'), X)
        np.save(os.path.join(data_dir, 'y.npy'), y)
        
        # Convert numpy types to Python native types
        metadata = {
            'function': str(function),
            'n': int(n),
            'd': int(d),
            'm': int(m),
            'xi': float(xi),
            'timestamp': str(timestamp)
        }
        
        with open(os.path.join(data_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=4)

    def load_data(self, data_dir):
        '''
        Load data and metadata
        
        Parameters
        ----------
        data_dir : str
            Path to data directory
        
        Returns
        -------
        tuple
            (X, y, metadata) loaded data and metadata
        '''
        X = np.load(os.path.join(data_dir, 'X.npy'))
        y = np.load(os.path.join(data_dir, 'y.npy'))
        
        with open(os.path.join(data_dir, 'metadata.json'), 'r') as f:
            metadata = json.load(f)
        
        return X, y, metadata

    def list_datasets(self):
        '''
        List all available datasets
        
        Returns
        -------
        list
            List of dataset directories
        '''
        datasets = []
        for dirname in os.listdir(self.save_dir):
            if os.path.isdir(os.path.join(self.save_dir, dirname)):
                datasets.append(dirname)
        return datasets 