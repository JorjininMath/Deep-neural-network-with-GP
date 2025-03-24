import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

class SparseNeuralNetwork(nn.Module):
    '''
    Sparse neural network implementation with pruning capabilities
    
    Attributes
    ----------
    input_dim : int
        Input dimension
    hidden_layers : list
        List of integers specifying the number of nodes in each hidden layer
    sparsity : float
        Target sparsity ratio (between 0 and 1)
    output_bound : float
        Bound for the output values
    masks : list
        List of binary masks for each layer's weights
    '''
    def __init__(self, input_dim, hidden_layers, sparsity=0.9, output_bound=1.0):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.sparsity = sparsity
        self.output_bound = output_bound
        
        # Create layers
        layer_sizes = [input_dim] + hidden_layers + [1]
        self.layers = nn.ModuleList([
            nn.Linear(layer_sizes[i], layer_sizes[i+1])
            for i in range(len(layer_sizes)-1)
        ])
        
        # Initialize masks for pruning
        self.masks = [torch.ones_like(layer.weight.data)
                     for layer in self.layers]
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        '''Initialize network weights'''
        for layer in self.layers:
            nn.init.kaiming_normal_(layer.weight)
            nn.init.zeros_(layer.bias)
    
    def forward(self, x):
        '''Forward pass with ReLU activation and output scaling'''
        for i, layer in enumerate(self.layers[:-1]):
            x = torch.relu(layer(x))
        x = self.layers[-1](x)  # Linear output layer
        return self.output_bound * torch.tanh(x)  # Scale output
    
    def prune_weights(self, prune_ratio):
        '''
        Prune network weights based on magnitude
        
        Parameters
        ----------
        prune_ratio : float
            Ratio of weights to prune in this iteration
        '''
        for i, layer in enumerate(self.layers):
            # Get current weights
            weights = layer.weight.data.abs()
            
            # Calculate threshold
            threshold = torch.quantile(
                weights[self.masks[i] == 1],
                prune_ratio
            )
            
            # Update mask
            self.masks[i] = (weights > threshold).float()
            
            # Apply mask
            layer.weight.data *= self.masks[i]
    
    def apply_masks(self):
        '''Apply stored masks to weights'''
        for i, layer in enumerate(self.layers):
            layer.weight.data *= self.masks[i]
    
    def count_parameters(self):
        '''
        Count total and non-zero parameters
        
        Returns
        -------
        tuple
            (total_parameters, nonzero_parameters)
        '''
        total_params = sum(p.numel() for p in self.parameters())
        nonzero_params = sum(
            (p != 0).sum().item() for p in self.parameters()
        )
        return total_params, nonzero_params

class NetworkEstimator:
    '''
    Neural network estimator class for handling training and prediction
    with GPU support for Mac
    
    Attributes
    ----------
    model : SparseNeuralNetwork
        Neural network model
    prune_iterations : int
        Number of pruning iterations
    retrain_epochs : int
        Number of retraining epochs after each pruning
    device : torch.device
        Device to run the model on (CPU or MPS for Mac GPU)
    batch_size : int
        Size of mini-batches for training
    patience : int
        Number of epochs to wait before early stopping
    '''
    def __init__(self, input_dim, hidden_layers, sparsity=0.9, output_bound=1.0,
                 prune_iterations=10, retrain_epochs=5, batch_size=32, patience=10):
        # Set device
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        self.model = SparseNeuralNetwork(
            input_dim=input_dim,
            hidden_layers=hidden_layers,
            sparsity=sparsity,
            output_bound=output_bound
        ).to(self.device)
        
        self.prune_iterations = prune_iterations
        self.retrain_epochs = retrain_epochs
        self.batch_size = batch_size
        self.patience = patience
        self.best_loss = float('inf')
        self.best_model_state = None

    def process_data(self, X, y, m):
        '''
        Process data with replicated observations
        
        Parameters
        ----------
        X : numpy.ndarray
            Input features, shape: (n*m, d)
        y : numpy.ndarray
            Noisy output values, shape: (n*m,)
        m : int
            Number of replications per design point
        
        Returns
        -------
        tuple
            (X_processed, y_processed) where y is the sample mean at each point
        '''
        n = X.shape[0] // m
        X_processed = X[::m]  # Take first X value from each group
        y_processed = np.zeros(n)
        
        # Calculate sample mean for each design point
        for i in range(n):
            y_processed[i] = np.mean(y[i*m:(i+1)*m])
        
        return X_processed, y_processed

    def create_data_loaders(self, X, y, val_split=0.2):
        '''
        Create training and validation data loaders
        
        Parameters
        ----------
        X : torch.Tensor
            Input features
        y : torch.Tensor
            Output values
        val_split : float
            Fraction of data to use for validation
            
        Returns
        -------
        tuple
            (train_loader, val_loader)
        '''
        dataset_size = len(X)
        indices = torch.randperm(dataset_size)
        val_size = int(val_split * dataset_size)
        
        train_indices = indices[val_size:]
        val_indices = indices[:val_size]
        
        train_dataset = TensorDataset(X[train_indices], y[train_indices])
        val_dataset = TensorDataset(X[val_indices], y[val_indices])
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size)
        
        return train_loader, val_loader

    def evaluate(self, dataloader, criterion):
        '''
        Evaluate the model on the given dataloader
        
        Parameters
        ----------
        dataloader : DataLoader
            Data loader to evaluate on
        criterion : nn.Module
            Loss function
            
        Returns
        -------
        float
            Average loss
        '''
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch_X, batch_y in dataloader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                y_pred = self.model(batch_X)
                loss = criterion(y_pred, batch_y)
                total_loss += loss.item()
                num_batches += 1
        
        self.model.train()
        return total_loss / num_batches

    def fit(self, X, y, m=1, epochs=100, val_split=0.2):
        '''
        Train the model using mini-batch processing with validation
        
        Parameters
        ----------
        X : numpy.ndarray
            Input features
        y : numpy.ndarray
            Output values
        m : int
            Number of replications per design point
        epochs : int
            Number of training epochs
        val_split : float
            Fraction of data to use for validation
        '''
        # Process replicated observations
        if m > 1:
            X, y = self.process_data(X, y, m)
        
        # Convert to tensors
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)
        
        # Create data loaders
        train_loader, val_loader = self.create_data_loaders(X, y, val_split)
        
        # Initialize optimizer, scheduler, and loss function
        optimizer = torch.optim.Adam(self.model.parameters())
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=False)
        criterion = nn.MSELoss()
        
        # Early stopping variables
        patience_counter = 0
        best_val_loss = float('inf')
        
        # Initial training with mini-batches
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            total_train_loss = 0
            num_train_batches = 0
            
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                optimizer.zero_grad()
                y_pred = self.model(batch_X)
                loss = criterion(y_pred, batch_y)
                loss.backward()
                optimizer.step()
                
                total_train_loss += loss.item()
                num_train_batches += 1
            
            # Validation phase
            val_loss = self.evaluate(val_loader, criterion)
            
            # Update learning rate
            scheduler.step(val_loss)
            
            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.best_model_state = self.model.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= self.patience:
                break
        
        # Load best model before pruning
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
        
        # Calculate pruning ratio for each iteration
        prune_ratio_per_iter = 1 - (1 - self.model.sparsity) ** (1 / self.prune_iterations)
        
        # Iterative pruning with mini-batches
        for iteration in range(self.prune_iterations):
            # Perform pruning
            self.model.prune_weights(prune_ratio_per_iter)
            
            # Reset early stopping variables
            patience_counter = 0
            best_val_loss = float('inf')
            
            # Retrain with mini-batches
            for epoch in range(self.retrain_epochs):
                # Training phase
                self.model.train()
                total_train_loss = 0
                num_train_batches = 0
                
                for batch_X, batch_y in train_loader:
                    batch_X = batch_X.to(self.device)
                    batch_y = batch_y.to(self.device)
                    
                    optimizer.zero_grad()
                    y_pred = self.model(batch_X)
                    loss = criterion(y_pred, batch_y)
                    loss.backward()
                    optimizer.step()
                    self.model.apply_masks()
                    
                    total_train_loss += loss.item()
                    num_train_batches += 1
                
                # Validation phase
                val_loss = self.evaluate(val_loader, criterion)
                
                # Update learning rate
                scheduler.step(val_loss)
                
                # Early stopping check
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.best_model_state = self.model.state_dict()
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= self.patience:
                    break
            
            # Load best model after retraining
            if self.best_model_state is not None:
                self.model.load_state_dict(self.best_model_state)
            
            # Check sparsity
            total_params, nonzero_params = self.model.count_parameters()
            current_sparsity = 1 - (nonzero_params / total_params)
            
            # Stop if target sparsity is reached
            if current_sparsity >= self.model.sparsity:
                break

    def predict(self, X, batch_size=None):
        '''
        Make predictions for new data using mini-batches
        
        Parameters
        ----------
        X : numpy.ndarray
            Input features
        batch_size : int, optional
            Batch size for prediction. If None, uses training batch size
        
        Returns
        -------
        numpy.ndarray
            Predicted values
        '''
        if batch_size is None:
            batch_size = self.batch_size
            
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32)
        
        # Create dataloader for prediction
        dataset = TensorDataset(X)
        dataloader = DataLoader(dataset, batch_size=batch_size)
        
        predictions = []
        self.model.eval()  # Set model to evaluation mode
        
        with torch.no_grad():
            for (batch_X,) in dataloader:
                batch_X = batch_X.to(self.device)
                batch_pred = self.model(batch_X)
                predictions.append(batch_pred.cpu())
        
        self.model.train()  # Set model back to training mode
        return torch.cat(predictions, dim=0).numpy()

    def get_model_info(self):
        '''
        Get model information
        
        Returns
        -------
        dict
            Dictionary containing model parameter statistics
        '''
        total_params, nonzero_params = self.model.count_parameters()
        return {
            'total_parameters': total_params,
            'nonzero_parameters': nonzero_params,
            'sparsity_ratio': 1 - nonzero_params/total_params
        } 