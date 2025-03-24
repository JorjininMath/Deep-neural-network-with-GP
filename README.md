# Deep Neural Networks Combined with Gaussian Processes

This project implements a nonparametric experimental framework combining deep neural networks, designed to study neural network performance across different dimensions and sample sizes.

## Project Structure

```
.
├── README.md
├── main.py               # Main entry point
├── config.py            # Configuration management
├── experiment_runner.py  # Experiment execution logic
├── network_estimator.py # Neural network model implementation
├── data_generator.py    # Data generator
├── plotter.py          # Results visualization and plotting
└── plot_results.py     # Legacy plotting script (for backward compatibility)
```

## Key Features

- Support for multiple input dimensions (d=20, 100, 500)
- Adaptive network architecture design (theory-based)
- Sparse training process
- GPU acceleration support (Mac MPS)
- Parallel experiment execution
- Automated result collection and visualization
- Modular configuration management
- Flexible plotting system

## Requirements

- Python 3.8+
- PyTorch 2.0+
- NumPy
- Pandas
- Matplotlib
- Seaborn
- tqdm

## Installation

1. Clone the repository:
```bash
git clone [repository-url]
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running Experiments

```bash
# Enable MPS backend (Mac GPU support)
export PYTORCH_ENABLE_MPS_FALLBACK=1
python main.py
```

### Experiment Configuration

Parameters are configured in `config.py`:

- Dimensions (d): [20, 100, 500]
- Sample sizes (n): 
  - d=20: [512, 1024, 2048, 4096]
  - d=100,500: [2048, 4096, 8192, 16384]
- Noise levels (ξ): [0.1, 10.0]
- Replications per point (m): 10
- Macro replications (R): 30

### Network Architecture

Network architecture parameters are automatically determined based on theoretical calculations:

- φ(n) = n^(-2β/(2β+1))
- Depth = max(1, int(n * φ(n)))
- Width = int(2 * n * φ(n))
- Sparsity is calculated based on total parameters and target non-zero parameters

### Output

Experiment results are saved in `experiments/experiment_[timestamp]` directory:
- `results.json`: Contains all experiment configurations and results
- `data/`: Stores generated data
- Computation time statistics
- RMSE evaluation results

## Result Analysis

Use `plotter.py` to generate visualizations:
```bash
python main.py --plot path/to/results.json
```

Generated plots include:
- RMSE vs Sample Size
- Computation Time vs Sample Size
- Computation Time vs Total Parameters

## Important Notes

1. GPU acceleration recommended for large-scale experiments
2. Ensure sufficient storage space for experiment results
3. Performance can be optimized by adjusting the number of parallel processes
4. Configuration can be modified through `config.py` without changing the main code
5. The legacy `plot_results.py` is maintained for backward compatibility

## Contributing

Contributions are welcome. Before submitting code, please:
1. Run the complete test suite
2. Update relevant documentation
3. Follow existing code style

## License

[Your License Type] 