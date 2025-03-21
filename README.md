# Simple Neural Network Implementation

A PyTorch implementation of a simple neural network that supports GPU acceleration using MPS (Metal Performance Shaders) on Mac devices.

## Features

- Two-layer neural network with ReLU activation function
- GPU acceleration support for Apple Silicon (M1/M2/M3)
- Training and testing time statistics
- Automatic device selection (MPS/CUDA/CPU)

## Requirements

- Python 3.x
- PyTorch >= 2.1.0
- NumPy >= 1.21.0

## Installation

Install the required packages:

```bash
pip install -r requirements.txt
```

## Usage

Run the Python file directly:

```bash
python toy_neural_network.py
```

## Output Example

The program will display:
- Device type being used
- Loss value and time cost every 10 epochs during training
- Total training time
- Test set performance and prediction time

## Project Structure

- `toy_neural_network.py`: Main program file
- `requirements.txt`: List of dependencies

## Performance Notes

- For Apple Silicon Mac users: The code automatically uses MPS backend for GPU acceleration
- For Intel Mac users: The code will fall back to CPU computation
- Training and prediction times are measured and displayed for performance monitoring 