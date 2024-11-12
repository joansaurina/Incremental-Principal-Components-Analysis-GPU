# Incremental Principal Components Analysis GPU

GPU-optimized Incremental Principal Component Analysis (IPCA) for large datasets. This repo accelerates IPCA with GPU processing, enabling efficient dimensionality reduction on data too large for memory. Perfect for high-dimensional data in machine learning, computer vision, NLP, and scientific computing, offering faster, scalable analysis.

## Features

- GPU-accelerated computations with PyTorch backend
- Batch processing for large-scale datasets
- Memory-efficient incremental learning
- Automatic device management (CPU/GPU)
- FP16 support for increased performance
- Multiple GPU support
- Whitening transformation support

## Installation

```bash
pip install torch  # Required dependency
pip install numpy  # Required dependency
```

## Usage

### Basic Example

```python
from ipca_gpu import IncrementalPCAonGPU
import numpy as np

# Initialize IPCA
ipca = IncrementalPCAonGPU(
    n_components=2,
    batch_size=1000,
    device='cuda'  # or 'cpu', 'cuda:1', etc.
)

# Fit and transform
X_transformed = ipca.fit_transform(X)

# Transform new data
X_new_transformed = ipca.transform(X_new)

# Inverse transform
X_reconstructed = ipca.inverse_transform(X_transformed)
```

### Advanced Usage

```python
# Enable FP16 for faster processing
ipca = IncrementalPCAonGPU(
    n_components=2,
    use_fp16=True,
    device='cuda:0'
)

# Process data incrementally
ipca.partial_fit(batch1)
ipca.partial_fit(batch2)

# Move between devices
ipca.to('cuda:1')
```

## API Reference

### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| n_components | int | None | Number of components to keep |
| whiten | bool | False | Enable whitening transformation |
| batch_size | int | None | Samples per batch (default: 5 * n_features) |
| use_fp16 | bool | False | Enable half precision |
| device | str | None | Computation device |

### Key Methods

- `fit(X)`: Fit the model
- `transform(X)`: Apply dimensionality reduction
- `fit_transform(X)`: Fit and transform in one step
- `inverse_transform(X)`: Reverse transformation
- `partial_fit(X)`: Incremental fitting
- `to(device)`: Change computation device

## Requirements

- Python 3.6+
- PyTorch 1.7+
- NumPy 1.19+

## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

## Citation

```bibtex
@software{ipca_gpu,
  title={Incremental Principal Components Analysis GPU},
  author={Joan Saurina Ricós},
  year={2024},
  url={https://github.com/yourusername/ipca-gpu}
}
```
