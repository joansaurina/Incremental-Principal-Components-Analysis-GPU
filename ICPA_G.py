import torch
from typing import Optional, Tuple, Union
import numpy as np

class IncrementalPCAonGPU:
    def __init__(self, 
                 n_components: Optional[int] = None, *, 
                 whiten: bool = False, 
                 copy: bool = True, 
                 batch_size: Optional[int] = None,
                 use_fp16: bool = False,
                 device: Optional[Union[str, torch.device]] = None):
        """
        Initialize IncrementalPCAonGPU.
        
        Args:
            n_components: Number of components to keep. If None, keep all components.
            whiten: When True, the components_ vectors are scaled to have unit variance.
            copy: If False, X will be overwritten whenever possible.
            batch_size: The number of samples to use for each batch.
            use_fp16: If True, use half precision (float16).
            device: Device to use for computations ('cpu', 'cuda', 'cuda:0', etc.).
                   If None, uses CUDA if available, else CPU.
        """
        self.n_components = n_components
        self.whiten = whiten
        self.copy = copy
        self.batch_size = batch_size
        
        # Device handling
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        # Validate device if CUDA is requested
        if self.device.type.startswith('cuda') and not torch.cuda.is_available():
            raise RuntimeError("CUDA device requested but CUDA is not available. "
                             "Falling back to CPU.")
            self.device = torch.device('cpu')
            
        self.dtype = torch.float16 if use_fp16 and self.device.type.startswith('cuda') else torch.float32
        
        # Initialize attributes
        self.mean_ = None
        self.var_ = None
        self.n_samples_seen_ = 0
        self.components_ = None
        self.singular_values_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.noise_variance_ = None
        
    def _validate_data(self, X: Union[np.ndarray, torch.Tensor], 
                      copy: bool = True) -> torch.Tensor:
        """Validate and convert input data to tensor with proper dtype and device."""
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X)
            
        # If X is not on the correct device or has wrong dtype
        if X.dtype != self.dtype or X.device != self.device:
            X = X.to(dtype=self.dtype, device=self.device)
        elif copy:
            X = X.clone()
            
        return X
    
    def _check_memory(self, X: torch.Tensor) -> bool:
        """Check if there's enough GPU memory for the operation."""
        if self.device.type.startswith('cuda'):
            # Get the correct CUDA device index
            device_idx = 0 if self.device.index is None else self.device.index
            required_memory = X.element_size() * X.nelement() * 3  # Estimate for basic operations
            available_memory = (torch.cuda.get_device_properties(device_idx).total_memory - 
                              torch.cuda.memory_allocated(device_idx))
            return required_memory < available_memory
        return True

    def to(self, device: Union[str, torch.device]) -> 'IncrementalPCAonGPU':
        """
        Move the model to a different device.
        
        Args:
            device: The device to move the model to.
            
        Returns:
            self: The model instance moved to the specified device.
        """
        new_device = torch.device(device)
        
        # Validate device if CUDA is requested
        if new_device.type.startswith('cuda') and not torch.cuda.is_available():
            raise RuntimeError("CUDA device requested but CUDA is not available.")
        
        # Move all tensors to the new device
        if hasattr(self, 'mean_') and self.mean_ is not None:
            self.mean_ = self.mean_.to(new_device)
        if hasattr(self, 'var_') and self.var_ is not None:
            self.var_ = self.var_.to(new_device)
        if hasattr(self, 'components_') and self.components_ is not None:
            self.components_ = self.components_.to(new_device)
        if hasattr(self, 'singular_values_') and self.singular_values_ is not None:
            self.singular_values_ = self.singular_values_.to(new_device)
        if hasattr(self, 'explained_variance_') and self.explained_variance_ is not None:
            self.explained_variance_ = self.explained_variance_.to(new_device)
        if hasattr(self, 'explained_variance_ratio_') and self.explained_variance_ratio_ is not None:
            self.explained_variance_ratio_ = self.explained_variance_ratio_.to(new_device)
            
        self.device = new_device
        return self

    @torch.cuda.amp.autocast()
    def fit_transform(self, X: Union[np.ndarray, torch.Tensor], 
                     check_input: bool = True) -> torch.Tensor:
        """Fit the model with X and apply the dimensionality reduction on X."""
        X = self._validate_data(X) if check_input else X
        n_samples, n_features = X.shape
        
        # Initialize batch_size if not set
        if self.batch_size is None:
            self.batch_size_ = min(5 * n_features, n_samples)
        else:
            self.batch_size_ = self.batch_size
            
        # Process data in batches if needed
        if not self._check_memory(X):
            transformed_data = []
            for start in range(0, n_samples, self.batch_size_):
                end = min(start + self.batch_size_, n_samples)
                X_batch = X[start:end]
                
                # For first batch, do full fit
                if start == 0:
                    batch_transformed = self.partial_fit(X_batch, check_input=False)._transform_batch(X_batch)
                else:
                    batch_transformed = self._transform_batch(X_batch)
                    
                transformed_data.append(batch_transformed)
                torch.cuda.empty_cache() if self.device.type.startswith('cuda') else None
                
            return torch.cat(transformed_data, dim=0)
        else:
            # If we have enough memory, do everything at once
            self.fit(X, check_input=False)
            return self._transform_batch(X)

    def _transform_batch(self, X: torch.Tensor) -> torch.Tensor:
        """Internal method to transform a batch of data."""
        X_transformed = X - self.mean_
        if self.whiten:
            return torch.mm(X_transformed, 
                          (self.components_.T / self.singular_values_.view(-1, 1)))
        return torch.mm(X_transformed, self.components_.T)

    @torch.cuda.amp.autocast()
    def inverse_transform(self, X: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Transform data back to its original space."""
        X = self._validate_data(X)
        
        if not hasattr(self, 'components_'):
            raise AttributeError("Model not fitted yet.")
            
        if self.whiten:
            X_transformed = torch.mm(X, 
                                   (self.components_ * self.singular_values_.view(-1, 1)))
        else:
            X_transformed = torch.mm(X, self.components_)
            
        return X_transformed + self.mean_

    def _cleanup_gpu_memory(self):
        """Clean up GPU memory by moving unused tensors to CPU."""
        if self.device.type.startswith('cuda'):
            torch.cuda.empty_cache()
            
    def __del__(self):
        """Cleanup when object is deleted."""
        self._cleanup_gpu_memory()
