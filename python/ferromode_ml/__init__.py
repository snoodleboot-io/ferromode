"""Ferromode ML: Machine Learning integration for Ferromode.

This package provides differentiable EMD implementations for deep learning
frameworks including TensorFlow/Keras and PyTorch, plus learnable boundary
prediction modules for task-specific decomposition adaptation.

Modules:
    rust_bridge: Low-level PyO3 bindings to Rust EMD functions
    keras_emd: Keras layer for differentiable EMD (TensorFlow)
    torch_emd: PyTorch module for differentiable EMD
    learnable_boundary: Neural network for boundary prediction

Classes:
    DifferentiableEMDLayer: Keras layer for EMD
    DifferentiableEMD: PyTorch module for EMD
    DifferentiableEMDFunction: PyTorch custom autograd function
    LearnableBoundaryPredictor: PyTorch module for boundary prediction
    PretrainBoundaryPredictorConfig: Configuration for pre-training

Functions:
    call_rust_emd_forward: Call Rust EMD forward pass
    call_rust_emd_backward: Call Rust EMD backward pass
    pretrain_boundary_predictor: Pre-train boundary predictor

Example:
    >>> # TensorFlow/Keras
    >>> from ferromode_ml import DifferentiableEMDLayer
    >>> import tensorflow as tf
    >>> layer = DifferentiableEMDLayer(max_imfs=5)
    >>> inputs = tf.random.normal((32, 256))
    >>> outputs = layer(inputs)  # (32, num_imfs, 256)

    >>> # PyTorch
    >>> from ferromode_ml import DifferentiableEMD
    >>> import torch
    >>> module = DifferentiableEMD(max_imfs=5)
    >>> inputs = torch.randn(32, 256)
    >>> outputs = module(inputs)  # (32, num_imfs, 256)

    >>> # Learnable Boundary Predictor
    >>> from ferromode_ml import LearnableBoundaryPredictor
    >>> predictor = LearnableBoundaryPredictor(context_length=10, output_length=15)
    >>> context = torch.randn(32, 10)
    >>> boundary = predictor(context)  # (32, 15)

Version: 0.1.0
License: MIT OR Apache-2.0
"""

__version__ = "0.1.0"
__author__ = "Ferromode Contributors"

# Import main classes
try:
    from .keras_emd import DifferentiableEMDLayer
except ImportError:
    DifferentiableEMDLayer = None  # type: ignore

try:
    from .torch_emd import DifferentiableEMD, DifferentiableEMDFunction
except ImportError:
    DifferentiableEMD = None  # type: ignore
    DifferentiableEMDFunction = None  # type: ignore

try:
    from .learnable_boundary import (
        LearnableBoundaryPredictor,
        PretrainBoundaryPredictorConfig,
        pretrain_boundary_predictor,
    )
except ImportError:
    LearnableBoundaryPredictor = None  # type: ignore
    PretrainBoundaryPredictorConfig = None  # type: ignore
    pretrain_boundary_predictor = None  # type: ignore

# Import bridge functions
from .rust_bridge import call_rust_emd_backward, call_rust_emd_forward

__all__ = [
    "DifferentiableEMDLayer",  # Keras
    "DifferentiableEMD",  # PyTorch
    "DifferentiableEMDFunction",  # PyTorch autograd
    "LearnableBoundaryPredictor",  # Learnable boundary
    "PretrainBoundaryPredictorConfig",  # Configuration
    "pretrain_boundary_predictor",  # Pre-training function
    "call_rust_emd_forward",  # Bridge function
    "call_rust_emd_backward",  # Bridge function
]
