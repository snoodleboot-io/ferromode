"""PyO3 FFI bridge for accessing Rust EMD differentiation functions.

This module provides low-level access to the Rust differentiable EMD
implementation through PyO3 bindings. It abstracts the details of
converting between Python tensors and Rust types.

Functions:
    call_rust_emd_forward: Call Rust EMD forward pass, return IMFs
    call_rust_emd_backward: Call Rust EMD backward pass, return signal gradients
"""

from typing import Any, Dict, List, Tuple, Union
import numpy as np


def call_rust_emd_forward(
    signal: Union[np.ndarray, "tf.Tensor", "torch.Tensor"],
    config: Dict[str, Any],
    framework: str = "numpy",
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Call Rust EMD forward pass and return IMFs and context.

    Decomposes a 1D signal using EMD implemented in Rust, capturing
    all context needed for implicit differentiation backward pass.

    Args:
        signal: Input signal (1D numpy array or tensor)
            Shape: (signal_length,)
        config: Configuration dictionary for EMD
            Keys: max_imfs, boundary_type, sifting_iterations, etc.
        framework: Framework of input tensor ("numpy", "tensorflow", "torch")

    Returns:
        Tuple of:
            imfs: Array of IMFs
                Shape: (num_imfs, signal_length)
            context: Dictionary with forward pass context
                Keys: {
                    "imfs": list of arrays,
                    "residue": array,
                    "extrema": list of (max_idx, min_idx) tuples,
                    "num_sifts": list of counts,
                    "config": config dict,
                    "metadata": additional metadata
                }

    Raises:
        ImportError: If ferromode_py (Rust bindings) not installed
        ValueError: If signal is empty or contains NaN/Inf
        RuntimeError: If EMD decomposition fails

    Examples:
        >>> signal = np.array([1.0, 2.0, 3.0, 4.0, 3.0, 2.0, 1.0])
        >>> config = {"max_imfs": 3}
        >>> imfs, context = call_rust_emd_forward(signal, config, "numpy")
        >>> print(f"Extracted {imfs.shape[0]} IMFs")
    """
    # Convert input to numpy for Rust call
    signal_np = _to_numpy(signal, framework)

    # Validate input
    if signal_np.ndim != 1:
        raise ValueError(f"Signal must be 1D, got shape {signal_np.shape}")
    if len(signal_np) == 0:
        raise ValueError("Signal cannot be empty")
    if not np.all(np.isfinite(signal_np)):
        raise ValueError("Signal contains NaN or Inf values")

    try:
        # Import Rust bindings
        from ferromode_py import emd_forward as rust_emd_forward

        # Call Rust forward pass
        result = rust_emd_forward(signal_np.tolist(), config)

    except ImportError as e:
        raise ImportError(
            "ferromode_py module not found. Install it with:\n"
            "  pip install ferromode-py\n"
            "  or build from source: cargo build -p ferromode-py --release"
        ) from e

    # Extract IMFs from result
    imfs_list: List[List[float]] = result.get("imfs", [])
    residue_list: List[float] = result.get("residue", [])
    extrema: List[Tuple[List[int], List[int]]] = result.get("extrema", [])
    num_sifts: List[int] = result.get("num_sifts", [])
    metadata: Dict[str, str] = result.get("metadata", {})

    # Convert to numpy arrays
    imfs_np = (
        np.array(imfs_list, dtype=np.float64)
        if imfs_list
        else np.empty((0, len(signal_np)))
    )
    residue_np = (
        np.array(residue_list, dtype=np.float64)
        if residue_list
        else np.zeros(len(signal_np))
    )

    # Package context for backward pass
    context: Dict[str, Any] = {
        "imfs": imfs_list,
        "residue": residue_list,
        "extrema": extrema,
        "num_sifts": num_sifts,
        "config": config,
        "metadata": metadata,
        "signal_length": len(signal_np),
    }

    return imfs_np, context


def call_rust_emd_backward(
    grad_imfs: Union[np.ndarray, "tf.Tensor", "torch.Tensor"],
    signal: Union[np.ndarray, "tf.Tensor", "torch.Tensor"],
    context: Dict[str, Any],
    framework: str = "numpy",
) -> np.ndarray:
    """Call Rust EMD backward pass and return signal gradients.

    Computes gradients of the loss with respect to the input signal
    using implicit differentiation. Uses the context saved during
    the forward pass.

    Args:
        grad_imfs: Upstream gradients from loss w.r.t. IMFs
            Shape: (num_imfs, signal_length)
        signal: Original input signal (from forward pass)
            Shape: (signal_length,)
        context: Context dictionary from forward pass
            Should contain "extrema", "residue", "config", etc.
        framework: Framework of input tensors ("numpy", "tensorflow", "torch")

    Returns:
        grad_signal: Gradients w.r.t. input signal
            Shape: (signal_length,)

    Raises:
        ImportError: If ferromode_py not installed
        ValueError: If gradient shapes don't match context
        RuntimeError: If implicit differentiation fails

    Examples:
        >>> grad_imfs = np.random.randn(3, 7)  # From upstream loss
        >>> signal = np.array([1.0, 2.0, 3.0, 4.0, 3.0, 2.0, 1.0])
        >>> context = {...}  # From forward pass
        >>> grad_signal = call_rust_emd_backward(grad_imfs, signal, context, "numpy")
        >>> print(f"Signal gradients shape: {grad_signal.shape}")
    """
    # Convert inputs to numpy
    grad_imfs_np = _to_numpy(grad_imfs, framework)
    signal_np = _to_numpy(signal, framework)

    # Validate shapes
    signal_len = context["signal_length"]
    if signal_np.shape[0] != signal_len:
        raise ValueError(
            f"Signal length mismatch: got {signal_np.shape[0]}, "
            f"expected {signal_len} from context"
        )
    if grad_imfs_np.ndim != 2 or grad_imfs_np.shape[1] != signal_len:
        raise ValueError(
            f"Gradient shape {grad_imfs_np.shape} incompatible with "
            f"signal length {signal_len}"
        )

    try:
        # Import Rust bindings
        from ferromode_py import emd_backward as rust_emd_backward

        # Call Rust backward pass
        grad_signal_list: List[float] = rust_emd_backward(
            grad_imfs_np.tolist(),
            signal_np.tolist(),
            context["config"],
            context.get("extrema", []),
        )

    except ImportError as e:
        raise ImportError(
            "ferromode_py module not found. Install it with:\n"
            "  pip install ferromode-py"
        ) from e

    # Convert result to numpy array
    grad_signal_np = np.array(grad_signal_list, dtype=np.float64)

    # Validate output
    if not np.all(np.isfinite(grad_signal_np)):
        raise RuntimeError("Backward pass produced NaN or Inf gradients")

    return grad_signal_np


def _to_numpy(
    tensor: Union[np.ndarray, "tf.Tensor", "torch.Tensor"],
    framework: str,
) -> np.ndarray:
    """Convert tensor from any framework to numpy.

    Args:
        tensor: Input tensor
        framework: "numpy", "tensorflow", or "torch"

    Returns:
        Numpy array with same data as tensor

    Raises:
        ValueError: If framework not recognized
    """
    if framework == "numpy":
        if isinstance(tensor, np.ndarray):
            return tensor
        return np.array(tensor)

    elif framework == "tensorflow":
        try:
            import tensorflow as tf

            if isinstance(tensor, tf.Tensor):
                return tensor.numpy()
            return np.array(tensor)
        except ImportError as e:
            raise ImportError("TensorFlow not installed") from e

    elif framework == "torch":
        try:
            import torch

            if isinstance(tensor, torch.Tensor):
                if tensor.is_cuda:
                    return tensor.detach().cpu().numpy()
                return tensor.detach().numpy()
            return np.array(tensor)
        except ImportError as e:
            raise ImportError("PyTorch not installed") from e

    else:
        raise ValueError(
            f"Unknown framework: {framework}. Expected 'numpy', 'tensorflow', or 'torch'"
        )


def _to_tensor(
    array: np.ndarray,
    framework: str,
    device: str = "cpu",
) -> Union[np.ndarray, "tf.Tensor", "torch.Tensor"]:
    """Convert numpy array to tensor in target framework.

    Args:
        array: Input numpy array
        framework: "numpy", "tensorflow", or "torch"
        device: Device for torch tensors ("cpu" or "cuda")

    Returns:
        Tensor in the target framework

    Raises:
        ValueError: If framework not recognized
    """
    if framework == "numpy":
        return array

    elif framework == "tensorflow":
        try:
            import tensorflow as tf

            return tf.constant(array, dtype=tf.float32)
        except ImportError as e:
            raise ImportError("TensorFlow not installed") from e

    elif framework == "torch":
        try:
            import torch

            tensor = torch.from_numpy(array).float()
            if device == "cuda" and torch.cuda.is_available():
                return tensor.cuda()
            return tensor
        except ImportError as e:
            raise ImportError("PyTorch not installed") from e

    else:
        raise ValueError(f"Unknown framework: {framework}")
