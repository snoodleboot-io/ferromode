"""PyTorch module for differentiable EMD.

This module provides a PyTorch nn.Module that integrates differentiable EMD
into PyTorch models, enabling end-to-end gradient-based training with
torch.autograd.

Classes:
    DifferentiableEMDFunction: Custom autograd.Function for EMD
    DifferentiableEMD: PyTorch nn.Module wrapper
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None  # type: ignore
    nn = None  # type: ignore

from .rust_bridge import call_rust_emd_backward, call_rust_emd_forward


class DifferentiableEMDFunction(torch.autograd.Function):  # type: ignore
    """Custom autograd function for differentiable EMD.

    This class implements the forward and backward passes for EMD using
    torch.autograd.Function. The forward pass calls the Rust EMD implementation,
    and the backward pass uses implicit differentiation to compute gradients.

    The backward pass saves the necessary context during the forward pass
    and uses it to compute signal gradients via implicit differentiation,
    avoiding expensive backpropagation through the sifting loop.
    """

    @staticmethod
    def forward(
        ctx: Any,  # type: ignore
        signals: torch.Tensor,  # type: ignore
        config: Dict[str, Any],
    ) -> torch.Tensor:  # type: ignore
        """Forward pass: decompose signals into IMFs.

        Args:
            ctx: PyTorch autograd context (for saving state)
            signals: Input signals
                Shape: (batch_size, signal_length) or (signal_length,)
            config: EMD configuration dictionary

        Returns:
            IMFs stacked into tensor
            Shape: (batch_size, num_imfs, signal_length) or (num_imfs, signal_length)
        """
        # Ensure batch dimension
        is_batched = signals.dim() == 2
        if not is_batched:
            signals = signals.unsqueeze(0)

        batch_size, signal_length = signals.shape
        device = signals.device
        dtype = signals.dtype

        # Process each signal in batch
        all_imfs: List[np.ndarray] = []
        all_contexts: List[Dict[str, Any]] = []
        max_imfs_found = 0

        signals_np = signals.cpu().numpy().astype(np.float64)

        for i in range(batch_size):
            signal = signals_np[i]

            # Call Rust forward pass
            imfs, context = call_rust_emd_forward(signal, config, "numpy")

            all_imfs.append(imfs)
            all_contexts.append(context)
            max_imfs_found = max(max_imfs_found, imfs.shape[0])

        # Pad IMFs to same number
        padded_imfs: List[np.ndarray] = []
        for imfs in all_imfs:
            if imfs.shape[0] < max_imfs_found:
                padding = np.zeros(
                    (max_imfs_found - imfs.shape[0], imfs.shape[1]),
                    dtype=np.float64,
                )
                imfs = np.vstack([imfs, padding])
            padded_imfs.append(imfs)

        # Convert to tensor
        output_np = np.stack(padded_imfs, axis=0).astype(np.float32)
        output = torch.from_numpy(output_np).to(device=device, dtype=dtype)

        # Save for backward
        ctx.save_for_backward(signals)
        ctx.config = config
        ctx.contexts = all_contexts
        ctx.is_batched = is_batched
        ctx.original_imf_counts = [imfs.shape[0] for imfs in all_imfs]

        return output

    @staticmethod
    def backward(
        ctx: Any,  # type: ignore
        grad_output: torch.Tensor,  # type: ignore
    ) -> Tuple[torch.Tensor, None]:  # type: ignore
        """Backward pass: compute signal gradients via implicit differentiation.

        Args:
            ctx: PyTorch autograd context with saved state
            grad_output: Upstream gradients from loss w.r.t. IMFs

        Returns:
            Tuple of (grad_signals, None)
            - grad_signals: Gradients w.r.t. input signals
            - None: No gradients w.r.t. config (it's not a tensor)
        """
        (signals,) = ctx.saved_tensors
        config = ctx.config
        contexts = ctx.contexts
        is_batched = ctx.is_batched

        batch_size = signals.shape[0]
        device = signals.device
        dtype = signals.dtype

        # Convert gradients to numpy
        grad_output_np = grad_output.cpu().numpy().astype(np.float64)
        signals_np = signals.cpu().numpy().astype(np.float64)

        # Compute gradients for each signal
        grad_signals_list: List[np.ndarray] = []

        for i in range(batch_size):
            context = contexts[i]
            signal = signals_np[i]
            grad_imfs_i = grad_output_np[i, :, :]

            # Call Rust backward pass
            grad_signal = call_rust_emd_backward(
                grad_imfs_i,
                signal,
                context,
                "numpy",
            )

            grad_signals_list.append(grad_signal)

        # Stack and convert to tensor
        grad_signals_np = np.stack(grad_signals_list, axis=0).astype(np.float32)
        grad_signals = torch.from_numpy(grad_signals_np).to(device=device, dtype=dtype)

        # Remove batch dimension if input wasn't batched
        if not is_batched:
            grad_signals = grad_signals.squeeze(0)

        return grad_signals, None


class DifferentiableEMD(nn.Module):  # type: ignore
    """PyTorch module for differentiable EMD decomposition.

    This module wraps DifferentiableEMDFunction to provide a convenient
    interface compatible with PyTorch nn.Module and torch.nn.Sequential.

    The module accepts batches of signals and returns a batch of decomposed IMFs.
    Gradients flow backward through implicit differentiation.

    Args:
        max_imfs: Maximum number of IMFs to extract (int or None)
            If None, determined automatically by stopping criteria.
            Default: None
        boundary: Boundary extension method ("mirror", "periodic", "symm")
            Default: "mirror"
        sifting_iterations: Maximum sifting iterations per IMF (int)
            Default: 100
        sifting_tol: Convergence tolerance for sifting loop (float)
            Default: 1e-6

    Input Shape:
        (batch_size, signal_length) or (signal_length,)

    Output Shape:
        (batch_size, num_imfs, signal_length) if batch present
        (num_imfs, signal_length) if no batch

    Examples:
        >>> # Simple forward pass
        >>> emd = DifferentiableEMD(max_imfs=5)
        >>> signals = torch.randn(32, 256)  # 32 signals of length 256
        >>> imfs = emd(signals)  # (32, num_imfs, 256)

        >>> # In a neural network with backpropagation
        >>> class SignalClassifier(nn.Module):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.emd = DifferentiableEMD(max_imfs=5)
        ...         self.fc = nn.Linear(5 * 256, 10)
        ...
        ...     def forward(self, x):
        ...         imfs = self.emd(x)  # (batch, 5, 256)
        ...         imfs_flat = imfs.reshape(imfs.shape[0], -1)  # (batch, 1280)
        ...         return self.fc(imfs_flat)

        >>> # Train with gradients flowing through EMD
        >>> model = SignalClassifier()
        >>> optimizer = torch.optim.Adam(model.parameters())
        >>> signals = torch.randn(32, 256)
        >>> targets = torch.randint(0, 10, (32,))
        >>> outputs = model(signals)
        >>> loss = nn.CrossEntropyLoss()(outputs, targets)
        >>> loss.backward()  # Gradients flow through EMD
        >>> optimizer.step()

    Notes:
        - Input signals should be floating-point tensors (float32 or float64)
        - The number of extracted IMFs may vary per signal; automatic padding is applied
        - Gradients are computed using implicit differentiation for numerical stability
        - Batch processing is supported and recommended for efficiency
        - GPU tensors are automatically converted to CPU for Rust processing
    """

    def __init__(
        self,
        max_imfs: Optional[int] = None,
        boundary: str = "mirror",
        sifting_iterations: int = 100,
        sifting_tol: float = 1e-6,
    ):
        """Initialize DifferentiableEMD module."""
        if torch is None:
            raise ImportError("PyTorch is required for DifferentiableEMD")

        super().__init__()

        self.max_imfs = max_imfs
        self.boundary = boundary
        self.sifting_iterations = sifting_iterations
        self.sifting_tol = sifting_tol

        # Build EMD configuration
        self.emd_config: Dict[str, Any] = {
            "boundary": boundary,
            "sifting_iterations": sifting_iterations,
            "sifting_tol": sifting_tol,
        }
        if max_imfs is not None:
            self.emd_config["max_imfs"] = max_imfs

    def forward(self, signals: torch.Tensor) -> torch.Tensor:  # type: ignore
        """Forward pass: decompose batch of signals into IMFs.

        Args:
            signals: Input signals
                Shape: (batch_size, signal_length) or (signal_length,)
                Dtype: float32 or float64
                Device: cpu or cuda (will be moved to CPU for processing)

        Returns:
            IMFs for all signals
            Shape: (batch_size, num_imfs, signal_length) or (num_imfs, signal_length)

        Raises:
            RuntimeError: If signal contains NaN or Inf
            ValueError: If signal is empty
        """
        # Validate input
        if not torch.isfinite(signals).all():
            raise RuntimeError("Input signals contain NaN or Inf values")

        # Ensure float dtype
        if signals.dtype not in (torch.float32, torch.float64):
            signals = signals.float()

        # Apply custom autograd function
        return DifferentiableEMDFunction.apply(signals, self.emd_config)

    def extra_repr(self) -> str:
        """String representation of module parameters.

        Returns:
            String describing layer configuration
        """
        return (
            f"max_imfs={self.max_imfs}, "
            f"boundary={self.boundary}, "
            f"sifting_iterations={self.sifting_iterations}, "
            f"sifting_tol={self.sifting_tol}"
        )


# Convenience alias for compatibility
DifferentiableEMDLayer = DifferentiableEMD
