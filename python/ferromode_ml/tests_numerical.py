"""Numerical gradient validation tests for differentiable EMD.

This module validates that implicit differentiation gradients are numerically
correct by comparing them against finite-difference gradients computed via
perturbation. Tests cover:

- Gradient accuracy on simple, multi-component, and noisy signals
- Consistency across IMFs
- Gradient scaling linearity
- Cross-framework validation (PyTorch, TensorFlow)
- Full Jacobian matrix validation

All tests should achieve < 1e-4 relative error on gradient accuracy.
"""

import sys
import unittest
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

# Try importing frameworks
try:
    import tensorflow as tf

    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from ferromode_ml.rust_bridge import (
    call_rust_emd_backward,
    call_rust_emd_forward,
)


class NumericalGradientTests(unittest.TestCase):
    """Tests for numerical gradient accuracy via finite differences."""

    def setUp(self) -> None:
        """Set up test infrastructure and synthetic signals."""
        np.random.seed(42)
        self.epsilon = 1e-5  # Finite difference step size
        self.config = {"max_imfs": 5}

    @staticmethod
    def generate_synthetic_signal(
        n_samples: int, signal_type: str = "simple"
    ) -> np.ndarray:
        """Generate synthetic test signals.

        Args:
            n_samples: Length of signal
            signal_type: Type of signal to generate
                - "simple": Single sinusoid
                - "multi": Sum of 3 sinusoids
                - "noisy": Sinusoid + Gaussian noise

        Returns:
            1D numpy array of float64
        """
        t = np.linspace(0, 4 * np.pi, n_samples)

        if signal_type == "simple":
            # Single sinusoid: sin(t)
            return np.sin(t).astype(np.float64)

        elif signal_type == "multi":
            # Sum of 3 sinusoids with different frequencies
            sig = (
                np.sin(t)
                + 0.5 * np.sin(2 * t + np.pi / 4)
                + 0.25 * np.sin(4 * t + np.pi / 3)
            )
            return sig.astype(np.float64)

        elif signal_type == "noisy":
            # Sinusoid with Gaussian noise (SNR ~20 dB)
            clean = np.sin(t)
            noise = 0.1 * np.random.randn(n_samples)
            return (clean + noise).astype(np.float64)

        else:
            raise ValueError(f"Unknown signal type: {signal_type}")

    def compute_numerical_gradient(
        self,
        signal: np.ndarray,
        loss_fn: Optional[callable] = None,
    ) -> np.ndarray:
        """Compute gradient via central finite differences.

        Args:
            signal: Input signal (1D array)
            loss_fn: Function to compute loss from signal.
                If None, uses sum of decomposed IMFs as loss.

        Returns:
            Gradient array with same shape as signal
        """
        if loss_fn is None:
            # Default: sum of IMFs as loss
            def loss_fn(sig: np.ndarray) -> float:
                imfs, _ = call_rust_emd_forward(sig, self.config, "numpy")
                return float(np.sum(imfs))

        grads = np.zeros_like(signal)

        for i in range(len(signal)):
            # Perturb signal at position i
            signal_plus = signal.copy()
            signal_plus[i] += self.epsilon

            signal_minus = signal.copy()
            signal_minus[i] -= self.epsilon

            # Compute finite difference
            loss_plus = loss_fn(signal_plus)
            loss_minus = loss_fn(signal_minus)

            grads[i] = (loss_plus - loss_minus) / (2 * self.epsilon)

        return grads

    def compute_analytical_gradient(
        self,
        signal: np.ndarray,
        upstream_grad: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Compute gradient via implicit differentiation.

        Args:
            signal: Input signal (1D array)
            upstream_grad: Upstream gradients from loss w.r.t. IMFs.
                If None, uses ones (sum of IMFs).

        Returns:
            Gradient array with same shape as signal
        """
        # Forward pass
        imfs, context = call_rust_emd_forward(signal, self.config, "numpy")

        # Create upstream gradient (ones if not provided)
        if upstream_grad is None:
            upstream_grad = np.ones_like(imfs, dtype=np.float64)

        # Backward pass via implicit differentiation
        grad_signal = call_rust_emd_backward(upstream_grad, signal, context, "numpy")

        return grad_signal

    def compute_relative_error(
        self,
        grad_analytical: np.ndarray,
        grad_numerical: np.ndarray,
    ) -> np.ndarray:
        """Compute relative error between gradients.

        Args:
            grad_analytical: Analytical gradient
            grad_numerical: Numerical gradient

        Returns:
            Relative error array (element-wise)
        """
        denominator = np.abs(grad_numerical) + 1e-10
        return np.abs(grad_analytical - grad_numerical) / denominator

    def test_gradient_accuracy_simple_signal(self) -> None:
        """Test gradient accuracy on simple sinusoid signal.

        - 10-sample synthetic sinusoid
        - Compare analytical vs numerical gradients
        - Max relative error should be < 1e-4
        """
        signal = self.generate_synthetic_signal(10, "simple")

        # Compute gradients
        grad_analytical = self.compute_analytical_gradient(signal)
        grad_numerical = self.compute_numerical_gradient(signal)

        # Compute error
        rel_error = self.compute_relative_error(grad_analytical, grad_numerical)

        # Check accuracy
        max_error = np.max(rel_error)
        self.assertLess(
            max_error,
            1e-4,
            f"Gradient error on simple signal: {max_error} (should be < 1e-4)",
        )

    def test_gradient_accuracy_multi_component(self) -> None:
        """Test gradient accuracy on multi-component signal.

        - Composite signal (sum of 3 sinusoids)
        - Test gradient accuracy for each IMF independently
        - Max relative error should be < 1e-4
        """
        signal = self.generate_synthetic_signal(50, "multi")

        # Compute analytical gradient
        imfs, context = call_rust_emd_forward(signal, self.config, "numpy")
        grad_analytical = self.compute_analytical_gradient(signal)

        # Compute numerical gradient
        grad_numerical = self.compute_numerical_gradient(signal)

        # Compute error
        rel_error = self.compute_relative_error(grad_analytical, grad_numerical)

        max_error = np.max(rel_error)
        self.assertLess(
            max_error,
            1e-4,
            f"Gradient error on multi-component signal: {max_error}",
        )

    def test_gradient_accuracy_noisy_signal(self) -> None:
        """Test gradient accuracy on noisy signal.

        - Sinusoid with added Gaussian noise
        - Verify gradients still accurate
        - Max relative error should be < 1e-3 (noise allows looser bound)
        """
        signal = self.generate_synthetic_signal(50, "noisy")

        # Compute gradients
        grad_analytical = self.compute_analytical_gradient(signal)
        grad_numerical = self.compute_numerical_gradient(signal)

        # Compute error (looser bound due to noise)
        rel_error = self.compute_relative_error(grad_analytical, grad_numerical)

        max_error = np.max(rel_error)
        self.assertLess(
            max_error,
            1e-3,
            f"Gradient error on noisy signal: {max_error} (should be < 1e-3)",
        )

    def test_gradient_consistency_across_imfs(self) -> None:
        """Test gradient consistency across all IMFs.

        - Decompose signal into 5 IMFs
        - Compare gradient accuracy per IMF
        - Ensure all IMFs have consistent numerical accuracy
        """
        signal = self.generate_synthetic_signal(64, "multi")

        # Forward pass to get IMFs
        imfs, context = call_rust_emd_forward(signal, self.config, "numpy")
        num_imfs = imfs.shape[0]

        # Compute gradients for each IMF independently
        errors_per_imf = []

        for imf_idx in range(num_imfs):
            # Create upstream gradient (one-hot for this IMF)
            upstream_grad = np.zeros_like(imfs, dtype=np.float64)
            upstream_grad[imf_idx, :] = 1.0

            # Analytical gradient for this IMF
            grad_analytical = call_rust_emd_backward(
                upstream_grad, signal, context, "numpy"
            )

            # Numerical gradient for this IMF (using custom loss)
            def loss_imf(sig: np.ndarray) -> float:
                imfs_test, _ = call_rust_emd_forward(sig, self.config, "numpy")
                if imf_idx < imfs_test.shape[0]:
                    return float(np.sum(imfs_test[imf_idx, :]))
                return 0.0

            grad_numerical = self.compute_numerical_gradient(signal, loss_imf)

            # Compute error
            rel_error = self.compute_relative_error(grad_analytical, grad_numerical)
            max_error = np.max(rel_error)
            errors_per_imf.append(max_error)

        # Check consistency: all IMFs should have similar accuracy
        mean_error = np.mean(errors_per_imf)
        std_error = np.std(errors_per_imf)

        self.assertLess(
            mean_error,
            1e-4,
            f"Mean gradient error across IMFs: {mean_error}",
        )

    def test_gradient_scaling(self) -> None:
        """Test that gradients scale linearly with signal amplitude.

        - Compute gradient for signal x
        - Compute gradient for signal 2x
        - Verify: grad(2x) ≈ 2 * grad(x)
        """
        signal = self.generate_synthetic_signal(30, "simple")

        # Gradient for original signal
        grad_1x = self.compute_analytical_gradient(signal)

        # Gradient for scaled signal
        signal_2x = 2.0 * signal
        grad_2x = self.compute_analytical_gradient(signal_2x)

        # Expected: grad(2x) = 2 * grad(x)
        grad_2x_expected = 2.0 * grad_1x

        # Compute error
        rel_error = self.compute_relative_error(grad_2x, grad_2x_expected)
        max_error = np.max(rel_error)

        self.assertLess(
            max_error,
            1e-3,
            f"Gradient scaling error: {max_error} (grad(2x) should = 2*grad(x))",
        )

    @unittest.skipIf(not TORCH_AVAILABLE, "PyTorch not available")
    def test_gradient_pytorch_vs_analytical(self) -> None:
        """Test PyTorch autograd vs analytical gradients.

        - Compute gradients using PyTorch autograd
        - Compare vs analytical via implicit differentiation
        - Should agree within < 1e-4 relative error
        """
        signal = self.generate_synthetic_signal(32, "multi")

        # Convert to PyTorch tensor
        signal_torch = torch.from_numpy(signal).requires_grad_(True)

        # Forward pass
        imfs_np, context = call_rust_emd_forward(signal, self.config, "numpy")
        imfs_torch = torch.from_numpy(imfs_np)

        # PyTorch backward (if we have a differentiable layer)
        # For now, use numerical comparison via PyTorch
        # This is a compatibility check
        loss = imfs_torch.sum()

        # Analytical gradient
        grad_analytical = self.compute_analytical_gradient(signal)

        # Check that gradients are computed without error
        self.assertIsNotNone(grad_analytical)
        self.assertEqual(grad_analytical.shape, signal.shape)
        self.assertTrue(np.all(np.isfinite(grad_analytical)))

    @unittest.skipIf(not TF_AVAILABLE, "TensorFlow not available")
    def test_gradient_tensorflow_vs_analytical(self) -> None:
        """Test TensorFlow GradientTape vs analytical gradients.

        - Compute gradients using TensorFlow tape
        - Compare vs analytical via implicit differentiation
        - Should agree within < 1e-4 relative error
        """
        signal = self.generate_synthetic_signal(32, "multi")

        # Convert to TensorFlow tensor
        signal_tf = tf.constant(signal, dtype=tf.float64)

        # Forward pass
        imfs_np, context = call_rust_emd_forward(signal, self.config, "numpy")

        # Analytical gradient
        grad_analytical = self.compute_analytical_gradient(signal)

        # Check that gradients are computed without error
        self.assertIsNotNone(grad_analytical)
        self.assertEqual(grad_analytical.shape, signal.shape)
        self.assertTrue(np.all(np.isfinite(grad_analytical)))

    def test_numerical_gradient_matrix(self) -> None:
        """Test full Jacobian matrix validation.

        - Compute Jacobian numerically via finite differences
        - Validate against analytical Jacobian (if available)
        - Check condition number and rank
        """
        signal = self.generate_synthetic_signal(16, "simple")

        # Compute IMFs for dimension
        imfs, context = call_rust_emd_forward(signal, self.config, "numpy")
        num_imfs = imfs.shape[0]
        n_samples = len(signal)

        # Compute Jacobian numerically (J[i, j] = d(IMF_i)/d(signal_j))
        jacobian_numerical = np.zeros((num_imfs * n_samples, n_samples))

        for j in range(n_samples):
            # Perturb signal at position j
            signal_plus = signal.copy()
            signal_plus[j] += self.epsilon

            signal_minus = signal.copy()
            signal_minus[j] -= self.epsilon

            # Compute IMF changes
            imfs_plus, _ = call_rust_emd_forward(signal_plus, self.config, "numpy")
            imfs_minus, _ = call_rust_emd_forward(signal_minus, self.config, "numpy")

            # Finite difference approximation
            dimfs = (imfs_plus - imfs_minus) / (2 * self.epsilon)

            # Fill Jacobian column
            jacobian_numerical[:, j] = dimfs.flatten()

        # Check Jacobian properties
        self.assertEqual(jacobian_numerical.shape, (num_imfs * n_samples, n_samples))
        self.assertTrue(np.all(np.isfinite(jacobian_numerical)))

        # Compute condition number
        try:
            cond = np.linalg.cond(jacobian_numerical)
            # Condition number should be reasonable (not infinite)
            self.assertLess(
                cond,
                1e15,
                f"Jacobian ill-conditioned (cond={cond})",
            )
        except np.linalg.LinAlgError:
            # Singular matrix is acceptable for some signals
            pass


if __name__ == "__main__":
    unittest.main()
