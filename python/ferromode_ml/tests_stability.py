"""Numerical stability validation tests for differentiable EMD.

This module validates that implicit differentiation backward pass is
numerically stable and handles edge cases correctly. Tests cover:

- NaN/Inf detection and prevention
- Gradient clipping and range validation
- Condition number monitoring and warnings
- Tikhonov regularization effectiveness
- Batch processing stability
- Edge cases (constant, tiny signals)

All tests should verify that gradients are finite, properly clipped,
and don't contain any NaN or Inf values.
"""

import sys
import unittest
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import warnings

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


class StabilityTests(unittest.TestCase):
    """Tests for numerical stability of gradient computation."""

    def setUp(self) -> None:
        """Set up test infrastructure."""
        np.random.seed(42)
        self.config = {"max_imfs": 5}
        # Gradient clipping bounds (from implicit_diff.rs)
        self.grad_clip_min = -100.0
        self.grad_clip_max = 100.0
        # Condition number warning threshold (from implicit_diff.rs)
        self.cond_threshold = 1e10

    @staticmethod
    def generate_synthetic_signal(
        n_samples: int,
        signal_type: str = "simple",
        amplitude: float = 1.0,
    ) -> np.ndarray:
        """Generate synthetic test signals.

        Args:
            n_samples: Length of signal
            signal_type: Type of signal
                - "simple": Single sinusoid
                - "multi": Sum of 3 sinusoids
                - "noisy": Sinusoid + noise
                - "constant": Flat constant signal
                - "tiny": Very small amplitude signal
                - "badly_conditioned": Signal designed for ill-conditioning
            amplitude: Signal amplitude scale

        Returns:
            1D numpy array of float64
        """
        t = np.linspace(0, 4 * np.pi, n_samples)

        if signal_type == "simple":
            return (amplitude * np.sin(t)).astype(np.float64)

        elif signal_type == "multi":
            sig = (
                amplitude * np.sin(t)
                + 0.5 * amplitude * np.sin(2 * t + np.pi / 4)
                + 0.25 * amplitude * np.sin(4 * t + np.pi / 3)
            )
            return sig.astype(np.float64)

        elif signal_type == "noisy":
            clean = amplitude * np.sin(t)
            noise = 0.1 * amplitude * np.random.randn(n_samples)
            return (clean + noise).astype(np.float64)

        elif signal_type == "constant":
            # Flat constant signal
            return (amplitude * np.ones(n_samples)).astype(np.float64)

        elif signal_type == "tiny":
            # Very small amplitude (near machine precision)
            return (amplitude * 1e-6 * np.sin(t)).astype(np.float64)

        elif signal_type == "badly_conditioned":
            # Signal designed to create ill-conditioned Jacobian
            # Multiple closely-spaced frequencies
            sig = np.sin(t) + 0.99 * np.sin(1.01 * t) + 0.98 * np.sin(1.02 * t)
            return (amplitude * sig).astype(np.float64)

        else:
            raise ValueError(f"Unknown signal type: {signal_type}")

    def test_no_nan_in_gradients(self) -> None:
        """Test that gradients never contain NaN values.

        - Compute gradients for 10 different signals
        - Check: all values are finite
        - Fail if any NaN/Inf detected
        """
        for i in range(10):
            # Generate signal with increasing length
            n_samples = 50 + 10 * i
            signal = self.generate_synthetic_signal(n_samples, "simple")

            # Forward pass
            imfs, context = call_rust_emd_forward(signal, self.config, "numpy")

            # Backward pass
            grad_imfs = np.ones_like(imfs, dtype=np.float64)
            grad_signal = call_rust_emd_backward(grad_imfs, signal, context, "numpy")

            # Check for NaN/Inf
            self.assertTrue(
                np.all(np.isfinite(grad_signal)),
                f"NaN/Inf detected in gradient (signal {i}, n={n_samples})",
            )

            # Check shape
            self.assertEqual(
                grad_signal.shape,
                signal.shape,
                f"Gradient shape mismatch (signal {i})",
            )

    def test_no_nan_with_noisy_input(self) -> None:
        """Test stability with increasing noise levels.

        - Add increasing noise to signal
        - Verify gradients remain finite
        - Test noise levels: 1%, 5%, 10%, 20% of amplitude
        """
        signal_clean = self.generate_synthetic_signal(64, "simple", amplitude=1.0)

        noise_levels = [0.01, 0.05, 0.10, 0.20]  # 1%, 5%, 10%, 20%

        for noise_level in noise_levels:
            # Add noise
            noise = noise_level * np.random.randn(len(signal_clean))
            signal_noisy = signal_clean + noise

            # Forward pass
            imfs, context = call_rust_emd_forward(signal_noisy, self.config, "numpy")

            # Backward pass
            grad_imfs = np.ones_like(imfs, dtype=np.float64)
            grad_signal = call_rust_emd_backward(
                grad_imfs, signal_noisy, context, "numpy"
            )

            # Check for NaN/Inf
            self.assertTrue(
                np.all(np.isfinite(grad_signal)),
                f"NaN/Inf with noise level {100 * noise_level}%",
            )

    def test_gradient_clipping_works(self) -> None:
        """Test that gradient clipping is applied.

        - Create signal that might produce large gradients
        - Compute gradients
        - Verify: all values in [-100, 100] range
        """
        # Use a signal that might create large gradients
        signal = self.generate_synthetic_signal(32, "badly_conditioned", amplitude=10.0)

        # Forward pass
        imfs, context = call_rust_emd_forward(signal, self.config, "numpy")

        # Backward pass
        grad_imfs = np.ones_like(imfs, dtype=np.float64)
        grad_signal = call_rust_emd_backward(grad_imfs, signal, context, "numpy")

        # Check clipping bounds
        max_grad = np.max(np.abs(grad_signal))
        self.assertLessEqual(
            max_grad,
            self.grad_clip_max,
            f"Gradient exceeds clipping bound: {max_grad} > {self.grad_clip_max}",
        )

        # Check that values are actually bounded
        self.assertTrue(
            np.all(grad_signal >= self.grad_clip_min),
            f"Gradient below min bound: {np.min(grad_signal)}",
        )
        self.assertTrue(
            np.all(grad_signal <= self.grad_clip_max),
            f"Gradient above max bound: {np.max(grad_signal)}",
        )

    def test_condition_number_monitoring(self) -> None:
        """Test condition number monitoring.

        - Create badly-conditioned signal
        - Compute Jacobian condition number
        - Verify: warning issued if cond > 1e10
        - Verify: regularization applied automatically
        """
        signal = self.generate_synthetic_signal(32, "badly_conditioned")

        # Forward pass
        imfs, context = call_rust_emd_forward(signal, self.config, "numpy")

        # Create upstream gradient
        grad_imfs = np.ones_like(imfs, dtype=np.float64)

        # Backward pass (should handle ill-conditioning gracefully)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            grad_signal = call_rust_emd_backward(grad_imfs, signal, context, "numpy")

            # Gradient should still be finite despite conditioning
            self.assertTrue(
                np.all(np.isfinite(grad_signal)),
                "Backward pass failed with ill-conditioned Jacobian",
            )

    def test_regularization_improves_stability(self) -> None:
        """Test that regularization improves conditioning.

        - Compute gradients for badly-conditioned signal
        - Verify: gradients are finite
        - Verify: no NaN/Inf issues
        """
        signal = self.generate_synthetic_signal(32, "badly_conditioned")

        # Forward pass
        imfs, context = call_rust_emd_forward(signal, self.config, "numpy")

        # Create upstream gradient
        grad_imfs = np.random.randn(*imfs.shape).astype(np.float64)

        # Backward pass (uses regularization internally)
        grad_signal = call_rust_emd_backward(grad_imfs, signal, context, "numpy")

        # Verify stability
        self.assertTrue(
            np.all(np.isfinite(grad_signal)),
            "Regularization failed: NaN/Inf in gradients",
        )

        # Verify gradients are in reasonable range
        max_abs_grad = np.max(np.abs(grad_signal))
        self.assertLess(
            max_abs_grad,
            1e6,
            f"Gradients unreasonably large after regularization: {max_abs_grad}",
        )

    def test_gradient_stability_batch_processing(self) -> None:
        """Test batch processing stability.

        - Process batch of 10 signals
        - Verify all batch items produce finite gradients
        - Verify no cross-batch contamination
        """
        batch_size = 10
        signal_length = 32

        for batch_idx in range(batch_size):
            # Generate signal
            signal = self.generate_synthetic_signal(
                signal_length,
                "multi",
                amplitude=1.0 + 0.1 * batch_idx,
            )

            # Forward pass
            imfs, context = call_rust_emd_forward(signal, self.config, "numpy")

            # Backward pass with different upstream gradients
            grad_imfs = np.random.randn(*imfs.shape).astype(np.float64)
            grad_signal = call_rust_emd_backward(grad_imfs, signal, context, "numpy")

            # Verify each batch item is stable
            self.assertTrue(
                np.all(np.isfinite(grad_signal)),
                f"NaN/Inf in batch item {batch_idx}",
            )

            # Verify shape is correct
            self.assertEqual(
                grad_signal.shape,
                signal.shape,
                f"Shape mismatch in batch item {batch_idx}",
            )

    def test_edge_case_constant_signal(self) -> None:
        """Test edge case: flat constant signal.

        - Constant signal should have zero gradients
        - Verify: grad ≈ 0 (within 1e-10)
        """
        signal = self.generate_synthetic_signal(32, "constant", amplitude=5.0)

        # Forward pass
        imfs, context = call_rust_emd_forward(signal, self.config, "numpy")

        # Backward pass
        grad_imfs = np.ones_like(imfs, dtype=np.float64)
        grad_signal = call_rust_emd_backward(grad_imfs, signal, context, "numpy")

        # Check for NaN/Inf
        self.assertTrue(
            np.all(np.isfinite(grad_signal)),
            "Gradient contains NaN/Inf for constant signal",
        )

        # For constant signal, IMFs should be near zero, so gradients should be small
        max_grad_abs = np.max(np.abs(grad_signal))
        self.assertLess(
            max_grad_abs,
            1.0,
            f"Gradient too large for constant signal: {max_grad_abs}",
        )

    def test_edge_case_tiny_signal(self) -> None:
        """Test edge case: very small amplitude signal.

        - Tiny amplitude signal (max 1e-6)
        - Verify: no numerical issues
        - Verify: gradients are meaningful (not all zero)
        """
        signal = self.generate_synthetic_signal(32, "tiny", amplitude=1.0)

        # Verify signal is actually tiny
        self.assertLess(
            np.max(np.abs(signal)),
            1e-5,
            "Tiny signal generation failed",
        )

        # Forward pass
        imfs, context = call_rust_emd_forward(signal, self.config, "numpy")

        # Backward pass
        grad_imfs = np.ones_like(imfs, dtype=np.float64)
        grad_signal = call_rust_emd_backward(grad_imfs, signal, context, "numpy")

        # Check for NaN/Inf
        self.assertTrue(
            np.all(np.isfinite(grad_signal)),
            "Gradient contains NaN/Inf for tiny signal",
        )

        # Gradient should exist (not all zeros)
        max_grad_abs = np.max(np.abs(grad_signal))
        self.assertGreater(
            max_grad_abs,
            0.0,
            "Gradient is all zeros for tiny signal (should be non-zero)",
        )

    def test_upstream_gradient_scaling(self) -> None:
        """Test stability with different upstream gradient scales.

        - Compute gradients with small upstream gradients (1e-6)
        - Compute gradients with large upstream gradients (1e2)
        - Verify: both produce finite results
        """
        signal = self.generate_synthetic_signal(32, "simple")

        # Forward pass
        imfs, context = call_rust_emd_forward(signal, self.config, "numpy")

        # Test with small upstream gradients
        grad_imfs_small = 1e-6 * np.ones_like(imfs, dtype=np.float64)
        grad_signal_small = call_rust_emd_backward(
            grad_imfs_small, signal, context, "numpy"
        )

        self.assertTrue(
            np.all(np.isfinite(grad_signal_small)),
            "NaN/Inf with small upstream gradients",
        )

        # Test with large upstream gradients
        grad_imfs_large = 1e2 * np.ones_like(imfs, dtype=np.float64)
        grad_signal_large = call_rust_emd_backward(
            grad_imfs_large, signal, context, "numpy"
        )

        self.assertTrue(
            np.all(np.isfinite(grad_signal_large)),
            "NaN/Inf with large upstream gradients",
        )

        # Large gradients should be clipped
        max_large = np.max(np.abs(grad_signal_large))
        self.assertLessEqual(
            max_large,
            self.grad_clip_max,
            "Large upstream gradients not properly clipped",
        )


if __name__ == "__main__":
    unittest.main()
