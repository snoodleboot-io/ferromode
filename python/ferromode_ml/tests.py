"""Tests for differentiable EMD layers in Keras and PyTorch.

This module contains comprehensive tests for:
- TensorFlow/Keras DifferentiableEMDLayer
- PyTorch DifferentiableEMD module
- Gradient flow through both implementations
- Batch processing
"""

import sys
import unittest
from typing import Any, List, Optional, Tuple
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

from ferromode_ml.rust_bridge import call_rust_emd_backward, call_rust_emd_forward


class TestRustBridge(unittest.TestCase):
    """Tests for rust_bridge module."""

    def setUp(self) -> None:
        """Set up test signals."""
        np.random.seed(42)
        self.signal = np.array([1.0, 2.0, 3.0, 4.0, 3.0, 2.0, 1.0], dtype=np.float64)
        self.signal_long = np.sin(np.linspace(0, 4 * np.pi, 256)).astype(np.float64)
        self.config = {"max_imfs": 3}

    def test_forward_basic_shape(self) -> None:
        """Test forward pass returns correct shape."""
        imfs, context = call_rust_emd_forward(self.signal, self.config, "numpy")

        # IMFs should be 2D: (num_imfs, signal_length)
        self.assertEqual(len(imfs.shape), 2)
        self.assertEqual(imfs.shape[1], len(self.signal))

        # Context should be a dict
        self.assertIsInstance(context, dict)
        self.assertIn("signal_length", context)

    def test_forward_reconstruction(self) -> None:
        """Test that IMFs + residue = original signal."""
        imfs, context = call_rust_emd_forward(self.signal_long, self.config, "numpy")
        residue = context["residue"]

        # Reconstruct
        reconstructed = np.sum(imfs, axis=0) + np.array(residue)

        # Check error
        error = np.max(np.abs(self.signal_long - reconstructed))
        self.assertLess(error, 1e-8, f"Reconstruction error {error} exceeds 1e-8")

    def test_forward_no_nans(self) -> None:
        """Test forward pass produces no NaN values."""
        imfs, context = call_rust_emd_forward(self.signal_long, self.config, "numpy")
        residue = context["residue"]

        # Check IMFs
        self.assertTrue(np.all(np.isfinite(imfs)), "IMFs contain NaN or Inf")

        # Check residue
        self.assertTrue(np.all(np.isfinite(residue)), "Residue contains NaN or Inf")

    def test_backward_shape(self) -> None:
        """Test backward pass returns correct shape."""
        imfs, context = call_rust_emd_forward(self.signal_long, self.config, "numpy")

        # Create dummy upstream gradients
        grad_imfs = np.random.randn(*imfs.shape)

        # Call backward
        grad_signal = call_rust_emd_backward(
            grad_imfs, self.signal_long, context, "numpy"
        )

        # Shape should match signal
        self.assertEqual(grad_signal.shape, (len(self.signal_long),))

    def test_backward_finite(self) -> None:
        """Test backward pass produces finite gradients."""
        imfs, context = call_rust_emd_forward(self.signal_long, self.config, "numpy")
        grad_imfs = np.random.randn(*imfs.shape)

        grad_signal = call_rust_emd_backward(
            grad_imfs, self.signal_long, context, "numpy"
        )

        self.assertTrue(
            np.all(np.isfinite(grad_signal)), "Gradients contain NaN or Inf"
        )


@unittest.skipIf(not TF_AVAILABLE, "TensorFlow not installed")
class TestKerasLayer(unittest.TestCase):
    """Tests for Keras DifferentiableEMDLayer."""

    def setUp(self) -> None:
        """Set up Keras layer and test data."""
        from ferromode_ml.keras_emd import DifferentiableEMDLayer

        np.random.seed(42)
        tf.random.set_seed(42)

        self.layer = DifferentiableEMDLayer(max_imfs=3)
        self.batch_signals = tf.constant(
            np.sin(np.linspace(0, 8 * np.pi, 128)).reshape(1, 128).astype(np.float32)
        )

    def test_layer_call_single_signal(self) -> None:
        """Test layer forward pass with single signal."""
        signal = tf.constant(np.sin(np.linspace(0, 4 * np.pi, 100)).astype(np.float32))

        output = self.layer(signal)

        # Output should be 2D: (num_imfs, signal_length)
        self.assertEqual(len(output.shape), 2)
        self.assertEqual(output.shape[1], 100)

    def test_layer_call_batch(self) -> None:
        """Test layer forward pass with batch of signals."""
        batch_size = 4
        signal_length = 128
        batch = tf.random.normal((batch_size, signal_length))

        output = self.layer(batch)

        # Output should be 3D: (batch_size, num_imfs, signal_length)
        self.assertEqual(len(output.shape), 3)
        self.assertEqual(output.shape[0], batch_size)
        self.assertEqual(output.shape[2], signal_length)

    def test_layer_gradients(self) -> None:
        """Test that gradients flow through layer."""
        signal = tf.Variable(np.sin(np.linspace(0, 4 * np.pi, 100)).astype(np.float32))

        with tf.GradientTape() as tape:
            output = self.layer(signal)
            loss = tf.reduce_sum(output)

        grad = tape.gradient(loss, signal)

        self.assertIsNotNone(grad, "Gradients should not be None")
        self.assertEqual(grad.shape, signal.shape)
        self.assertTrue(tf.reduce_all(tf.math.is_finite(grad)), "Gradients contain NaN")

    def test_layer_config(self) -> None:
        """Test layer serialization config."""
        config = self.layer.get_config()

        self.assertIn("max_imfs", config)
        self.assertEqual(config["max_imfs"], 3)


@unittest.skipIf(not TORCH_AVAILABLE, "PyTorch not installed")
class TestTorchModule(unittest.TestCase):
    """Tests for PyTorch DifferentiableEMD module."""

    def setUp(self) -> None:
        """Set up PyTorch module and test data."""
        from ferromode_ml.torch_emd import DifferentiableEMD

        torch.manual_seed(42)
        np.random.seed(42)

        self.module = DifferentiableEMD(max_imfs=3)
        self.module.eval()

    def test_module_call_single_signal(self) -> None:
        """Test module forward pass with single signal."""
        signal = torch.from_numpy(
            np.sin(np.linspace(0, 4 * np.pi, 100)).astype(np.float32)
        )

        output = self.module(signal)

        # Output should be 2D: (num_imfs, signal_length)
        self.assertEqual(len(output.shape), 2)
        self.assertEqual(output.shape[1], 100)

    def test_module_call_batch(self) -> None:
        """Test module forward pass with batch."""
        batch_size = 4
        signal_length = 128
        batch = torch.randn(batch_size, signal_length)

        output = self.module(batch)

        # Output should be 3D: (batch_size, num_imfs, signal_length)
        self.assertEqual(len(output.shape), 3)
        self.assertEqual(output.shape[0], batch_size)
        self.assertEqual(output.shape[2], signal_length)

    def test_module_gradients(self) -> None:
        """Test that gradients flow through module."""
        signal = torch.randn(128, requires_grad=True)

        output = self.module(signal)
        loss = output.sum()
        loss.backward()

        self.assertIsNotNone(signal.grad, "Gradients should not be None")
        self.assertEqual(signal.grad.shape, signal.shape)
        self.assertTrue(torch.all(torch.isfinite(signal.grad)), "Gradients contain NaN")

    def test_module_in_network(self) -> None:
        """Test module as part of a neural network."""
        from ferromode_ml.torch_emd import DifferentiableEMD

        class SimpleNet(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.emd = DifferentiableEMD(max_imfs=3)
                self.fc = torch.nn.Linear(3 * 64, 10)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                imfs = self.emd(x)  # (batch, 3, 64)
                imfs_flat = imfs.reshape(imfs.shape[0], -1)  # (batch, 192)
                return self.fc(imfs_flat)

        model = SimpleNet()
        batch = torch.randn(4, 64)

        output = model(batch)

        self.assertEqual(output.shape, (4, 10))

        # Test gradients through whole network
        loss = output.sum()
        loss.backward()

        for param in model.parameters():
            if param.requires_grad:
                self.assertTrue(param.grad is not None, "Parameter gradient is None")

    def test_module_extra_repr(self) -> None:
        """Test module string representation."""
        repr_str = self.module.extra_repr()
        self.assertIn("max_imfs", repr_str)
        self.assertIn("boundary", repr_str)


class TestBatchProcessing(unittest.TestCase):
    """Tests for batch processing across frameworks."""

    def setUp(self) -> None:
        """Set up test data."""
        np.random.seed(42)
        self.batch_size = 4
        self.signal_length = 128
        self.batch_np = np.random.randn(self.batch_size, self.signal_length).astype(
            np.float64
        )

    @unittest.skipIf(not TF_AVAILABLE, "TensorFlow not installed")
    def test_keras_batch_consistency(self) -> None:
        """Test Keras layer gives consistent results across batch."""
        from ferromode_ml.keras_emd import DifferentiableEMDLayer

        layer = DifferentiableEMDLayer(max_imfs=3)

        # Process batch
        batch_tf = tf.constant(self.batch_np.astype(np.float32))
        batch_output = layer(batch_tf)

        # Process individually
        for i in range(self.batch_size):
            signal = tf.constant(self.batch_np[i].astype(np.float32))
            single_output = layer(signal)

            # Shapes should match (after accounting for batch dim)
            self.assertEqual(single_output.shape[0], batch_output.shape[1])
            self.assertEqual(single_output.shape[1], batch_output.shape[2])

    @unittest.skipIf(not TORCH_AVAILABLE, "PyTorch not installed")
    def test_torch_batch_consistency(self) -> None:
        """Test PyTorch module processes batch correctly."""
        from ferromode_ml.torch_emd import DifferentiableEMD

        module = DifferentiableEMD(max_imfs=3)
        module.eval()

        # Process batch
        batch_torch = torch.from_numpy(self.batch_np.astype(np.float32))
        batch_output = module(batch_torch)

        # Process individually
        for i in range(self.batch_size):
            signal = torch.from_numpy(self.batch_np[i].astype(np.float32))
            single_output = module(signal)

            # Shapes should match
            self.assertEqual(single_output.shape[0], batch_output.shape[1])
            self.assertEqual(single_output.shape[1], batch_output.shape[2])


class TestNumericalStability(unittest.TestCase):
    """Tests for numerical stability."""

    def test_very_small_signal(self) -> None:
        """Test with very small amplitude signal."""
        signal = np.array([1e-8, 2e-8, 3e-8, 2e-8, 1e-8], dtype=np.float64)
        config = {"max_imfs": 2}

        imfs, context = call_rust_emd_forward(signal, config, "numpy")

        self.assertTrue(np.all(np.isfinite(imfs)))
        self.assertTrue(np.all(np.isfinite(context["residue"])))

    def test_large_amplitude_signal(self) -> None:
        """Test with large amplitude signal."""
        signal = np.array([1e8, 2e8, 3e8, 2e8, 1e8], dtype=np.float64)
        config = {"max_imfs": 2}

        imfs, context = call_rust_emd_forward(signal, config, "numpy")

        self.assertTrue(np.all(np.isfinite(imfs)))
        self.assertTrue(np.all(np.isfinite(context["residue"])))

    def test_constant_signal(self) -> None:
        """Test with constant signal (should be residue)."""
        signal = np.ones(100, dtype=np.float64)
        config = {"max_imfs": 2}

        imfs, context = call_rust_emd_forward(signal, config, "numpy")

        # All should be residue, no IMFs
        self.assertTrue(np.all(np.isfinite(context["residue"])))


def run_tests() -> None:
    """Run all tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestRustBridge))
    if TF_AVAILABLE:
        suite.addTests(loader.loadTestsFromTestCase(TestKerasLayer))
    if TORCH_AVAILABLE:
        suite.addTests(loader.loadTestsFromTestCase(TestTorchModule))
    suite.addTests(loader.loadTestsFromTestCase(TestBatchProcessing))
    suite.addTests(loader.loadTestsFromTestCase(TestNumericalStability))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(run_tests())
