"""Unit tests for data generator module."""

import pytest
import numpy as np
from training.data_generator import generate_synthetic_signals


class TestDataGenerator:
    """Test synthetic signal generation."""

    def test_generate_signals_default_params(self):
        """Test signal generation with default parameters."""
        signals, targets = generate_synthetic_signals()

        assert len(signals) == 1000
        assert len(targets) == 1000
        assert all(isinstance(s, np.ndarray) for s in signals)
        assert all(isinstance(t, np.ndarray) for t in targets)

    def test_generate_signals_custom_params(self):
        """Test signal generation with custom parameters."""
        signals, targets = generate_synthetic_signals(
            num_signals=100, signal_length=250, random_seed=123
        )

        assert len(signals) == 100
        assert len(targets) == 100
        assert all(len(s) == 250 for s in signals)

    def test_generate_signals_reproducible(self):
        """Test that the same seed produces the same signals."""
        signals1, targets1 = generate_synthetic_signals(num_signals=10, random_seed=42)
        signals2, targets2 = generate_synthetic_signals(num_signals=10, random_seed=42)

        for s1, s2 in zip(signals1, signals2):
            np.testing.assert_array_almost_equal(s1, s2)

        for t1, t2 in zip(targets1, targets2):
            np.testing.assert_array_almost_equal(t1, t2)

    def test_signal_properties(self):
        """Test properties of generated signals."""
        signals, targets = generate_synthetic_signals(num_signals=100, signal_length=500)

        # All signals should have the correct length
        assert all(len(s) == 500 for s in signals)

        # All targets should be 10 samples
        assert all(len(t) == 10 for t in targets)

        # Signals should be reasonable (not NaN or inf)
        for signal in signals:
            assert np.all(np.isfinite(signal))

        # Targets should be reasonable
        for target in targets:
            assert np.all(np.isfinite(target))

    def test_signal_range(self):
        """Test that signal values are in reasonable range."""
        signals, targets = generate_synthetic_signals(num_signals=50)

        for signal in signals:
            # Signals should be roughly in [-2, 2] range (with some margin)
            assert np.max(np.abs(signal)) < 5

        for target in targets:
            # Targets should be roughly in [-2, 2] range
            assert np.max(np.abs(target)) < 5

    def test_different_signal_types(self):
        """Test that different signal types are generated."""
        signals, targets = generate_synthetic_signals(num_signals=600)

        # Should have diversity in signals
        signal_norms = [np.linalg.norm(s) for s in signals]
        assert np.std(signal_norms) > 0  # Should have variation
