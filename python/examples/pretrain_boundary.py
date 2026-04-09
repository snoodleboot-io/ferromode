#!/usr/bin/env python3
"""Example: Pre-train boundary predictor on synthetic signals.

This script demonstrates how to pre-train a LearnableBoundaryPredictor on
unlabeled signals using self-supervised learning. The model learns to
reconstruct signal continuations from context.

The synthetic signals are mixtures of sinusoids with noise, representing
realistic time series data patterns.

Usage:
    python examples/pretrain_boundary.py
"""

import sys
import os
from pathlib import Path

import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import torch
except ImportError:
    print("ERROR: PyTorch is required. Install with: pip install torch")
    sys.exit(1)

from ferromode_ml.learnable_boundary import (
    LearnableBoundaryPredictor,
    PretrainBoundaryPredictorConfig,
    pretrain_boundary_predictor,
)


def generate_synthetic_signals(
    num_signals: int = 10000,
    signal_length: int = 100,
    seed: int = 42,
) -> torch.Tensor:
    """Generate synthetic training signals.

    Creates a diverse dataset of signals by mixing sinusoids with different
    frequencies and amplitudes, plus Gaussian noise. This simulates realistic
    time series data that might appear in EMD applications.

    Args:
        num_signals: Number of signals to generate (default: 10000)
        signal_length: Length of each signal in samples (default: 100)
        seed: Random seed for reproducibility (default: 42)

    Returns:
        Tensor of shape (num_signals, signal_length) containing float32 signals
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    signals = []

    for _ in range(num_signals):
        # Time axis
        t = np.linspace(0, 4 * np.pi, signal_length)

        # Random mixing of sinusoids
        freq1 = np.random.uniform(1, 10)
        freq2 = np.random.uniform(1, 10)
        amp1 = np.random.uniform(0.5, 2.0)
        amp2 = np.random.uniform(0.5, 2.0)

        # Noise
        noise = np.random.normal(0, 0.1, signal_length)

        # Combined signal
        signal = amp1 * np.sin(freq1 * t) + amp2 * np.cos(freq2 * t) + noise
        signals.append(torch.from_numpy(signal).float())

    return torch.stack(signals)


def main():
    """Pre-train boundary predictor."""
    print("=" * 70)
    print("LearnableBoundaryPredictor Pre-Training Example")
    print("=" * 70)

    # Configuration
    config = PretrainBoundaryPredictorConfig()
    config.context_length = 10
    config.output_length = 15
    config.hidden_dim = 128
    config.num_epochs = 50
    config.learning_rate = 1e-3
    config.batch_size = 32

    print("\n[1/4] Configuration")
    print(f"  Context length:  {config.context_length}")
    print(f"  Output length:   {config.output_length}")
    print(f"  Hidden dim:      {config.hidden_dim}")
    print(f"  Batch size:      {config.batch_size}")
    print(f"  Learning rate:   {config.learning_rate}")
    print(f"  Max epochs:      {config.num_epochs}")
    print(f"  Device:          {config.device}")

    # Generate synthetic signals
    print("\n[2/4] Generating synthetic signals...")
    signals = generate_synthetic_signals(num_signals=10000, signal_length=100)
    print(f"  Generated {signals.shape[0]} signals of length {signals.shape[1]}")
    print(f"  Mean: {signals.mean():.4f}, Std: {signals.std():.4f}")

    # Pre-train boundary predictor
    print("\n[3/4] Pre-training boundary predictor...")
    print("  (Training progress displayed below)")
    print()

    model = pretrain_boundary_predictor(signals, config, verbose=True)

    print()

    # Evaluate on test sample
    print("\n[4/4] Evaluation on test sample")
    test_signal = signals[0]  # Use first signal as test

    # Create context-target pair
    context = test_signal[: config.context_length]
    target = test_signal[
        config.context_length : config.context_length + config.output_length
    ]

    with torch.no_grad():
        prediction = model(context)

    # Compute error
    mse = ((prediction - target) ** 2).mean().item()
    rmse = np.sqrt(mse)

    print(f"  Test context shape:     {context.shape}")
    print(f"  Predicted shape:        {prediction.shape}")
    print(f"  Target shape:           {target.shape}")
    print(f"  MSE:                    {mse:.6f}")
    print(f"  RMSE:                   {rmse:.6f}")

    # Save model
    output_path = Path(__file__).parent.parent / "pretrained_boundary_model.pt"
    torch.save(model.state_dict(), output_path)
    print(f"\n  Model saved to: {output_path}")

    # Summary
    print("\n" + "=" * 70)
    print("✓ Pre-training complete!")
    print("=" * 70)
    print("\nNext steps:")
    print("  1. Use pretrained_boundary_model.pt in fine-tuning scripts")
    print("  2. Run examples/finetune_with_learnable_boundary.py")
    print("  3. Integrate with DifferentiableEMD for task-specific learning")
    print()


if __name__ == "__main__":
    main()
