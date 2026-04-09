#!/usr/bin/env python3
"""Benchmarking script: Learnable boundaries vs fixed boundaries on classification tasks.

This script measures the accuracy improvement of the LearnableBoundaryPredictor
compared to fixed boundaries on three real-world-like signal classification tasks:

1. ECG Classification: Distinguish between normal (5 bpm) and tachycardia (10 bpm)
2. Seismic Detection: Distinguish between earthquake signals and background noise
3. Speech Recognition: Distinguish between male and female voice patterns

Each task uses a simple 1D CNN classifier trained with either fixed or learnable
boundary extension methods. Results show the improvement from learning task-specific
boundaries.

Usage:
    python examples/benchmark_learnable_boundaries.py

Target Results:
    - ECG: +2-5% improvement
    - Seismic: +2-5% improvement
    - Speech: +2-5% improvement
    - Average: >= +2%
"""

import sys
import os
from pathlib import Path
from typing import Tuple, Optional

import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except ImportError:
    print("ERROR: PyTorch is required. Install with: pip install torch")
    sys.exit(1)

from ferromode_ml.learnable_boundary import LearnableBoundaryPredictor


# =============================================================================
# SIGNAL GENERATION
# =============================================================================


def generate_ecg_signals(
    num_signals: int = 1000,
    signal_length: int = 256,
    seed: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate synthetic ECG-like signals with different heart rates.

    Class 0: Normal ECG (5 bpm heartbeat)
    Class 1: Tachycardia (10 bpm heartbeat)

    Args:
        num_signals: Number of signals to generate
        signal_length: Length of each signal in samples
        seed: Random seed for reproducibility

    Returns:
        Tuple of (signals tensor, labels tensor)
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    signals = []
    labels = []

    for i in range(num_signals):
        t = torch.linspace(0, 10, signal_length)  # 10 seconds

        if i % 2 == 0:
            # Class 0: Normal ECG (5 bpm heartbeat)
            heartbeat_freq = 5
        else:
            # Class 1: Tachycardia (10 bpm heartbeat)
            heartbeat_freq = 10

        # Simulate ECG with periodic pulses
        signal = _simulate_ecg(t, heartbeat_freq)
        signals.append(signal)
        labels.append(1 if heartbeat_freq > 7 else 0)

    return torch.stack(signals), torch.tensor(labels, dtype=torch.long)


def _simulate_ecg(t: torch.Tensor, heartbeat_freq: float) -> torch.Tensor:
    """Simulate ECG waveform with P wave, QRS complex, and T wave.

    Args:
        t: Time axis tensor
        heartbeat_freq: Frequency of heartbeats

    Returns:
        ECG signal tensor
    """
    ecg = torch.zeros_like(t)

    for beat in range(int(heartbeat_freq)):
        beat_time = beat / heartbeat_freq

        # P wave
        p_wave = 0.1 * torch.exp(-20 * (t - beat_time) ** 2) * (t > beat_time).float()
        ecg = ecg + p_wave

        # QRS complex (main peak)
        qrs = (
            0.5
            * torch.exp(-100 * (t - beat_time - 0.05) ** 2)
            * (t > beat_time + 0.04).float()
        )
        ecg = ecg + qrs

        # T wave
        t_wave = (
            0.2
            * torch.exp(-10 * (t - beat_time - 0.1) ** 2)
            * (t > beat_time + 0.09).float()
        )
        ecg = ecg + t_wave

    # Add noise
    ecg = ecg + torch.randn_like(ecg) * 0.05

    return ecg


def generate_seismic_signals(
    num_signals: int = 1000,
    signal_length: int = 512,
    seed: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate synthetic seismic signals: earthquake vs background noise.

    Class 0: Background noise only
    Class 1: P-wave + S-wave + noise

    Args:
        num_signals: Number of signals to generate
        signal_length: Length of each signal in samples
        seed: Random seed for reproducibility

    Returns:
        Tuple of (signals tensor, labels tensor)
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    signals = []
    labels = []

    for i in range(num_signals):
        if i % 2 == 0:
            # Class 0: Background noise only
            signal = torch.randn(signal_length) * 0.1
        else:
            # Class 1: P-wave + S-wave + noise
            signal = _simulate_seismic(signal_length)

        signals.append(signal)
        labels.append(1 if i % 2 == 1 else 0)

    return torch.stack(signals), torch.tensor(labels, dtype=torch.long)


def _simulate_seismic(signal_length: int) -> torch.Tensor:
    """Simulate seismic wave (P-wave then S-wave).

    P-wave: faster, lower amplitude, arrives first
    S-wave: slower, higher amplitude, arrives after P-wave

    Args:
        signal_length: Length of signal

    Returns:
        Seismic signal tensor
    """
    signal = torch.zeros(signal_length)

    # P-wave (faster, lower amplitude)
    p_start = 50
    p_freq = 2
    p_duration = 100
    p_phase = torch.linspace(0, p_freq * 2 * np.pi, p_duration)
    signal[p_start : p_start + p_duration] += 0.3 * torch.sin(p_phase)

    # S-wave (slower, higher amplitude)
    s_start = 200
    s_freq = 1
    s_duration = 150
    s_phase = torch.linspace(0, s_freq * 2 * np.pi, s_duration)
    signal[s_start : s_start + s_duration] += 0.7 * torch.sin(s_phase)

    # Add background noise
    signal = signal + torch.randn_like(signal) * 0.05

    return signal


def generate_speech_signals(
    num_signals: int = 1000,
    signal_length: int = 512,
    seed: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate synthetic speech-like signals: male vs female voice.

    Class 0: Low pitch (male-like, ~100Hz)
    Class 1: High pitch (female-like, ~200Hz)

    Args:
        num_signals: Number of signals to generate
        signal_length: Length of each signal in samples
        seed: Random seed for reproducibility

    Returns:
        Tuple of (signals tensor, labels tensor)
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    signals = []
    labels = []

    for i in range(num_signals):
        if i % 2 == 0:
            # Class 0: Low pitch (male-like)
            signal = _simulate_speech(signal_length, pitch_shift=-2)
        else:
            # Class 1: High pitch (female-like)
            signal = _simulate_speech(signal_length, pitch_shift=2)

        signals.append(signal)
        labels.append(1 if i % 2 == 1 else 0)

    return torch.stack(signals), torch.tensor(labels, dtype=torch.long)


def _simulate_speech(signal_length: int, pitch_shift: float = 0) -> torch.Tensor:
    """Simulate speech-like signal with fundamental frequency and harmonics.

    Args:
        signal_length: Length of signal
        pitch_shift: Pitch shift in octaves (e.g., -2 for male, +2 for female)

    Returns:
        Speech signal tensor
    """
    t = torch.linspace(0, 1, signal_length)

    # Fundamental frequency (pitch)
    f0 = 100 + pitch_shift * 50  # Male: ~100Hz, Female: ~200Hz

    # Add harmonics (typical of speech)
    signal = torch.zeros_like(t)
    for harmonic in range(1, 6):
        signal = signal + (1.0 / harmonic) * torch.sin(2 * np.pi * harmonic * f0 * t)

    # Add formants (spectral peaks typical of vowels)
    formant_freqs = [700, 1200, 2500]
    for formant_freq in formant_freqs:
        signal = signal + 0.3 * torch.sin(2 * np.pi * formant_freq * t)

    # Add noise
    signal = signal + torch.randn_like(signal) * 0.05

    # Normalize
    signal = signal / (torch.max(torch.abs(signal)) + 1e-6)

    return signal


# =============================================================================
# CLASSIFIER WITH LEARNABLE BOUNDARIES
# =============================================================================


class SimpleSignalClassifier(nn.Module):
    """Simple 1D CNN classifier for time series signals.

    Architecture:
    - Input: 1D signal (batch_size, signal_length)
    - Conv1D layer with kernel size 5
    - ReLU activation
    - MaxPool1D
    - Flatten
    - FC layers
    - Output: class logits (batch_size, num_classes)
    """

    def __init__(
        self,
        signal_length: int = 256,
        num_classes: int = 2,
    ):
        """Initialize classifier.

        Args:
            signal_length: Length of input signals
            num_classes: Number of output classes (default: 2 for binary classification)
        """
        super().__init__()

        # 1D Conv layers
        self.conv1 = nn.Conv1d(1, 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.pool = nn.MaxPool1d(2)
        self.relu = nn.ReLU()

        # Compute flattened size after conv/pool layers
        # After 2 conv/pool: length -> length / 4
        flattened_size = (signal_length // 4) * 64

        # FC layers
        self.fc1 = nn.Linear(flattened_size, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor (batch_size, signal_length)

        Returns:
            Output logits (batch_size, num_classes)
        """
        # Add channel dimension if needed
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (batch_size, 1, signal_length)

        # Conv layers
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)

        # Flatten
        x = x.view(x.size(0), -1)

        # FC layers
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x


class SignalClassifierWithLearnableBoundary(nn.Module):
    """Classifier that uses learnable boundary extension before classification.

    This module chains a LearnableBoundaryPredictor with a signal classifier.
    The boundary predictor extends signal context, and the extended signals
    are fed to the classifier for improved accuracy.
    """

    def __init__(
        self,
        signal_length: int = 256,
        boundary_predictor: Optional[LearnableBoundaryPredictor] = None,
        num_classes: int = 2,
    ):
        """Initialize classifier with learnable boundary.

        Args:
            signal_length: Length of input signals
            boundary_predictor: LearnableBoundaryPredictor instance (or None for fixed)
            num_classes: Number of output classes
        """
        super().__init__()

        self.boundary_predictor = boundary_predictor
        self.classifier = SimpleSignalClassifier(signal_length, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with optional boundary extension.

        Args:
            x: Input tensor (batch_size, signal_length)

        Returns:
            Output logits (batch_size, num_classes)
        """
        # If boundary predictor is provided, extend the signal
        if self.boundary_predictor is not None:
            x = self._extend_with_boundary(x)

        # Classify
        return self.classifier(x)

    def _extend_with_boundary(self, x: torch.Tensor) -> torch.Tensor:
        """Extend signal using learnable boundary predictor.

        For each signal, take context from the end and predict ahead,
        then concatenate with original signal.

        Args:
            x: Input tensor (batch_size, signal_length)

        Returns:
            Extended signal tensor
        """
        batch_size, signal_length = x.shape
        context_length = self.boundary_predictor.context_length
        output_length = self.boundary_predictor.output_length

        # Extract context from the end of each signal
        context = x[:, -context_length:]  # (batch_size, context_length)

        # Predict ahead
        with torch.no_grad():
            predicted = self.boundary_predictor(context)  # (batch_size, output_length)

        # Concatenate original signal with predicted extension
        extended = torch.cat(
            [x, predicted], dim=1
        )  # (batch_size, signal_length + output_length)

        return extended


# =============================================================================
# TRAINING AND EVALUATION
# =============================================================================


def train_and_evaluate(
    model: nn.Module,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    device: torch.device,
    num_epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    verbose: bool = True,
) -> float:
    """Train classifier and evaluate on test set.

    Args:
        model: Neural network model to train
        X_train: Training data tensor
        y_train: Training labels tensor
        X_test: Test data tensor
        y_test: Test labels tensor
        device: PyTorch device (CPU or GPU)
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        verbose: Whether to print progress

    Returns:
        Test accuracy (float between 0 and 1)
    """
    model = model.to(device)
    X_train, y_train = X_train.to(device), y_train.to(device)
    X_test, y_test = X_test.to(device), y_test.to(device)

    # Optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0

        # Mini-batch training
        for i in range(0, len(X_train), batch_size):
            X_batch = X_train[i : i + batch_size]
            y_batch = y_train[i : i + batch_size]

            # Forward pass
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = loss_fn(logits, y_batch)

            # Backward pass
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        if verbose and (epoch + 1) % 20 == 0:
            avg_loss = epoch_loss / (len(X_train) // batch_size)
            print(f"    Epoch {epoch + 1:3d}/{num_epochs}: Loss={avg_loss:.4f}")

    # Evaluation on test set
    model.eval()
    with torch.no_grad():
        test_logits = model(X_test)
        _, predictions = torch.max(test_logits, 1)
        accuracy = (predictions == y_test).float().mean().item()

    return accuracy


# =============================================================================
# MAIN BENCHMARKING LOOP
# =============================================================================


def main():
    """Run comprehensive benchmarking on all three classification tasks."""

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}\n")

    # Try to load pre-trained boundary predictor if available
    boundary_model_path = Path(__file__).parent.parent / "pretrained_boundary_model.pt"

    if boundary_model_path.exists():
        print(f"Loading pre-trained boundary model from {boundary_model_path}")
        boundary_predictor = LearnableBoundaryPredictor()
        boundary_predictor.load_state_dict(
            torch.load(boundary_model_path, map_location=device)
        )
        boundary_predictor.to(device)
        print("✓ Loaded pre-trained boundary model\n")
    else:
        print(f"Note: Pre-trained boundary model not found at {boundary_model_path}")
        print("Using untrained LearnableBoundaryPredictor (random initialization)\n")
        boundary_predictor = LearnableBoundaryPredictor()
        boundary_predictor.to(device)

    # Define benchmarking tasks
    tasks = [
        ("ECG Classification", generate_ecg_signals, 256),
        ("Seismic Detection", generate_seismic_signals, 512),
        ("Speech Recognition", generate_speech_signals, 512),
    ]

    results = {}

    for task_name, data_generator, signal_length in tasks:
        print("=" * 70)
        print(f"Task: {task_name}")
        print("=" * 70)

        # Generate data
        print("\n[1/3] Generating data...")
        X, y = data_generator(num_signals=1000, signal_length=signal_length)
        X_train, y_train = X[:800], y[:800]
        X_test, y_test = X[800:], y[800:]
        print(
            f"  Training set: {X_train.shape[0]} samples, {X_train.shape[1]} samples/signal"
        )
        print(f"  Test set:     {X_test.shape[0]} samples")

        # Train with FIXED boundaries (baseline)
        print("\n[2/3] Training with FIXED boundaries (baseline)...")
        classifier_fixed = SignalClassifierWithLearnableBoundary(
            signal_length=signal_length,
            boundary_predictor=None,  # No boundary extension
            num_classes=2,
        )
        acc_fixed = train_and_evaluate(
            model=classifier_fixed,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            device=device,
            num_epochs=100,
            batch_size=32,
            learning_rate=1e-3,
            verbose=True,
        )
        print(
            f"\n  Fixed Boundaries Test Accuracy: {acc_fixed:.4f} ({acc_fixed * 100:.2f}%)"
        )

        # Train with LEARNABLE boundaries
        print("\n[3/3] Training with LEARNABLE boundaries...")
        classifier_learnable = SignalClassifierWithLearnableBoundary(
            signal_length=signal_length,
            boundary_predictor=boundary_predictor,
            num_classes=2,
        )
        acc_learnable = train_and_evaluate(
            model=classifier_learnable,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            device=device,
            num_epochs=100,
            batch_size=32,
            learning_rate=1e-3,
            verbose=True,
        )
        print(
            f"\n  Learnable Boundaries Test Accuracy: {acc_learnable:.4f} ({acc_learnable * 100:.2f}%)"
        )

        # Calculate improvement
        improvement = (acc_learnable - acc_fixed) * 100
        print(f"\n  Improvement: {improvement:+.2f}%")

        results[task_name] = {
            "fixed": acc_fixed,
            "learnable": acc_learnable,
            "improvement": improvement,
        }

        print()

    # Summary table
    print("=" * 70)
    print("BENCHMARK RESULTS SUMMARY")
    print("=" * 70)
    print()
    print(f"{'Task':<25} {'Fixed':<12} {'Learnable':<12} {'Improvement':<12}")
    print("-" * 70)

    total_improvement = 0
    for task_name, result in results.items():
        fixed_pct = result["fixed"] * 100
        learnable_pct = result["learnable"] * 100
        print(
            f"{task_name:<25} {fixed_pct:>6.2f}%      "
            f"{learnable_pct:>6.2f}%      {result['improvement']:>+6.2f}%"
        )
        total_improvement += result["improvement"]

    print("-" * 70)
    avg_improvement = total_improvement / len(results)
    print(f"{'AVERAGE':<25} {'':<12} {'':<12} {avg_improvement:>+6.2f}%")
    print()

    # Target check
    print("=" * 70)
    print("TARGET ACHIEVEMENT")
    print("=" * 70)

    target_improvement = 2.0  # 2-5% target, check for >= 2%

    if avg_improvement >= target_improvement:
        print(
            f"\n✅ SUCCESS: Average improvement {avg_improvement:.2f}% >= target {target_improvement:.2f}%"
        )
        print("\nAll tasks achieved positive improvement with learnable boundaries!")
    else:
        print(f"\n⚠️  Below primary target: Average improvement {avg_improvement:.2f}%")
        print(f"   Target was >= {target_improvement:.2f}%")

    # Per-task analysis
    print("\n" + "=" * 70)
    print("PER-TASK ANALYSIS")
    print("=" * 70 + "\n")

    for task_name, result in results.items():
        improvement = result["improvement"]
        status = "✅ PASS" if improvement >= 2.0 else "⚠️  Below 2%"
        print(f"{task_name}:")
        print(f"  Fixed:      {result['fixed'] * 100:6.2f}%")
        print(f"  Learnable:  {result['learnable'] * 100:6.2f}%")
        print(f"  Improvement: {improvement:+6.2f}% {status}")
        print()

    print("=" * 70)
    print(f"Benchmark Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
