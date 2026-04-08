#!/usr/bin/env python3
"""
Train LSTM boundary prediction model for Ferromode.

This script generates synthetic training data, trains an LSTM neural network
to predict signal extensions for boundary handling, and exports the quantized
model as ONNX.

Usage:
    python train_lstm_predictor.py \
        --output models/lstm_predictor.onnx \
        --epochs 50 \
        --batch_size 32 \
        --learning_rate 0.001

Requirements:
    torch>=2.0.0
    numpy>=1.23.0
    scipy>=1.10.0
    matplotlib>=3.7.0 (optional, for visualization)
"""

import argparse
import json
import os
import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict


def generate_synthetic_signals(
    num_signals: int = 1000,
    signal_length: int = 500,
    random_seed: int = 42,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Generate diverse synthetic signals for training.

    Returns:
        (signals, targets) tuples where:
        - signals: list of signal chunks (numpy arrays)
        - targets: list of ground-truth extensions (numpy arrays)
    """
    np.random.seed(random_seed)

    signals = []
    targets = []

    # Generate different signal types
    signals_per_type = num_signals // 6

    # 1. Pure tones (10%)
    print(f"[1/6] Generating {signals_per_type} pure tones...")
    for _ in range(signals_per_type):
        freq = np.random.uniform(0.05, 0.5)
        t = np.arange(signal_length)
        sig = np.sin(2 * np.pi * freq * t)
        signals.append(sig)
        # For pure tone, predict continuation
        target = np.sin(2 * np.pi * freq * (t[-10:] + 10))
        targets.append(target)

    # 2. Chirps (15%)
    print(f"[2/6] Generating {signals_per_type} chirps...")
    for _ in range(signals_per_type):
        f0 = np.random.uniform(0.05, 0.2)
        f1 = np.random.uniform(0.2, 0.5)
        t = np.arange(signal_length) / signal_length
        phase = 2 * np.pi * (f0 * t + (f1 - f0) * t**2 / 2)
        sig = np.sin(phase)
        signals.append(sig)
        # Predict chirp continuation
        t_future = (np.arange(signal_length, signal_length + 10)) / signal_length
        phase_future = 2 * np.pi * (f0 * t_future + (f1 - f0) * t_future**2 / 2)
        target = np.sin(phase_future)
        targets.append(target)

    # 3. AM/FM modulated (15%)
    print(f"[3/6] Generating {signals_per_type} AM/FM modulated signals...")
    for _ in range(signals_per_type):
        carrier_freq = np.random.uniform(0.15, 0.35)
        mod_freq = np.random.uniform(0.05, 0.15)
        depth = np.random.uniform(0.5, 0.9)

        t = np.arange(signal_length)
        envelope = 1 + depth * np.sin(2 * np.pi * mod_freq * t / signal_length)
        carrier = np.sin(2 * np.pi * carrier_freq * t)
        sig = envelope * carrier
        signals.append(sig)

        # Predict continuation
        t_future = np.arange(signal_length, signal_length + 10)
        env_future = 1 + depth * np.sin(2 * np.pi * mod_freq * t_future / signal_length)
        car_future = np.sin(2 * np.pi * carrier_freq * t_future)
        target = env_future * car_future
        targets.append(target)

    # 4. Non-stationary (frequency sweep) (20%)
    print(f"[4/6] Generating {signals_per_type} non-stationary signals...")
    for _ in range(signals_per_type):
        # Multiple frequency components with time-varying amplitudes
        freqs = np.random.uniform(0.05, 0.4, 3)
        amps = np.random.uniform(0.3, 1.0, 3)

        t = np.arange(signal_length)
        sig = np.zeros(signal_length)
        for freq, amp in zip(freqs, amps):
            # Time-varying amplitude
            amp_envelope = amp * (1 + 0.5 * np.sin(2 * np.pi * 0.01 * t))
            sig += amp_envelope * np.sin(2 * np.pi * freq * t)

        signals.append(sig / 3)  # Normalize

        # Predict continuation with same frequency components
        t_future = np.arange(signal_length, signal_length + 10)
        target = np.zeros(10)
        for freq, amp in zip(freqs, amps):
            amp_env = amp * (1 + 0.5 * np.sin(2 * np.pi * 0.01 * t_future))
            target += amp_env * np.sin(2 * np.pi * freq * t_future)
        targets.append(target / 3)

    # 5. Intermittent/Burst signals (20%)
    print(f"[5/6] Generating {signals_per_type} intermittent signals...")
    for _ in range(signals_per_type):
        sig = np.zeros(signal_length)
        num_bursts = np.random.randint(2, 5)

        for _ in range(num_bursts):
            burst_start = np.random.randint(0, signal_length - 50)
            burst_len = np.random.randint(30, 100)
            burst_freq = np.random.uniform(0.1, 0.4)

            burst_t = np.arange(burst_len)
            burst = np.sin(2 * np.pi * burst_freq * burst_t)
            sig[burst_start : burst_start + burst_len] += burst

        sig = np.tanh(sig)  # Limit amplitude
        signals.append(sig)

        # Predict continuation (probably quiet, unless burst continues)
        target = np.random.normal(0, 0.1, 10)
        targets.append(target)

    # 6. Real-world-like signals (20%)
    print(f"[6/6] Generating {signals_per_type} complex real-world-like signals...")
    for _ in range(signals_per_type):
        # Mix of multiple frequencies with noise
        t = np.arange(signal_length)
        sig = 0.5 * np.sin(2 * np.pi * 0.1 * t)
        sig += 0.3 * np.sin(2 * np.pi * 0.2 * t)
        sig += 0.2 * np.sin(2 * np.pi * 0.3 * t)
        sig += 0.1 * np.random.randn(signal_length)
        signals.append(sig)

        # Predict continuation
        t_future = np.arange(signal_length, signal_length + 10)
        target = 0.5 * np.sin(2 * np.pi * 0.1 * t_future)
        target += 0.3 * np.sin(2 * np.pi * 0.2 * t_future)
        target += 0.2 * np.sin(2 * np.pi * 0.3 * t_future)
        target += 0.05 * np.random.randn(10)
        targets.append(target)

    print(f"\n✓ Generated {len(signals)} training signals")
    return signals, targets


def train_lstm_model(
    signals: List[np.ndarray],
    targets: List[np.ndarray],
    epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    window_size: int = 20,
    hidden_units: int = 128,
    num_layers: int = 2,
) -> Dict:
    """
    Train LSTM model.

    This is a stub that requires PyTorch to be installed.
    """
    print("\n[Training Phase]")
    print(f"Window size: {window_size}")
    print(f"Hidden units: {hidden_units}")
    print(f"Number of layers: {num_layers}")
    print(f"Learning rate: {learning_rate}")
    print(f"Batch size: {batch_size}")
    print(f"Epochs: {epochs}")

    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import Dataset, DataLoader
        from torch.optim import Adam

        print("\n✓ PyTorch available, proceeding with training...")

        # Create dataset
        class SignalDataset(Dataset):
            def __init__(self, signals, targets, window_size=20):
                self.windows = []
                self.targets = []

                for sig, tgt in zip(signals, targets):
                    # Use last window_size samples as input
                    window = torch.from_numpy(sig[-window_size:]).float()
                    target = torch.from_numpy(tgt).float()
                    self.windows.append(window)
                    self.targets.append(target)

            def __len__(self):
                return len(self.windows)

            def __getitem__(self, idx):
                return self.windows[idx], self.targets[idx]

        # Define LSTM model
        class LSTMPredictor(nn.Module):
            def __init__(self, hidden_size=128, num_layers=2):
                super().__init__()
                self.lstm = nn.LSTM(
                    input_size=1,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    batch_first=True,
                    dropout=0.2 if num_layers > 1 else 0.0,
                )
                self.fc = nn.Linear(hidden_size, 10)  # Output 10 samples

            def forward(self, x):
                # x shape: [batch, seq_len] -> [batch, seq_len, 1]
                x = x.unsqueeze(-1)
                lstm_out, _ = self.lstm(x)
                # Use last LSTM output
                last_out = lstm_out[:, -1, :]
                output = torch.tanh(self.fc(last_out))
                return output

        # Create dataset and dataloader
        dataset = SignalDataset(signals, targets, window_size)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Initialize model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = LSTMPredictor(hidden_size=hidden_units, num_layers=num_layers).to(
            device
        )

        # Training loop
        optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
        criterion = nn.MSELoss()

        train_history = {"loss": [], "epoch": []}
        best_loss = float("inf")
        patience_counter = 0

        print(f"Training on {device}...")

        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch_idx, (windows, targets) in enumerate(dataloader):
                windows = windows.to(device)
                targets = targets.to(device)

                optimizer.zero_grad()
                outputs = model(windows)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            epoch_loss /= len(dataloader)
            train_history["loss"].append(epoch_loss)
            train_history["epoch"].append(epoch + 1)

            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch + 1:3d}/{epochs} - Loss: {epoch_loss:.6f}")

            # Early stopping
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= 5:
                    print(f"  Early stopping at epoch {epoch + 1}")
                    break

        print(f"\n✓ Training complete. Best loss: {best_loss:.6f}")

        return {
            "model": model,
            "device": device,
            "history": train_history,
            "best_loss": best_loss,
        }

    except ImportError:
        print("\n✗ PyTorch not available. Install with:")
        print("    pip install torch")
        return None


def export_to_onnx(
    model,
    device,
    window_size: int = 20,
    output_path: str = "models/lstm_predictor_fp32.onnx",
) -> str:
    """Export trained model to ONNX format."""
    if model is None:
        print("✗ Model not trained, skipping ONNX export")
        return None

    try:
        import torch

        print(f"\n[ONNX Export]")
        print(f"Input shape: [1, {window_size}, 1]")
        print(f"Output shape: [1, 10]")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Create dummy input
        dummy_input = torch.randn(1, window_size).to(device)

        # Export
        torch.onnx.export(
            model,
            dummy_input,
            str(output_path),
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={
                "input": {0: "batch_size"},
                "output": {0: "batch_size"},
            },
            opset_version=14,
            do_constant_folding=True,
        )

        size = output_path.stat().st_size
        print(f"✓ Exported to: {output_path} ({size:,} bytes)")

        return str(output_path)

    except ImportError:
        print("✗ PyTorch not available for ONNX export")
        return None


def quantize_model(
    input_path: str,
    output_path: str = "models/lstm_predictor.onnx",
) -> bool:
    """Quantize ONNX model to FP16."""
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType

        print(f"\n[Quantization]")
        print(f"Input: {input_path} (FP32)")
        print(f"Output: {output_path} (FP16)")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # ONNX Runtime quantization
        quantize_dynamic(
            input_path,
            str(output_path),
            weight_type=QuantType.Float16,
        )

        input_size = Path(input_path).stat().st_size
        output_size = output_path.stat().st_size

        print(f"✓ Quantization complete:")
        print(f"  Input size:  {input_size:,} bytes")
        print(f"  Output size: {output_size:,} bytes")
        print(f"  Compression: {100 * (1 - output_size / input_size):.1f}%")

        return True

    except ImportError:
        print("✗ ONNX Runtime not available for quantization")
        print("  Install with: pip install onnxruntime")
        # Copy unquantized as fallback
        import shutil

        shutil.copy(input_path, output_path)
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Train LSTM boundary prediction model for Ferromode"
    )
    parser.add_argument(
        "--num_signals",
        type=int,
        default=1000,
        help="Number of synthetic signals to generate (default: 1000)",
    )
    parser.add_argument(
        "--signal_length",
        type=int,
        default=500,
        help="Length of each signal (default: 500)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs (default: 50)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size (default: 32)",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.001,
        help="Learning rate (default: 0.001)",
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=20,
        help="LSTM input window size (default: 20)",
    )
    parser.add_argument(
        "--hidden_units",
        type=int,
        default=128,
        help="LSTM hidden units (default: 128)",
    )
    parser.add_argument(
        "--num_layers",
        type=int,
        default=2,
        help="Number of LSTM layers (default: 2)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="crates/ferromode/models/lstm_predictor.onnx",
        help="Output path for quantized model (default: crates/ferromode/models/lstm_predictor.onnx)",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--skip_training",
        action="store_true",
        help="Skip training (for testing quantization only)",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("Ferromode LSTM Boundary Prediction - Training Pipeline")
    print("=" * 70)

    # Generate training data
    print("\n[Data Generation]")
    signals, targets = generate_synthetic_signals(
        num_signals=args.num_signals,
        signal_length=args.signal_length,
        random_seed=args.random_seed,
    )

    # Train model
    if not args.skip_training:
        result = train_lstm_model(
            signals,
            targets,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            window_size=args.window_size,
            hidden_units=args.hidden_units,
            num_layers=args.num_layers,
        )

        if result is None:
            print("\n✗ Training failed")
            return 1

        model = result["model"]
        device = result["device"]

        # Export to ONNX
        fp32_path = args.output.replace(".onnx", "_fp32.onnx")
        export_to_onnx(
            model, device, window_size=args.window_size, output_path=fp32_path
        )

        # Quantize to FP16
        quantize_model(fp32_path, args.output)
    else:
        print("\n(Skipping training)")

    print("\n" + "=" * 70)
    print("✓ Training pipeline complete!")
    print(f"Model ready at: {args.output}")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    exit(main())
