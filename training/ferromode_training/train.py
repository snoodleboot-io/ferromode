"""
Train LSTM boundary prediction model for Ferromode.

This script generates synthetic ferromode_training data, trains an LSTM neural network
to predict signal extensions for boundary handling, and exports the quantized
model as ONNX.

Usage:
    python -m ferromode_training.train \\
        --output models/lstm_predictor.onnx \\
        --epochs 50 \\
        --batch_size 32 \\
        --learning_rate 0.001
"""

import argparse
import json
import logging
import shutil
from pathlib import Path
from typing import Dict, Optional, List, Tuple

import numpy as np

from ferromode_training.data_generator import generate_synthetic_signals

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def train_lstm_model(
    signals: List[np.ndarray],
    targets: List[np.ndarray],
    epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    window_size: int = 20,
    hidden_units: int = 128,
    num_layers: int = 2,
) -> Optional[Dict]:
    """
    Train LSTM model.

    Args:
        signals: List of ferromode_training signals
        targets: List of target predictions
        epochs: Number of ferromode_training epochs (default: 50)
        batch_size: Batch size for ferromode_training (default: 32)
        learning_rate: Learning rate (default: 0.001)
        window_size: LSTM input window size (default: 20)
        hidden_units: LSTM hidden units (default: 128)
        num_layers: Number of LSTM layers (default: 2)

    Returns:
        Dictionary with model, device, history, and best_loss, or None if failed
    """
    logger.info("[Training Phase]")
    logger.info(f"Window size: {window_size}")
    logger.info(f"Hidden units: {hidden_units}")
    logger.info(f"Number of layers: {num_layers}")
    logger.info(f"Learning rate: {learning_rate}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Epochs: {epochs}")

    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import Dataset, DataLoader
        from torch.optim import Adam

        logger.info("\n✓ PyTorch available, proceeding with ferromode_training...")

        # Create dataset
        class SignalDataset(Dataset):
            """Dataset for signal-target pairs."""

            def __init__(
                self, signals: List[np.ndarray], targets: List[np.ndarray], window_size: int = 20
            ):
                self.windows = []
                self.targets = []

                for sig, tgt in zip(signals, targets):
                    # Use last window_size samples as input
                    window = torch.from_numpy(sig[-window_size:]).float()
                    target = torch.from_numpy(tgt).float()
                    self.windows.append(window)
                    self.targets.append(target)

            def __len__(self) -> int:
                return len(self.windows)

            def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
                return self.windows[idx], self.targets[idx]

        # Define LSTM model
        class LSTMPredictor(nn.Module):
            """LSTM model for predicting signal extensions."""

            def __init__(self, hidden_size: int = 128, num_layers: int = 2):
                super().__init__()
                self.lstm = nn.LSTM(
                    input_size=1,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    batch_first=True,
                    dropout=0.2 if num_layers > 1 else 0.0,
                )
                self.fc = nn.Linear(hidden_size, 10)  # Output 10 samples

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                """Forward pass through LSTM."""
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
        model = LSTMPredictor(hidden_size=hidden_units, num_layers=num_layers).to(device)

        # Training loop
        optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
        criterion = nn.MSELoss()

        train_history: Dict[str, List] = {"loss": [], "epoch": []}
        best_loss = float("inf")
        patience_counter = 0

        logger.info(f"Training on {device}...")

        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch_idx, (windows, tgt) in enumerate(dataloader):
                windows = windows.to(device)
                tgt = tgt.to(device)

                optimizer.zero_grad()
                outputs = model(windows)
                loss = criterion(outputs, tgt)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            epoch_loss /= len(dataloader)
            train_history["loss"].append(epoch_loss)
            train_history["epoch"].append(epoch + 1)

            if (epoch + 1) % 10 == 0:
                logger.info(f"  Epoch {epoch + 1:3d}/{epochs} - Loss: {epoch_loss:.6f}")

            # Early stopping
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= 5:
                    logger.info(f"  Early stopping at epoch {epoch + 1}")
                    break

        logger.info(f"\n✓ Training complete. Best loss: {best_loss:.6f}")

        return {
            "model": model,
            "device": device,
            "history": train_history,
            "best_loss": best_loss,
        }

    except ImportError:
        logger.error("\n✗ PyTorch not available. Install with:")
        logger.error("    pip install torch")
        return None


def export_to_onnx(
    model,
    device,
    window_size: int = 20,
    output_path: str = "lstm_predictor_fp32.onnx",
) -> Optional[str]:
    """
    Export trained model to ONNX format.

    Args:
        model: Trained PyTorch model
        device: Device model is on
        window_size: Input window size (default: 20)
        output_path: Path to save ONNX model

    Returns:
        Path to exported model or None if failed
    """
    if model is None:
        logger.warning("✗ Model not trained, skipping ONNX export")
        return None

    try:
        import torch

        logger.info(f"\n[ONNX Export]")
        logger.info(f"Input shape: [1, {window_size}, 1]")
        logger.info(f"Output shape: [1, 10]")

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
        logger.info(f"✓ Exported to: {output_path} ({size:,} bytes)")

        return str(output_path)

    except ImportError:
        logger.error("✗ PyTorch not available for ONNX export")
        return None


def convert_to_fp16(fp32_path: str, fp16_path: str) -> None:
    """Convert ONNX model from FP32 to FP16 format.

    Args:
        fp32_path: Path to FP32 ONNX model
        fp16_path: Path to save FP16 model
    """
    try:
        import onnx
        from onnxruntime.transformers.float16 import convert_float_to_float16

        logger.info(f"\n[FP16 Conversion]")
        logger.info(f"Loading FP32 model: {fp32_path}")
        model = onnx.load(fp32_path)

        logger.info("Converting to FP16...")
        model_fp16 = convert_float_to_float16(model)

        # Create output directory if needed
        output_path = Path(fp16_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving FP16 model: {fp16_path}")
        onnx.save(model_fp16, fp16_path)

        # Show size reduction
        import os

        fp32_size = os.path.getsize(fp32_path) / (1024 * 1024)
        fp16_size = os.path.getsize(fp16_path) / (1024 * 1024)
        compression = 100 * (1 - fp16_size / fp32_size)

        logger.info(f"✓ FP16 conversion complete")
        logger.info(
            f"  FP32: {fp32_size:.2f} MB → FP16: {fp16_size:.2f} MB ({compression:.1f}% smaller)"
        )

    except ImportError as e:
        logger.error(f"✗ Required packages missing: {e}")
        logger.info("Keeping FP32 model instead")
        shutil.copy(fp32_path, fp16_path)
    except Exception as e:
        logger.error(f"✗ FP16 conversion failed: {e}")
        logger.info("Keeping FP32 model instead")
        shutil.copy(fp32_path, fp16_path)


def main() -> int:
    """Main ferromode_training pipeline."""
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
        default=100,
        help="Number of ferromode_training epochs (default: 100)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size (default: 16)",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.0005,
        help="Learning rate (default: 0.0005)",
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
        default="../crates/ferromode/models/lstm_predictor.onnx",
        help="Output path for quantized model",
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
        help="Skip ferromode_training (for testing quantization only)",
    )

    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("Ferromode LSTM Boundary Prediction - Training Pipeline")
    logger.info("=" * 70)

    # Generate ferromode_training data
    logger.info("\n[Data Generation]")
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
            logger.error("\n✗ Training failed")
            return 1

        model = result["model"]
        device = result["device"]

        # Export to ONNX
        fp32_path = args.output.replace(".onnx", "_fp32.onnx")
        export_to_onnx(model, device, window_size=args.window_size, output_path=fp32_path)

        # Convert to FP16
        convert_to_fp16(fp32_path, args.output)
    else:
        logger.info("\n(Skipping ferromode_training)")

    logger.info("\n" + "=" * 70)
    logger.info("✓ Training pipeline complete!")
    logger.info(f"Model ready at: {args.output}")
    logger.info("=" * 70)

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
