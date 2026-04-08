"""
Validate trained LSTM boundary prediction model.

This script:
1. Loads the trained ONNX model
2. Runs inference on the test dataset
3. Computes performance metrics (MSE, MAE, R²)
4. Generates validation report
5. Benchmarks inference speed

Usage:
    python -m training.validate \\
        --model lstm_predictor.onnx \\
        --output validation_report.json \\
        --benchmark
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class ModelValidator:
    """Validate LSTM boundary prediction model."""

    def __init__(self, model_path: str):
        """
        Initialize validator with ONNX model.

        Args:
            model_path: Path to ONNX model file
        """
        self.model_path = Path(model_path)
        self.session = None
        self.input_name = None
        self.output_name = None

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        self._load_model()

    def _load_model(self) -> None:
        """Load ONNX model using ONNX Runtime."""
        try:
            import onnxruntime as ort

            logger.info(f"Loading model: {self.model_path}")

            # Create session
            self.session = ort.InferenceSession(
                str(self.model_path), providers=["CPUExecutionProvider"]
            )

            # Get input/output names
            input_info = self.session.get_inputs()[0]
            output_info = self.session.get_outputs()[0]

            self.input_name = input_info.name
            self.output_name = output_info.name

            # Log model info
            logger.info(f"✓ Model loaded successfully")
            logger.info(f"  Input: {self.input_name} (shape: {input_info.shape})")
            logger.info(f"  Output: {self.output_name} (shape: {output_info.shape})")
            logger.info(f"  File size: {self.model_path.stat().st_size:,} bytes")

        except ImportError:
            logger.error("ONNX Runtime not available. Install with: pip install onnxruntime")
            raise

    def run_inference(self, signal: np.ndarray) -> np.ndarray:
        """
        Run model inference on a single signal.

        Args:
            signal: Input signal (1D array)

        Returns:
            Model output (predictions)
        """
        # Prepare input: [1, seq_length, 1]
        input_data = signal.reshape(1, -1, 1).astype(np.float32)

        # Run inference
        outputs = self.session.run([self.output_name], {self.input_name: input_data})

        return outputs[0]

    def validate_on_dataset(
        self,
        signals: np.ndarray,
        targets: np.ndarray,
        split: Tuple[float, float, float] = (0.8, 0.1, 0.1),
    ) -> Dict[str, float]:
        """
        Validate model on dataset.

        Args:
            signals: Array of signals (N, seq_len)
            targets: Array of target values (N,)
            split: Train/val/test split ratios

        Returns:
            Dictionary of validation metrics
        """
        logger.info("\nValidating model...")

        # Split dataset
        n_samples = len(signals)
        n_train = int(n_samples * split[0])
        n_val = int(n_samples * split[1])
        n_test = n_samples - n_train - n_val

        # Use test set for validation
        test_start = n_train + n_val
        test_signals = signals[test_start:]
        test_targets = targets[test_start:]

        logger.info(f"Test set size: {n_test} signals")

        # Run inference on test set
        predictions = []
        inference_times = []

        for i, signal in enumerate(test_signals):
            # Time inference
            start_time = time.perf_counter()
            pred = self.run_inference(signal)
            inference_time = (time.perf_counter() - start_time) * 1000  # ms

            predictions.append(pred[0, 0])  # Extract scalar prediction
            inference_times.append(inference_time)

            if (i + 1) % max(1, n_test // 10) == 0:
                logger.info(f"  Processed {i + 1}/{n_test} signals")

        predictions = np.array(predictions)

        # Compute metrics
        metrics = self._compute_metrics(test_targets, predictions, inference_times)

        return metrics

    @staticmethod
    def _compute_metrics(
        targets: np.ndarray, predictions: np.ndarray, inference_times: list
    ) -> Dict[str, float]:
        """
        Compute validation metrics.

        Args:
            targets: Ground truth targets (0 or 1 for binary, or continuous scores)
            predictions: Model predictions
            inference_times: List of inference times in ms

        Returns:
            Dictionary of metrics
        """
        # Rescale predictions from [-1, 1] to [0, 1]
        pred_scores = (predictions + 1) / 2

        # Ensure predictions are in valid range
        pred_scores = np.clip(pred_scores, 0, 1)

        # Binary classification metrics (if targets are binary)
        if np.all(np.isin(targets, [0, 1])):
            # Convert continuous predictions to binary (threshold 0.5)
            pred_binary = (pred_scores >= 0.5).astype(int)

            # Accuracy
            accuracy = np.mean(pred_binary == targets)

            # Precision, recall, F1
            tp = np.sum((pred_binary == 1) & (targets == 1))
            fp = np.sum((pred_binary == 1) & (targets == 0))
            fn = np.sum((pred_binary == 0) & (targets == 1))
            tn = np.sum((pred_binary == 0) & (targets == 0))

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1_score = (
                2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            )

            metrics = {
                "accuracy": float(accuracy),
                "precision": float(precision),
                "recall": float(recall),
                "f1_score": float(f1_score),
            }
        else:
            # Use predictions directly for scoring
            metrics = {}

        # Continuous metrics
        mse = np.mean((targets - pred_scores) ** 2)
        mae = np.mean(np.abs(targets - pred_scores))
        rmse = np.sqrt(mse)

        # R² score (for regression)
        ss_tot = np.sum((targets - np.mean(targets)) ** 2)
        ss_res = np.sum((targets - pred_scores) ** 2)
        r2_score = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        metrics.update(
            {
                "mse": float(mse),
                "mae": float(mae),
                "rmse": float(rmse),
                "r2_score": float(r2_score),
            }
        )

        # Inference speed metrics
        inference_times_arr = np.array(inference_times)
        metrics.update(
            {
                "inference_time_min_ms": float(inference_times_arr.min()),
                "inference_time_max_ms": float(inference_times_arr.max()),
                "inference_time_mean_ms": float(inference_times_arr.mean()),
                "inference_time_std_ms": float(inference_times_arr.std()),
            }
        )

        return metrics

    def benchmark_inference(self, signal_length: int = 1000, n_runs: int = 100) -> Dict:
        """
        Benchmark inference speed.

        Args:
            signal_length: Length of test signal
            n_runs: Number of inference runs

        Returns:
            Dictionary with benchmark results
        """
        logger.info(f"\nBenchmarking inference speed ({n_runs} runs)...")

        # Create dummy signal
        dummy_signal = np.random.randn(signal_length).astype(np.float32)

        times = []
        for i in range(n_runs):
            start_time = time.perf_counter()
            self.run_inference(dummy_signal)
            elapsed = (time.perf_counter() - start_time) * 1000  # ms
            times.append(elapsed)

            if (i + 1) % max(1, n_runs // 10) == 0:
                logger.info(f"  {i + 1}/{n_runs} runs completed")

        times_arr = np.array(times)

        benchmark = {
            "signal_length": signal_length,
            "n_runs": n_runs,
            "min_ms": float(times_arr.min()),
            "max_ms": float(times_arr.max()),
            "mean_ms": float(times_arr.mean()),
            "median_ms": float(np.median(times_arr)),
            "std_ms": float(times_arr.std()),
            "p95_ms": float(np.percentile(times_arr, 95)),
            "p99_ms": float(np.percentile(times_arr, 99)),
        }

        logger.info(f"✓ Benchmark complete:")
        logger.info(f"  Mean: {benchmark['mean_ms']:.3f} ms")
        logger.info(f"  Median: {benchmark['median_ms']:.3f} ms")
        logger.info(f"  P95: {benchmark['p95_ms']:.3f} ms")
        logger.info(f"  P99: {benchmark['p99_ms']:.3f} ms")

        return benchmark


def main() -> int:
    """Main validation pipeline."""
    parser = argparse.ArgumentParser(description="Validate trained LSTM boundary prediction model")
    parser.add_argument("--model", type=str, required=True, help="Path to ONNX model file")
    parser.add_argument("--data", type=str, help="Path to training dataset (NPZ format)")
    parser.add_argument(
        "--output",
        type=str,
        default="validation_report.json",
        help="Output path for validation report",
    )
    parser.add_argument("--benchmark", action="store_true", help="Run inference speed benchmark")
    parser.add_argument(
        "--benchmark_runs",
        type=int,
        default=100,
        help="Number of benchmark runs (default: 100)",
    )

    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("LSTM Model Validation")
    logger.info("=" * 70)

    try:
        # Initialize validator
        validator = ModelValidator(args.model)

        # Load dataset if provided
        report = {
            "model_path": str(args.model),
            "model_size_bytes": Path(args.model).stat().st_size,
        }

        if args.data:
            logger.info(f"\nLoading dataset: {args.data}")
            data = np.load(args.data)
            signals = data["signals"]
            stationarity_scores = data["stationarity_scores"]

            logger.info(f"Dataset shape: {signals.shape}")
            logger.info(f"Stationarity scores shape: {stationarity_scores.shape}")

            # Validate on dataset
            metrics = validator.validate_on_dataset(signals, stationarity_scores)
            report["metrics"] = metrics

            logger.info(f"\n✓ Validation metrics:")
            for key, value in metrics.items():
                logger.info(f"  {key}: {value:.6f}")

        # Run inference benchmark
        if args.benchmark:
            benchmark = validator.benchmark_inference(n_runs=args.benchmark_runs)
            report["benchmark"] = benchmark

        # Save report
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)

        logger.info(f"\n✓ Validation report saved to: {output_path}")
        logger.info("=" * 70)

        return 0

    except Exception as e:
        logger.error(f"✗ Validation failed: {e}")
        return 1


if __name__ == "__main__":
    import sys

    sys.exit(main())
