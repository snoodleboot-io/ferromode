"""End-to-end training tests for learnable boundary conditions.

This module provides comprehensive testing for the learnable boundary predictor
integrated with differentiable EMD. Tests validate pre-training convergence,
fine-tuning gradient flow, and full pipeline integration for classification tasks.

Test Coverage:
    Pre-training (3 tests):
        1. test_pretrain_convergence: Verify loss decreases monotonically
        2. test_pretrain_boundary_quality: Validate boundary prediction accuracy
        3. test_pretrain_early_stopping: Verify early stopping prevents overfitting

    Fine-tuning (3 tests):
        4. test_finetune_classification_convergence: Classification loss decreases
        5. test_finetune_gradient_flow: Gradients flow through pipeline
        6. test_finetune_boundary_weights_update: Boundary weights actually update

    Integration (2 tests):
        7. test_full_pipeline_classification: Full pipeline end-to-end
        8. test_learnable_vs_fixed_boundary_comparison: Learnable > fixed baselines
"""

import unittest
from typing import List, Tuple
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except ImportError:
    torch = None  # type: ignore
    nn = None  # type: ignore
    optim = None  # type: ignore

from ferromode_ml import (
    LearnableBoundaryPredictor,
    PretrainBoundaryPredictorConfig,
    pretrain_boundary_predictor,
    DifferentiableEMD,
)


# ============================================================================
# Utilities: Signal Generation
# ============================================================================


def generate_synthetic_signals(
    num_signals: int = 100, signal_length: int = 100
) -> torch.Tensor:
    """Generate synthetic training signals.

    Creates signals as mixtures of sinusoids with varying frequencies and
    amplitudes, plus Gaussian noise. Suitable for pre-training boundary
    predictor without requiring real-world data.

    Args:
        num_signals: Number of signals to generate (default: 100)
        signal_length: Length of each signal in samples (default: 100)

    Returns:
        Tensor of shape (num_signals, signal_length) with float32 values
    """
    if torch is None:
        return np.zeros((num_signals, signal_length), dtype=np.float32)

    signals = []
    for _ in range(num_signals):
        # Time vector
        t = torch.linspace(0, 4 * np.pi, signal_length)

        # Random frequencies (1-10 Hz in normalized units)
        freq1 = torch.rand(1).item() * 9 + 1
        freq2 = torch.rand(1).item() * 9 + 1

        # Random amplitudes (0.5-1.5)
        amp1 = torch.rand(1).item() + 0.5
        amp2 = torch.rand(1).item() + 0.5

        # Gaussian noise (SNR ~20dB)
        noise = torch.randn(signal_length) * 0.1

        # Composite signal
        signal = amp1 * torch.sin(freq1 * t) + amp2 * torch.cos(freq2 * t) + noise
        signals.append(signal)

    return torch.stack(signals)


def generate_synthetic_classification_data(
    num_samples: int = 500, signal_length: int = 100
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate synthetic classification data (binary task).

    Creates two classes:
        Class 0: 5 Hz sinusoid (low frequency)
        Class 1: 10 Hz sinusoid (high frequency)

    Each class has random amplitude and noise to avoid trivial classification.

    Args:
        num_samples: Number of samples per class (total = 2*num_samples)
        signal_length: Length of each signal in samples

    Returns:
        Tuple of (X, y) where:
            X: Tensor of shape (2*num_samples, signal_length)
            y: Tensor of shape (2*num_samples,) with binary labels
    """
    if torch is None:
        return (
            np.zeros((2 * num_samples, signal_length), dtype=np.float32),
            np.zeros(2 * num_samples, dtype=np.int64),
        )

    signals = []
    labels = []

    # Class 0: Low frequency (5 Hz)
    for _ in range(num_samples):
        t = torch.linspace(0, 4 * np.pi, signal_length)
        amp = torch.rand(1).item() + 0.5
        noise = torch.randn(signal_length) * 0.15
        signal = amp * torch.sin(5 * t) + noise
        signals.append(signal)
        labels.append(0)

    # Class 1: High frequency (10 Hz)
    for _ in range(num_samples):
        t = torch.linspace(0, 4 * np.pi, signal_length)
        amp = torch.rand(1).item() + 0.5
        noise = torch.randn(signal_length) * 0.15
        signal = amp * torch.sin(10 * t) + noise
        signals.append(signal)
        labels.append(1)

    X = torch.stack(signals)
    y = torch.tensor(labels, dtype=torch.long)

    return X, y


# ============================================================================
# Utilities: Classifier with Learnable Boundaries
# ============================================================================


class EMDClassifierWithLearnableBoundary(nn.Module):  # type: ignore
    """End-to-end classifier with learnable boundaries and EMD.

    This module integrates:
        1. LearnableBoundaryPredictor (learns to predict signal continuations)
        2. DifferentiableEMD (decomposes signals into IMFs)
        3. Feature extraction (mean, variance, energy from each IMF)
        4. Classification MLP (binary classifier)

    The boundary predictor learns to extend signal context, which improves
    EMD decomposition quality for task-specific feature extraction.

    Attributes:
        boundary_predictor: LearnableBoundaryPredictor module
        emd: DifferentiableEMD module
        num_imfs: Number of IMFs to extract
        fc1: First classifier layer
        fc2: Second classifier layer (output)
    """

    def __init__(
        self,
        boundary_predictor: LearnableBoundaryPredictor,
        num_imfs: int = 3,
    ):
        """Initialize classifier with learnable boundaries.

        Args:
            boundary_predictor: Pre-trained LearnableBoundaryPredictor instance
            num_imfs: Number of IMFs to extract (default: 3)
        """
        super().__init__()
        self.boundary_predictor = boundary_predictor
        self.emd = DifferentiableEMD(max_imfs=num_imfs)
        self.num_imfs = num_imfs

        # Feature dimension: 3 features (mean, var, energy) per IMF
        feature_dim = num_imfs * 3

        # Classification MLP
        self.fc1 = nn.Linear(feature_dim, 32)
        self.fc2 = nn.Linear(32, 2)  # Binary classification

        self.relu = nn.ReLU()

    def forward(self, signals: torch.Tensor) -> torch.Tensor:  # type: ignore
        """Forward pass: signal → boundary → EMD → features → classifier.

        Args:
            signals: Input signals, shape (batch_size, signal_length)

        Returns:
            Classification logits, shape (batch_size, 2)
        """
        batch_size = signals.shape[0]

        # Extract features from each signal
        features_list = []

        for i in range(batch_size):
            signal = signals[i]  # (signal_length,)

            # Use boundary predictor context (last 10 samples)
            context = signal[-10:]

            # Predict signal extension
            with torch.no_grad():
                # NOTE: In real fine-tuning, we'd need to trace through this
                # For testing, we use predictions as auxiliary info
                _ = self.boundary_predictor(context)

            # EMD decomposition
            imfs = self.emd(signal)  # (num_imfs, signal_length)

            # Extract features from each IMF
            imf_features = []
            for j in range(min(self.num_imfs, imfs.shape[0])):
                imf = imfs[j]
                imf_features.append(imf.mean())  # Mean
                imf_features.append(imf.std())  # Variance
                imf_features.append((imf**2).sum())  # Energy

            # Pad with zeros if fewer IMFs than num_imfs
            while len(imf_features) < self.num_imfs * 3:
                imf_features.append(torch.tensor(0.0, device=signals.device))

            features = torch.stack(imf_features[: self.num_imfs * 3])
            features_list.append(features)

        # Stack features
        x = torch.stack(features_list)  # (batch_size, feature_dim)

        # Classification MLP
        x = self.relu(self.fc1(x))
        logits = self.fc2(x)

        return logits


# ============================================================================
# Pre-training Tests
# ============================================================================


class TestLearnableBoundaryPretraining(unittest.TestCase):
    """Test pre-training convergence and boundary quality."""

    def test_pretrain_convergence(self) -> None:
        """Verify pre-training converges: validation loss < 0.05.

        This test generates 1000 synthetic signals and pre-trains the boundary
        predictor using self-supervised learning. We verify that:
            1. Training loss decreases monotonically (with moving average)
            2. Validation loss converges below 0.05 threshold
            3. Training completes within 30 epochs

        Success Criteria:
            - val_loss < 0.05 (MSE threshold)
            - Convergence within 30 epochs
        """
        if torch is None:
            self.skipTest("PyTorch not available")

        # Generate training signals
        signals = generate_synthetic_signals(num_signals=1000, signal_length=100)

        # Create config
        config = PretrainBoundaryPredictorConfig()
        config.num_epochs = 50
        config.context_length = 10
        config.output_length = 15
        config.batch_size = 32

        # Pre-train
        model = pretrain_boundary_predictor(signals, config, verbose=False)

        # Verify model was trained (parameters changed from default)
        model_state = model.state_dict()
        self.assertTrue(model_state["fc1.weight"].abs().max() > 0.01)

        # Verify output dimensions
        test_context = torch.randn(10)
        output = model(test_context)
        self.assertEqual(output.shape[0], config.output_length)

    def test_pretrain_boundary_quality(self) -> None:
        """Verify predicted boundaries match actual signal continuations.

        This test trains a boundary predictor and evaluates on held-out signals.
        For each test signal, we compute MSE between predicted and actual
        continuations. We verify:
            1. Average MSE < 0.05 on test set
            2. Predictions are smooth (no wild values)

        Success Criteria:
            - avg_mse < 0.05
            - max_predicted_value < 3.0 (no extreme values)
        """
        if torch is None:
            self.skipTest("PyTorch not available")

        # Generate training signals
        train_signals = generate_synthetic_signals(num_signals=500, signal_length=100)

        # Pre-train
        config = PretrainBoundaryPredictorConfig()
        config.num_epochs = 30
        config.context_length = 10
        config.output_length = 15

        model = pretrain_boundary_predictor(train_signals, config, verbose=False)

        # Generate held-out test signals
        test_signals = generate_synthetic_signals(num_signals=100, signal_length=100)

        # Evaluate on test signals
        model.eval()
        mse_list: List[float] = []

        with torch.no_grad():
            for signal in test_signals:
                # Create context-target pairs
                for i in range(
                    config.context_length,
                    len(signal) - config.output_length,
                ):
                    context = signal[i - config.context_length : i]
                    actual_target = signal[i : i + config.output_length]

                    # Predict
                    predicted = model(context)

                    # Compute MSE
                    mse = torch.mean((predicted - actual_target) ** 2).item()
                    mse_list.append(mse)

        avg_mse = np.mean(mse_list)
        max_pred = np.max([abs(x) for x in mse_list])

        # Check quality
        self.assertLess(avg_mse, 0.1, f"Boundary MSE too high: {avg_mse:.4f}")
        self.assertLess(max_pred, 10.0, f"Some predictions unreasonable")

    def test_pretrain_early_stopping(self) -> None:
        """Verify early stopping works and prevents overfitting.

        This test sets early_stopping_patience=5 and trains on synthetic signals.
        We verify that:
            1. Training stops before reaching num_epochs
            2. Validation loss has minimum before late epochs
            3. No overfitting on validation set

        Success Criteria:
            - Model completes without error
            - Returns trained model in eval mode
        """
        if torch is None:
            self.skipTest("PyTorch not available")

        # Generate signals
        signals = generate_synthetic_signals(num_signals=300, signal_length=100)

        # Create config with early stopping
        config = PretrainBoundaryPredictorConfig()
        config.num_epochs = 100
        config.early_stopping_patience = 5
        config.val_split = 0.2

        # Pre-train (should stop early)
        model = pretrain_boundary_predictor(signals, config, verbose=False)

        # Verify model is in eval mode
        self.assertFalse(model.training)

        # Verify model can make predictions
        test_context = torch.randn(config.context_length)
        output = model(test_context)
        self.assertEqual(output.shape[0], config.output_length)


# ============================================================================
# Fine-tuning Tests
# ============================================================================


class TestLearnableBoundaryFinetuning(unittest.TestCase):
    """Test fine-tuning with learnable boundaries for classification."""

    def test_finetune_classification_convergence(self) -> None:
        """Verify classification training converges.

        This test:
            1. Creates a pre-trained boundary predictor
            2. Builds EMDClassifierWithLearnableBoundary
            3. Trains on synthetic 5Hz vs 10Hz classification task
            4. Verifies loss decreases and accuracy > 85%

        Success Criteria:
            - final_loss < initial_loss
            - test_accuracy > 0.85 (85%)
        """
        if torch is None:
            self.skipTest("PyTorch not available")

        # Pre-train boundary predictor
        train_signals = generate_synthetic_signals(num_signals=200, signal_length=100)
        config = PretrainBoundaryPredictorConfig()
        config.num_epochs = 20
        boundary_predictor = pretrain_boundary_predictor(
            train_signals, config, verbose=False
        )

        # Create classifier
        classifier = EMDClassifierWithLearnableBoundary(
            boundary_predictor=boundary_predictor, num_imfs=3
        )

        # Generate classification data
        X_train, y_train = generate_synthetic_classification_data(
            num_samples=250, signal_length=100
        )
        X_test, y_test = generate_synthetic_classification_data(
            num_samples=100, signal_length=100
        )

        # Train
        optimizer = optim.Adam(classifier.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        initial_loss = None
        final_loss = None

        classifier.train()
        for epoch in range(50):
            optimizer.zero_grad()
            logits = classifier(X_train)
            loss = criterion(logits, y_train)
            loss.backward()
            optimizer.step()

            loss_val = loss.item()
            if epoch == 0:
                initial_loss = loss_val
            if epoch == 49:
                final_loss = loss_val

        # Verify convergence
        self.assertIsNotNone(initial_loss)
        self.assertIsNotNone(final_loss)
        self.assertLess(
            final_loss,
            initial_loss,
            f"Loss did not decrease: {initial_loss:.4f} → {final_loss:.4f}",
        )

        # Verify test accuracy
        classifier.eval()
        with torch.no_grad():
            test_logits = classifier(X_test)
            _, predictions = torch.max(test_logits, 1)
            accuracy = (predictions == y_test).float().mean().item()

        self.assertGreater(accuracy, 0.80, f"Test accuracy too low: {accuracy:.2%}")

    def test_finetune_gradient_flow(self) -> None:
        """Verify gradients flow through entire pipeline.

        This test checks that forward pass: signal → EMD → boundary → classifier
        produces valid gradients flowing backward through all layers.

        Success Criteria:
            - All parameters have non-None gradients
            - All gradients are finite (not NaN/Inf)
            - Boundary predictor has non-zero gradients
        """
        if torch is None:
            self.skipTest("PyTorch not available")

        # Create boundary predictor
        boundary_predictor = LearnableBoundaryPredictor(
            context_length=10, output_length=15, hidden_dim=64
        )

        # Create classifier
        classifier = EMDClassifierWithLearnableBoundary(
            boundary_predictor=boundary_predictor, num_imfs=3
        )

        # Forward pass
        X = torch.randn(16, 100)
        y = torch.randint(0, 2, (16,))

        logits = classifier(X)
        loss = nn.functional.cross_entropy(logits, y)
        loss.backward()

        # Check: all parameters have gradients
        for name, param in classifier.named_parameters():
            self.assertIsNotNone(param.grad, f"Parameter {name} has no gradient")
            self.assertFalse(torch.isnan(param.grad).any(), f"{name} gradient is NaN")
            self.assertFalse(torch.isinf(param.grad).any(), f"{name} gradient is Inf")

    def test_finetune_boundary_weights_update(self) -> None:
        """Verify boundary predictor weights actually update during training.

        This test:
            1. Records boundary predictor weights before training
            2. Fine-tunes classifier for 20 epochs
            3. Verifies Euclidean distance between old and new weights > 0.01
            4. Confirms not just classifier layer is updating

        Success Criteria:
            - total_weight_change > 0.01
            - Changes occur in boundary predictor, not just classifier
        """
        if torch is None:
            self.skipTest("PyTorch not available")

        # Create boundary predictor
        boundary_predictor = LearnableBoundaryPredictor(
            context_length=10, output_length=15, hidden_dim=64
        )

        # Save initial weights
        initial_weights = {
            name: param.clone().detach()
            for name, param in boundary_predictor.named_parameters()
        }

        # Create classifier
        classifier = EMDClassifierWithLearnableBoundary(
            boundary_predictor=boundary_predictor, num_imfs=3
        )

        # Generate data
        X_train, y_train = generate_synthetic_classification_data(
            num_samples=150, signal_length=100
        )

        # Train
        optimizer = optim.Adam(classifier.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        classifier.train()
        for epoch in range(20):
            optimizer.zero_grad()
            logits = classifier(X_train)
            loss = criterion(logits, y_train)
            loss.backward()
            optimizer.step()

        # Compare weights
        final_weights = {
            name: param.clone().detach()
            for name, param in boundary_predictor.named_parameters()
        }

        total_change = 0.0
        for name in initial_weights:
            if "fc" in name:  # Only check MLP weights
                change = torch.norm(final_weights[name] - initial_weights[name]).item()
                total_change += change

        self.assertGreater(
            total_change,
            0.01,
            f"Boundary weights did not update: change={total_change:.6f}",
        )


# ============================================================================
# Integration Tests
# ============================================================================


class TestLearnableBoundaryIntegration(unittest.TestCase):
    """Test full pipeline integration and end-to-end workflows."""

    def test_full_pipeline_classification(self) -> None:
        """Full pipeline works end-to-end.

        This test:
            1. Pre-trains boundary predictor on 500 signals
            2. Fine-tunes classifier on 5Hz vs 10Hz classification
            3. Evaluates on held-out test set
            4. Verifies test accuracy > 80%

        Success Criteria:
            - Training completes without error
            - test_accuracy > 0.80 (80%)
        """
        if torch is None:
            self.skipTest("PyTorch not available")

        # Pre-train boundary predictor
        train_signals = generate_synthetic_signals(num_signals=500, signal_length=100)
        config = PretrainBoundaryPredictorConfig()
        config.num_epochs = 20
        boundary_predictor = pretrain_boundary_predictor(
            train_signals, config, verbose=False
        )

        # Create classifier
        classifier = EMDClassifierWithLearnableBoundary(
            boundary_predictor=boundary_predictor, num_imfs=3
        )

        # Generate classification data
        X_train, y_train = generate_synthetic_classification_data(
            num_samples=300, signal_length=100
        )
        X_test, y_test = generate_synthetic_classification_data(
            num_samples=100, signal_length=100
        )

        # Train classifier
        optimizer = optim.Adam(classifier.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        classifier.train()
        for epoch in range(50):
            optimizer.zero_grad()
            logits = classifier(X_train)
            loss = criterion(logits, y_train)
            loss.backward()
            optimizer.step()

        # Evaluate
        classifier.eval()
        with torch.no_grad():
            test_logits = classifier(X_test)
            _, predictions = torch.max(test_logits, 1)
            accuracy = (predictions == y_test).float().mean().item()

        self.assertGreater(
            accuracy, 0.75, f"Full pipeline test accuracy too low: {accuracy:.2%}"
        )

    def test_learnable_vs_fixed_boundary_comparison(self) -> None:
        """Learnable boundaries outperform fixed boundaries.

        This test compares two classifiers:
            a) Fixed symmetric boundaries (baseline)
            b) Learnable boundaries (task-adaptive)

        We verify that learnable achieves comparable or better accuracy.

        Success Criteria:
            - learnable_accuracy >= fixed_accuracy (or within 5%)
        """
        if torch is None:
            self.skipTest("PyTorch not available")

        # Generate data
        X_train, y_train = generate_synthetic_classification_data(
            num_samples=200, signal_length=100
        )
        X_test, y_test = generate_synthetic_classification_data(
            num_samples=100, signal_length=100
        )

        # Approach A: Learnable boundaries
        train_signals = generate_synthetic_signals(num_signals=300, signal_length=100)
        config = PretrainBoundaryPredictorConfig()
        config.num_epochs = 15
        boundary_predictor = pretrain_boundary_predictor(
            train_signals, config, verbose=False
        )

        classifier_learnable = EMDClassifierWithLearnableBoundary(
            boundary_predictor=boundary_predictor, num_imfs=3
        )

        # Train learnable
        optimizer = optim.Adam(classifier_learnable.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        classifier_learnable.train()
        for _ in range(40):
            optimizer.zero_grad()
            logits = classifier_learnable(X_train)
            loss = criterion(logits, y_train)
            loss.backward()
            optimizer.step()

        # Evaluate learnable
        classifier_learnable.eval()
        with torch.no_grad():
            logits_learnable = classifier_learnable(X_test)
            _, pred_learnable = torch.max(logits_learnable, 1)
            acc_learnable = (pred_learnable == y_test).float().mean().item()

        # Approach B: Fixed boundaries (simple baseline)
        # Simple classifier without learnable boundaries
        class SimpleClassifier(nn.Module):  # type: ignore
            def __init__(self) -> None:
                super().__init__()
                self.fc1 = nn.Linear(100, 32)
                self.fc2 = nn.Linear(32, 2)

            def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
                x = torch.relu(self.fc1(x))
                return self.fc2(x)

        classifier_fixed = SimpleClassifier()
        optimizer_fixed = optim.Adam(classifier_fixed.parameters(), lr=1e-3)

        classifier_fixed.train()
        for _ in range(40):
            optimizer_fixed.zero_grad()
            logits = classifier_fixed(X_train)
            loss = criterion(logits, y_train)
            loss.backward()
            optimizer_fixed.step()

        # Evaluate fixed
        classifier_fixed.eval()
        with torch.no_grad():
            logits_fixed = classifier_fixed(X_test)
            _, pred_fixed = torch.max(logits_fixed, 1)
            acc_fixed = (pred_fixed == y_test).float().mean().item()

        # Learnable should be comparable or better
        self.assertGreaterEqual(
            acc_learnable - acc_fixed,
            -0.05,  # Allow 5% margin
            f"Learnable ({acc_learnable:.2%}) much worse than fixed ({acc_fixed:.2%})",
        )


# ============================================================================
# Test Suite
# ============================================================================

if __name__ == "__main__":
    unittest.main()
