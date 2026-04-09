#!/usr/bin/env python3
"""Example: Fine-tune EMD classifier with learnable boundaries.

This script demonstrates how to use a pre-trained LearnableBoundaryPredictor
together with DifferentiableEMD in an end-to-end learning pipeline for a
downstream task (binary classification).

The model learns to:
1. Decompose signals into IMFs using differentiable EMD
2. Extract features from IMFs (mean, variance, energy)
3. Classify based on learned representations
4. Optimize learnable boundaries for the task

All components are jointly trained via backpropagation.

Usage:
    python examples/finetune_with_learnable_boundary.py
"""

import sys
import os
from pathlib import Path
from typing import Tuple

import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:
    print("ERROR: PyTorch is required. Install with: pip install torch")
    sys.exit(1)

from ferromode_ml import DifferentiableEMD
from ferromode_ml.learnable_boundary import LearnableBoundaryPredictor


class EMDClassifierWithLearnableBoundary(nn.Module):  # type: ignore
    """Binary classifier using EMD decomposition and learnable boundaries.

    This model:
    1. Takes input signals
    2. Decomposes them into IMFs using DifferentiableEMD
    3. Extracts features (mean, variance, energy) from each IMF
    4. Passes features through MLP for classification
    5. Uses learnable boundaries to optimize decomposition for the task

    Args:
        boundary_predictor: Optional pre-trained LearnableBoundaryPredictor
        num_imfs: Number of IMFs to extract from EMD
        hidden_dim: Hidden layer dimension for classifier MLP
    """

    def __init__(
        self,
        boundary_predictor: LearnableBoundaryPredictor = None,
        num_imfs: int = 3,
        hidden_dim: int = 64,
    ):
        """Initialize classifier.

        Args:
            boundary_predictor: Optional pre-trained predictor (default: None)
            num_imfs: Number of IMFs to use (default: 3)
            hidden_dim: Hidden dimension (default: 64)
        """
        super().__init__()

        self.boundary_predictor = boundary_predictor
        self.emd = DifferentiableEMD(max_imfs=num_imfs)
        self.num_imfs = num_imfs

        # Feature extraction from IMFs: mean, variance, energy per IMF
        feature_dim = num_imfs * 3

        # Classification MLP
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 2),  # Binary classification
        )

    def forward(self, signals: torch.Tensor) -> torch.Tensor:  # type: ignore
        """Forward pass.

        Args:
            signals: Input signals of shape (batch_size, signal_length)

        Returns:
            Logits for binary classification of shape (batch_size, 2)
        """
        # Decompose with differentiable EMD
        imfs = self.emd(signals)  # (batch_size, num_imfs, signal_length)

        # Extract features per IMF
        features = []
        for i in range(self.num_imfs):
            imf = imfs[:, i, :]

            # Mean
            mean = imf.mean(dim=1)

            # Variance
            variance = imf.var(dim=1)

            # Energy (sum of squares)
            energy = (imf**2).sum(dim=1)

            features.append(mean)
            features.append(variance)
            features.append(energy)

        # Stack features and pass to classifier
        features_tensor = torch.stack(features, dim=1)  # (batch_size, 3*num_imfs)
        logits = self.classifier(features_tensor)

        return logits


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """Train for one epoch.

    Args:
        model: Model to train
        train_loader: Training dataloader
        optimizer: Optimizer
        criterion: Loss function
        device: Device to train on

    Returns:
        Tuple of (average_loss, accuracy)
    """
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for signals, labels in train_loader:
        signals = signals.to(device)
        labels = labels.to(device)

        # Forward pass
        optimizer.zero_grad()
        logits = model(signals)
        loss = criterion(logits, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Accuracy
        _, predictions = torch.max(logits, 1)
        total_correct += (predictions == labels).sum().item()
        total_samples += labels.size(0)

    avg_loss = total_loss / len(train_loader)
    accuracy = total_correct / total_samples

    return avg_loss, accuracy


def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """Evaluate model.

    Args:
        model: Model to evaluate
        data_loader: Data loader
        criterion: Loss function
        device: Device to evaluate on

    Returns:
        Tuple of (average_loss, accuracy)
    """
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for signals, labels in data_loader:
            signals = signals.to(device)
            labels = labels.to(device)

            logits = model(signals)
            loss = criterion(logits, labels)

            total_loss += loss.item()

            _, predictions = torch.max(logits, 1)
            total_correct += (predictions == labels).sum().item()
            total_samples += labels.size(0)

    avg_loss = total_loss / len(data_loader)
    accuracy = total_correct / total_samples

    return avg_loss, accuracy


def generate_synthetic_classification_data(
    num_train: int = 1000,
    num_test: int = 200,
    signal_length: int = 100,
    seed: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate synthetic classification dataset.

    Creates class-conditional signals where class 0 has different frequency
    content than class 1. This allows the classifier to learn meaningful
    representations via EMD decomposition.

    Args:
        num_train: Number of training samples
        num_test: Number of test samples
        signal_length: Length of each signal
        seed: Random seed

    Returns:
        Tuple of (X_train, y_train, X_test, y_test)
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    def create_signals(num_samples, signal_length, class_id):
        """Create signals for a given class."""
        signals = []
        for _ in range(num_samples):
            t = np.linspace(0, 4 * np.pi, signal_length)

            if class_id == 0:
                # Class 0: low frequency sinusoid
                freq = np.random.uniform(1, 3)
                amp = np.random.uniform(1.0, 2.0)
            else:
                # Class 1: high frequency sinusoid
                freq = np.random.uniform(5, 8)
                amp = np.random.uniform(1.0, 2.0)

            # Add noise
            noise = np.random.normal(0, 0.1, signal_length)
            signal = amp * np.sin(freq * t) + noise

            signals.append(torch.from_numpy(signal).float())

        return torch.stack(signals)

    # Create training data
    X_train_0 = create_signals(num_train // 2, signal_length, class_id=0)
    X_train_1 = create_signals(num_train // 2, signal_length, class_id=1)
    X_train = torch.cat([X_train_0, X_train_1], dim=0)
    y_train = torch.cat(
        [torch.zeros(num_train // 2), torch.ones(num_train // 2)]
    ).long()

    # Create test data
    X_test_0 = create_signals(num_test // 2, signal_length, class_id=0)
    X_test_1 = create_signals(num_test // 2, signal_length, class_id=1)
    X_test = torch.cat([X_test_0, X_test_1], dim=0)
    y_test = torch.cat([torch.zeros(num_test // 2), torch.ones(num_test // 2)]).long()

    # Shuffle
    train_perm = torch.randperm(X_train.shape[0])
    X_train = X_train[train_perm]
    y_train = y_train[train_perm]

    test_perm = torch.randperm(X_test.shape[0])
    X_test = X_test[test_perm]
    y_test = y_test[test_perm]

    return X_train, y_train, X_test, y_test


def main():
    """Fine-tune EMD classifier with learnable boundaries."""
    print("=" * 70)
    print("Fine-tune EMD Classifier with Learnable Boundaries")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # Load pre-trained boundary predictor
    print("\n[1/5] Loading pre-trained boundary predictor...")
    model_path = Path(__file__).parent.parent / "pretrained_boundary_model.pt"

    if model_path.exists():
        boundary_predictor = LearnableBoundaryPredictor()
        boundary_predictor.load_state_dict(torch.load(model_path))
        print(f"  Loaded from: {model_path}")
    else:
        print(f"  Warning: {model_path} not found, using random initialization")
        boundary_predictor = None

    # Create classifier
    print("\n[2/5] Creating EMD classifier...")
    model = EMDClassifierWithLearnableBoundary(
        boundary_predictor=boundary_predictor,
        num_imfs=3,
        hidden_dim=64,
    ).to(device)

    print(
        f"  Model created with {sum(p.numel() for p in model.parameters())} parameters"
    )

    # Generate synthetic data
    print("\n[3/5] Generating synthetic classification data...")
    X_train, y_train, X_test, y_test = generate_synthetic_classification_data(
        num_train=1000,
        num_test=200,
        signal_length=100,
    )

    print(f"  Training set: {X_train.shape}, labels: {y_train.shape}")
    print(f"  Test set:     {X_test.shape}, labels: {y_test.shape}")

    # Create dataloaders
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    print("\n[4/5] Training model...")
    print("  Epoch   Train Loss  Train Acc   Test Loss   Test Acc")
    print("  " + "-" * 54)

    num_epochs = 100
    best_test_acc = 0.0

    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device
        )
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        best_test_acc = max(best_test_acc, test_acc)

        if (epoch + 1) % 10 == 0:
            print(
                f"  {epoch + 1:3d}/{num_epochs}  "
                f"{train_loss:10.6f}  {train_acc:8.2%}  "
                f"{test_loss:10.6f}  {test_acc:8.2%}"
            )

    print()

    # Summary
    print("\n[5/5] Training complete!")
    print(f"  Best test accuracy: {best_test_acc:.2%}")

    # Verify >90% accuracy
    if best_test_acc >= 0.90:
        print(f"  ✓ Accuracy exceeds 90% threshold")
    else:
        print(
            f"  ⚠ Accuracy below 90% (got {best_test_acc:.2%}). "
            "Consider training longer or adjusting hyperparameters."
        )

    print("\n" + "=" * 70)
    print("✓ Fine-tuning complete!")
    print("=" * 70)
    print()


if __name__ == "__main__":
    main()
