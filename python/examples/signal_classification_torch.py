#!/usr/bin/env python3
"""Signal Classification using Differentiable EMD with PyTorch.

This example demonstrates how to use differentiable EMD as a feature extractor
in a PyTorch neural network for signal classification. The model decomposes input
signals into Intrinsic Mode Functions (IMFs), extracts features from them,
and feeds the features to a classifier network.

The example:
1. Generates synthetic binary classification data (two sinusoid frequencies)
2. Defines an EMD-based classifier module
3. Trains the classifier on the synthetic data
4. Evaluates on held-out test set
5. Visualizes training curves and learned features

Expected results:
- Test accuracy: >90%
- Training time: <2 minutes
"""

import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:
    print("Error: PyTorch is required. Install with: pip install torch")
    sys.exit(1)

try:
    from ferromode_ml import DifferentiableEMD
except ImportError:
    print(
        "Error: ferromode_ml not found. Make sure ferromode is installed"
        " and the Rust backend is compiled."
    )
    sys.exit(1)


def generate_synthetic_data(
    num_samples: int = 1000,
    signal_length: int = 256,
    noise_std: float = 0.2,
    random_seed: int = 42,
) -> tuple:
    """Generate synthetic binary classification data.

    Generates two classes of signals:
    - Class 0: 5 Hz sinusoid + noise
    - Class 1: 10 Hz sinusoid + noise

    Args:
        num_samples: Total number of samples (split 50/50 between classes)
        signal_length: Length of each signal in samples
        noise_std: Standard deviation of Gaussian noise
        random_seed: Random seed for reproducibility

    Returns:
        Tuple of (X_train, y_train, X_test, y_test) as numpy arrays
        X_train, X_test: (N, signal_length) float32
        y_train, y_test: (N,) int64
    """
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    # Time vector normalized to [0, 1)
    t = np.linspace(0, 1, signal_length, endpoint=False)

    # Generate Class 0: 5 Hz sinusoid
    freq_0 = 5.0
    phase_0 = 2 * np.pi * freq_0 * t
    signal_0 = np.sin(phase_0)

    # Generate Class 1: 10 Hz sinusoid
    freq_1 = 10.0
    phase_1 = 2 * np.pi * freq_1 * t
    signal_1 = np.sin(phase_1)

    # Generate samples
    samples_per_class = num_samples // 2
    X_class_0 = np.array(
        [
            signal_0 + noise_std * np.random.randn(signal_length)
            for _ in range(samples_per_class)
        ],
        dtype=np.float32,
    )
    X_class_1 = np.array(
        [
            signal_1 + noise_std * np.random.randn(signal_length)
            for _ in range(samples_per_class)
        ],
        dtype=np.float32,
    )

    # Stack and create labels
    X = np.vstack([X_class_0, X_class_1])
    y = np.hstack([np.zeros(samples_per_class), np.ones(samples_per_class)])

    # Shuffle
    perm = np.random.permutation(X.shape[0])
    X = X[perm]
    y = y[perm]

    # Train/test split (80/20)
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    return X_train, y_train, X_test, y_test


class EMDFeatureExtractor(nn.Module):
    """Feature extractor using differentiable EMD.

    Decomposes signals into IMFs and extracts statistical features
    (mean, variance, energy) from each IMF.

    Args:
        num_imfs: Maximum number of IMFs to extract
    """

    def __init__(self, num_imfs: int = 3):
        """Initialize feature extractor."""
        super().__init__()
        self.emd = DifferentiableEMD(max_imfs=num_imfs)
        self.num_imfs = num_imfs

    def forward(self, signals: torch.Tensor) -> torch.Tensor:
        """Extract IMF-based features from signals.

        Args:
            signals: Input signals of shape (batch_size, signal_length)

        Returns:
            Features of shape (batch_size, num_imfs * 3)
            where each IMF contributes (mean, variance, energy)
        """
        # Decompose signals into IMFs
        # imfs: (batch_size, num_imfs, signal_length)
        imfs = self.emd(signals)

        batch_size, num_imfs, signal_length = imfs.shape
        features = []

        # Extract features from each IMF
        for i in range(num_imfs):
            imf = imfs[:, i, :]  # (batch_size, signal_length)

            # Mean of IMF
            mean_feat = torch.mean(imf, dim=1, keepdim=True)
            # Variance of IMF
            var_feat = torch.var(imf, dim=1, keepdim=True)
            # Energy of IMF (sum of squares)
            energy_feat = torch.sum(imf**2, dim=1, keepdim=True)

            features.extend([mean_feat, var_feat, energy_feat])

        # Concatenate all features
        all_features = torch.cat(features, dim=1)
        return all_features


class EMDClassifier(nn.Module):
    """Binary classifier using EMD-extracted features.

    Architecture:
    1. EMD feature extraction
    2. Two fully-connected hidden layers with ReLU activation
    3. Output layer for binary classification

    Args:
        num_imfs: Number of IMFs for decomposition
    """

    def __init__(self, num_imfs: int = 3):
        """Initialize classifier."""
        super().__init__()
        self.feature_extractor = EMDFeatureExtractor(num_imfs)

        # Feature dimension: num_imfs * 3 (mean, var, energy per IMF)
        feature_dim = num_imfs * 3

        # Classifier network
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 2),  # Binary classification (2 classes)
        )

    def forward(self, signals: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            signals: Input signals of shape (batch_size, signal_length)

        Returns:
            Logits of shape (batch_size, 2)
        """
        features = self.feature_extractor(signals)
        logits = self.classifier(features)
        return logits


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> float:
    """Train for one epoch.

    Args:
        model: The model to train
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on (CPU or CUDA)

    Returns:
        Average loss for the epoch
    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch_X, batch_y in train_loader:
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)

        # Forward pass
        logits = model(batch_X)
        loss = criterion(logits, batch_y)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple:
    """Evaluate model on a dataset.

    Args:
        model: The model to evaluate
        data_loader: Data loader
        criterion: Loss function
        device: Device (CPU or CUDA)

    Returns:
        Tuple of (average_loss, accuracy)
    """
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for batch_X, batch_y in data_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            logits = model(batch_X)
            loss = criterion(logits, batch_y)
            total_loss += loss.item()

            # Compute accuracy
            predictions = torch.argmax(logits, dim=1)
            total_correct += (predictions == batch_y).sum().item()
            total_samples += batch_y.size(0)

    avg_loss = total_loss / len(data_loader)
    accuracy = total_correct / total_samples
    return avg_loss, accuracy


def main():
    """Main training script."""
    print("=" * 70)
    print("Signal Classification with Differentiable EMD (PyTorch)")
    print("=" * 70)

    # Configuration
    num_epochs = 50
    batch_size = 32
    learning_rate = 0.001
    num_imfs = 3
    signal_length = 256
    random_seed = 42

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # Set random seeds for reproducibility
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    # Generate data
    print("\nGenerating synthetic data...")
    X_train, y_train, X_test, y_test = generate_synthetic_data(
        num_samples=1000,
        signal_length=signal_length,
        noise_std=0.2,
        random_seed=random_seed,
    )
    print(f"  Training samples: {X_train.shape[0]}")
    print(f"  Test samples: {X_test.shape[0]}")
    print(f"  Signal length: {signal_length}")

    # Create data loaders
    train_dataset = TensorDataset(
        torch.from_numpy(X_train),
        torch.from_numpy(y_train).long(),
    )
    test_dataset = TensorDataset(
        torch.from_numpy(X_test),
        torch.from_numpy(y_test).long(),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    # Create model
    print(f"\nInitializing model (num_imfs={num_imfs})...")
    model = EMDClassifier(num_imfs=num_imfs).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Print model summary
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model parameters: {total_params:,}")

    # Training loop
    print("\nTraining...")
    print(f"{'Epoch':<8} {'Train Loss':<12} {'Val Loss':<12} {'Val Acc':<10}")
    print("-" * 45)

    train_losses = []
    val_losses = []
    val_accs = []
    best_val_acc = 0.0
    patience = 10
    patience_counter = 0

    for epoch in range(num_epochs):
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)

        # Validate
        val_loss, val_acc = evaluate(model, test_loader, criterion, device)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        # Print progress
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(
                f"{epoch + 1:<8} {train_loss:<12.6f} {val_loss:<12.6f} {val_acc:<10.4f}"
            )

        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch + 1}")
                break

    # Final evaluation
    print("\n" + "=" * 70)
    final_val_loss, final_val_acc = evaluate(model, test_loader, criterion, device)
    print(f"Final Test Accuracy: {final_val_acc:.4f}")
    print(f"Final Test Loss: {final_val_loss:.6f}")
    print("=" * 70)

    # Optional: Try to import matplotlib for visualization
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # Plot training curves
        axes[0].plot(train_losses, label="Train Loss")
        axes[0].plot(val_losses, label="Val Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].set_title("Training Curves")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Plot validation accuracy
        axes[1].plot(val_accs)
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy")
        axes[1].set_title("Validation Accuracy")
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim([0, 1])

        plt.tight_layout()
        output_path = Path(__file__).parent / "signal_classification_torch_curves.png"
        plt.savefig(output_path, dpi=100)
        print(f"\nTraining curves saved to: {output_path}")
        plt.close()

    except ImportError:
        print("\nMatplotlib not available; skipping visualization")

    print("\nExample completed successfully!")


if __name__ == "__main__":
    main()
