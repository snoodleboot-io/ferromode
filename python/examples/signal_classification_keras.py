#!/usr/bin/env python3
"""Signal Classification using Differentiable EMD with TensorFlow/Keras.

This example demonstrates how to use differentiable EMD as a feature extractor
in a TensorFlow/Keras neural network for signal classification. The model decomposes
input signals into Intrinsic Mode Functions (IMFs), extracts features from them,
and feeds the features to a classifier network.

The example:
1. Generates synthetic binary classification data (two sinusoid frequencies)
2. Defines an EMD-based Keras model
3. Trains the model on the synthetic data using tf.keras.fit()
4. Evaluates on held-out test set
5. Visualizes training curves and performance

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
    import tensorflow as tf
except ImportError:
    print("Error: TensorFlow is required. Install with: pip install tensorflow")
    sys.exit(1)

try:
    from ferromode_ml import DifferentiableEMDLayer
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
        y_train, y_test: (N, 2) float32 (one-hot encoded)
    """
    np.random.seed(random_seed)
    tf.random.set_seed(random_seed)

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

    # One-hot encode labels
    y_train_onehot = tf.keras.utils.to_categorical(y_train, num_classes=2)
    y_test_onehot = tf.keras.utils.to_categorical(y_test, num_classes=2)

    return X_train, y_train_onehot, X_test, y_test_onehot


def create_emd_classifier(
    signal_length: int,
    num_imfs: int = 3,
) -> tf.keras.Model:
    """Create an EMD-based classifier model.

    Architecture:
    1. EMD layer for signal decomposition
    2. Flatten layer to vectorize IMFs
    3. Two dense hidden layers with ReLU and dropout
    4. Output layer for binary classification

    Args:
        signal_length: Length of input signals
        num_imfs: Maximum number of IMFs to extract

    Returns:
        Compiled Keras model
    """
    inputs = tf.keras.Input(shape=(signal_length,), name="signal_input")

    # EMD decomposition
    emd_output = DifferentiableEMDLayer(
        max_imfs=num_imfs,
        name="emd_layer",
    )(inputs)

    # Flatten IMF matrix to feature vector
    # emd_output shape: (batch, num_imfs, signal_length)
    # After flatten: (batch, num_imfs * signal_length)
    flattened = tf.keras.layers.Flatten()(emd_output)

    # Feature extraction via dense layers
    hidden1 = tf.keras.layers.Dense(
        128,
        activation="relu",
        name="hidden1",
    )(flattened)
    hidden1 = tf.keras.layers.Dropout(0.3)(hidden1)

    hidden2 = tf.keras.layers.Dense(
        64,
        activation="relu",
        name="hidden2",
    )(hidden1)
    hidden2 = tf.keras.layers.Dropout(0.3)(hidden2)

    # Output layer
    outputs = tf.keras.layers.Dense(
        2,
        activation="softmax",
        name="classification_output",
    )(hidden2)

    # Create model
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    return model


def main():
    """Main training script."""
    print("=" * 70)
    print("Signal Classification with Differentiable EMD (TensorFlow/Keras)")
    print("=" * 70)

    # Configuration
    num_epochs = 50
    batch_size = 32
    num_imfs = 3
    signal_length = 256
    random_seed = 42
    validation_split = 0.2

    # Random seed for reproducibility
    np.random.seed(random_seed)
    tf.random.set_seed(random_seed)

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

    # Create model
    print(f"\nCreating model (num_imfs={num_imfs})...")
    model = create_emd_classifier(
        signal_length=signal_length,
        num_imfs=num_imfs,
    )

    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    # Print model summary
    model.summary()

    # Training
    print("\nTraining...")
    history = model.fit(
        X_train,
        y_train,
        epochs=num_epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        verbose=1,
    )

    # Evaluation
    print("\n" + "=" * 70)
    test_loss, test_accuracy = model.evaluate(
        X_test,
        y_test,
        verbose=0,
    )
    print(f"Test Loss: {test_loss:.6f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print("=" * 70)

    # Visualization
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # Plot training curves
        axes[0].plot(history.history["loss"], label="Train Loss")
        axes[0].plot(history.history["val_loss"], label="Val Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].set_title("Training Curves")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Plot accuracy
        axes[1].plot(history.history["accuracy"], label="Train Accuracy")
        axes[1].plot(history.history["val_accuracy"], label="Val Accuracy")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy")
        axes[1].set_title("Model Accuracy")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim([0, 1])

        plt.tight_layout()
        output_path = Path(__file__).parent / "signal_classification_keras_curves.png"
        plt.savefig(output_path, dpi=100)
        print(f"\nTraining curves saved to: {output_path}")
        plt.close()

    except ImportError:
        print("\nMatplotlib not available; skipping visualization")

    print("\nExample completed successfully!")


if __name__ == "__main__":
    main()
