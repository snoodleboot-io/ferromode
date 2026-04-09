"""Learnable boundary predictor module for PyTorch.

This module provides a neural network-based boundary predictor that learns to
extend signal context via supervised or self-supervised learning. The learned
boundaries can be used with differentiable EMD to adapt decomposition to task-
specific requirements.

Classes:
    LearnableBoundaryPredictor: Neural network for boundary prediction
    PretrainBoundaryPredictorConfig: Configuration for pre-training

Functions:
    pretrain_boundary_predictor: Pre-train on unlabeled signals
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset, random_split
except ImportError:
    torch = None  # type: ignore
    nn = None  # type: ignore
    optim = None  # type: ignore
    DataLoader = None  # type: ignore
    TensorDataset = None  # type: ignore
    random_split = None  # type: ignore


class LearnableBoundaryPredictor(nn.Module):  # type: ignore
    """Neural network that learns task-specific boundary predictions.

    This module takes signal context (last N samples) and predicts boundary
    extension (M samples ahead). Parameters are trained via backpropagation
    to minimize reconstruction error on future signal samples.

    The network includes:
    - Optional learnable input/output normalization (scale factors)
    - Two-layer MLP with ReLU activation
    - Support for both single samples and batched inputs

    Attributes:
        context_length: Number of samples in input context
        output_length: Number of samples to predict ahead
        hidden_dim: Dimension of hidden layer
        normalize: Whether to use learnable normalization
        input_scale: Learnable input normalization (if normalize=True)
        output_scale: Learnable output normalization (if normalize=True)
        fc1: First linear layer (context_length -> hidden_dim)
        fc2: Second linear layer (hidden_dim -> hidden_dim)
        fc3: Output layer (hidden_dim -> output_length)
        activation: ReLU activation function
    """

    def __init__(
        self,
        context_length: int = 10,
        output_length: int = 15,
        hidden_dim: int = 128,
        normalize: bool = True,
    ):
        """Initialize LearnableBoundaryPredictor.

        Args:
            context_length: Number of samples in input context (default: 10)
            output_length: Number of samples to predict ahead (default: 15)
            hidden_dim: Hidden layer dimension (default: 128)
            normalize: Whether to use learnable normalization (default: True)
        """
        super().__init__()

        self.context_length = context_length
        self.output_length = output_length
        self.hidden_dim = hidden_dim
        self.normalize = normalize

        # Learnable normalization (scale factors)
        if normalize:
            self.input_scale = nn.Parameter(torch.ones(context_length))
            self.output_scale = nn.Parameter(torch.ones(output_length))
        else:
            self.register_parameter("input_scale", None)
            self.register_parameter("output_scale", None)

        # Core MLP
        self.fc1 = nn.Linear(context_length, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_length)

        # Activation
        self.activation = nn.ReLU()

        # Initialize weights
        nn.init.kaiming_normal_(self.fc1.weight)
        nn.init.kaiming_normal_(self.fc2.weight)
        nn.init.normal_(self.fc3.weight, std=0.01)

    def forward(self, signal_context: torch.Tensor) -> torch.Tensor:  # type: ignore
        """Forward pass: predict boundary extension from context.

        Args:
            signal_context: Input signal context
                Shape: (batch_size, context_length) or (context_length,)
                Values: Floating-point samples

        Returns:
            Predicted boundary extension
            Shape: (batch_size, output_length) or (output_length,)
        """
        # Ensure batch dimension
        if signal_context.dim() == 1:
            signal_context = signal_context.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        # Normalize input
        if self.normalize:
            x = signal_context * self.input_scale
        else:
            x = signal_context

        # MLP forward pass
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.fc3(x)

        # Denormalize output
        if self.normalize:
            x = x * self.output_scale

        # Remove batch dimension if input was single
        if squeeze_output:
            x = x.squeeze(0)

        return x


class PretrainBoundaryPredictorConfig:
    """Configuration for pre-training boundary predictor.

    This class centralizes all hyperparameters for the unsupervised pre-training
    phase where the predictor learns to reconstruct signal continuations.

    Attributes:
        context_length: Number of samples in input context
        output_length: Number of samples to predict ahead
        hidden_dim: Hidden layer dimension
        batch_size: Training batch size
        learning_rate: Adam optimizer learning rate
        num_epochs: Maximum number of training epochs
        val_split: Fraction of data to use for validation (0.0 to 1.0)
        early_stopping_patience: Epochs to wait before early stopping
        device: Device to train on ('cuda' or 'cpu')
    """

    def __init__(self):
        """Initialize default configuration."""
        self.context_length: int = 10
        self.output_length: int = 15
        self.hidden_dim: int = 128
        self.batch_size: int = 32
        self.learning_rate: float = 1e-3
        self.num_epochs: int = 50
        self.val_split: float = 0.2
        self.early_stopping_patience: int = 10
        self.device: str = "cuda" if torch.cuda.is_available() else "cpu"


def pretrain_boundary_predictor(
    signals: torch.Tensor,
    config: PretrainBoundaryPredictorConfig,
    verbose: bool = True,
) -> LearnableBoundaryPredictor:
    """Pre-train boundary predictor on unlabeled signals.

    This function trains a LearnableBoundaryPredictor to reconstruct signal
    continuations via self-supervised learning. For each signal in the training
    set, we create context-target pairs by sliding a window over the signal.

    The loss function is MSE: L = ||predicted - actual||^2

    The model is trained with early stopping based on validation loss.

    Args:
        signals: Training signals
            Shape: (num_signals, signal_length)
            Values: Floating-point time series data
        config: PretrainBoundaryPredictorConfig instance
        verbose: Whether to print training progress (default: True)

    Returns:
        Trained LearnableBoundaryPredictor (in eval mode)

    Raises:
        ValueError: If signals have insufficient length
    """
    if torch.is_tensor(signals):
        signals = signals
    else:
        signals = torch.from_numpy(np.asarray(signals, dtype=np.float32))

    device = torch.device(config.device)

    # Validate signal length
    min_length = config.context_length + config.output_length
    for i, signal in enumerate(signals):
        if signal.shape[0] < min_length:
            raise ValueError(
                f"Signal {i} has length {signal.shape[0]}, "
                f"but requires at least {min_length}"
            )

    # Initialize model
    model = LearnableBoundaryPredictor(
        context_length=config.context_length,
        output_length=config.output_length,
        hidden_dim=config.hidden_dim,
        normalize=True,
    ).to(device)

    # Create dataset: sliding window over each signal
    contexts: List[torch.Tensor] = []
    targets: List[torch.Tensor] = []

    for signal in signals:
        signal = signal.to(device)
        # Slide window: context is [i-L:i], target is [i:i+M]
        for i in range(
            config.context_length,
            len(signal) - config.output_length,
        ):
            context = signal[i - config.context_length : i]
            target = signal[i : i + config.output_length]
            contexts.append(context)
            targets.append(target)

    if not contexts:
        raise ValueError("No valid context-target pairs created from signals")

    contexts_tensor = torch.stack(contexts)
    targets_tensor = torch.stack(targets)

    # Create dataloaders
    dataset = TensorDataset(contexts_tensor, targets_tensor)
    val_size = int(len(dataset) * config.val_split)
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_set,
        batch_size=config.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=config.batch_size,
        shuffle=False,
    )

    # Optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.MSELoss()

    # Training loop with early stopping
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(config.num_epochs):
        # Train
        model.train()
        train_loss = 0.0
        for batch_context, batch_target in train_loader:
            batch_context = batch_context.to(device)
            batch_target = batch_target.to(device)

            optimizer.zero_grad()
            prediction = model(batch_context)
            loss = criterion(prediction, batch_target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_context, batch_target in val_loader:
                batch_context = batch_context.to(device)
                batch_target = batch_target.to(device)

                prediction = model(batch_context)
                loss = criterion(prediction, batch_target)
                val_loss += loss.item()

        val_loss /= len(val_loader)

        if verbose and (epoch + 1) % max(1, config.num_epochs // 10) == 0:
            print(
                f"Epoch {epoch + 1:3d}/{config.num_epochs}: "
                f"train_loss={train_loss:.6f}, val_loss={val_loss:.6f}"
            )

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state_dict = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= config.early_stopping_patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch + 1}")
                break

    # Load best model
    model.load_state_dict(best_state_dict)

    return model.eval()


# ============================================================================
# Unit Tests
# ============================================================================


def test_forward_shape_preservation():
    """Test that output shape matches expected dimensions.

    Verifies that both batch and single-sample inputs produce correctly
    shaped outputs with appropriate batch dimension handling.
    """
    if torch is None:
        return

    predictor = LearnableBoundaryPredictor(
        context_length=10,
        output_length=15,
        hidden_dim=64,
    )

    # Test batch input
    batch_input = torch.randn(32, 10)
    batch_output = predictor(batch_input)
    assert batch_output.shape == (32, 15), (
        f"Expected shape (32, 15), got {batch_output.shape}"
    )

    # Test single sample
    single_input = torch.randn(10)
    single_output = predictor(single_input)
    assert single_output.shape == (15,), (
        f"Expected shape (15,), got {single_output.shape}"
    )

    print("✓ test_forward_shape_preservation passed")


def test_forward_batch_and_single():
    """Test forward pass works for both batch and single samples.

    Verifies that batch and single sample processing produces equivalent
    results when accounting for dimension differences.
    """
    if torch is None:
        return

    predictor = LearnableBoundaryPredictor(
        context_length=10,
        output_length=15,
        hidden_dim=64,
    )

    single_sample = torch.randn(10)

    # Process as single
    single_output = predictor(single_sample)

    # Process as batch of 1
    batch_output = predictor(single_sample.unsqueeze(0))

    # Shapes should be compatible
    assert single_output.shape == (15,)
    assert batch_output.shape == (1, 15)

    # Values should match (with batch dimension)
    torch.testing.assert_close(single_output, batch_output.squeeze(0))

    print("✓ test_forward_batch_and_single passed")


def test_gradient_flow():
    """Test that gradients flow through the entire network.

    Verifies that backward pass computes gradients for all learnable
    parameters and that gradient values are non-zero.
    """
    if torch is None:
        return

    predictor = LearnableBoundaryPredictor(
        context_length=10,
        output_length=15,
        hidden_dim=64,
        normalize=True,
    )

    # Forward pass
    input_sample = torch.randn(32, 10, requires_grad=True)
    output = predictor(input_sample)

    # Backward pass
    loss = output.sum()
    loss.backward()

    # Check that input gradient exists
    assert input_sample.grad is not None
    assert input_sample.grad.abs().sum() > 0, "Input gradient is zero"

    # Check that all parameters have gradients
    for name, param in predictor.named_parameters():
        assert param.grad is not None, f"Parameter {name} has no gradient"
        assert param.grad.abs().sum() > 0, f"Gradient for {name} is zero"

    print("✓ test_gradient_flow passed")


def test_normalization_learnable():
    """Test that input/output normalization scales are learnable.

    Verifies that the learnable normalization parameters (input_scale,
    output_scale) are properly registered as model parameters and can be
    optimized during training.
    """
    if torch is None:
        return

    predictor = LearnableBoundaryPredictor(
        context_length=10,
        output_length=15,
        hidden_dim=64,
        normalize=True,
    )

    # Check that scale parameters exist
    assert predictor.input_scale is not None
    assert predictor.output_scale is not None
    assert predictor.input_scale.shape == (10,)
    assert predictor.output_scale.shape == (15,)

    # Check they are in parameter list
    param_names = [name for name, _ in predictor.named_parameters()]
    assert "input_scale" in param_names
    assert "output_scale" in param_names

    # Test that they're trainable (require grad)
    assert predictor.input_scale.requires_grad
    assert predictor.output_scale.requires_grad

    # Forward and backward to verify gradients
    input_sample = torch.randn(32, 10)
    output = predictor(input_sample)
    loss = output.sum()
    loss.backward()

    assert predictor.input_scale.grad is not None
    assert predictor.output_scale.grad is not None
    assert predictor.input_scale.grad.abs().sum() > 0
    assert predictor.output_scale.grad.abs().sum() > 0

    print("✓ test_normalization_learnable passed")


if __name__ == "__main__":
    test_forward_shape_preservation()
    test_forward_batch_and_single()
    test_gradient_flow()
    test_normalization_learnable()
    print("\n✓ All learnable boundary tests passed")
