#!/usr/bin/env python3
"""
PyTorch LSTM model definition for boundary prediction.

Architecture:
- Input: Time series signal (batch_size, seq_length, 1)
- LSTM: 2 layers, 128 hidden units, dropout=0.2
- Output: Single value (stationarity score) or boundary prediction

This model is trained to predict whether a signal is stationary (score > 0.7)
or non-stationary (score ≤ 0.7), which determines whether to use AR or LSTM
boundary prediction.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional


class BoundaryPredictorLSTM(nn.Module):
    """
    LSTM model for predicting signal boundaries and stationarity.

    Architecture:
    - 2-layer LSTM with 128 hidden units
    - Tanh activation
    - Dropout for regularization
    - Single output neuron with Tanh activation (output range [-1, 1])

    Input shape: (batch_size, seq_length, input_size)
    Output shape: (batch_size, 1)
    """

    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 128,
        num_layers: int = 2,
        output_size: int = 1,
        dropout: float = 0.2,
        bidirectional: bool = False,
    ):
        """
        Initialize LSTM model.

        Args:
            input_size: Input feature dimension (default: 1 for univariate signals)
            hidden_size: Number of LSTM hidden units (default: 128)
            num_layers: Number of LSTM layers (default: 2)
            output_size: Output dimension (default: 1 for binary classification)
            dropout: Dropout probability for LSTM (default: 0.2)
            bidirectional: Whether to use bidirectional LSTM (default: False)
        """
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.bidirectional = bidirectional

        # LSTM layer
        # Note: dropout is only applied between layers if num_layers > 1
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        # Fully connected layer
        lstm_output_size = hidden_size * (2 if bidirectional else 1)
        self.fc = nn.Linear(lstm_output_size, output_size)

        # Output activation: Tanh for range [-1, 1]
        # Will be rescaled to [0, 1] (stationarity score) during inference
        self.activation = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, seq_length, input_size)

        Returns:
            Output tensor of shape (batch_size, output_size)
        """
        # LSTM forward pass
        # lstm_out: (batch_size, seq_length, hidden_size*num_directions)
        # (h_n, c_n): tuple of (num_layers*num_directions, batch_size, hidden_size)
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Use the last timestep's output
        last_output = lstm_out[:, -1, :]  # (batch_size, hidden_size*num_directions)

        # Fully connected layer
        output = self.fc(last_output)  # (batch_size, output_size)

        # Apply activation
        output = self.activation(output)

        return output

    def forward_with_hidden(
        self,
        x: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass with explicit hidden state (for sequence processing).

        Args:
            x: Input tensor of shape (batch_size, seq_length, input_size)
            hidden: Optional tuple of (h_n, c_n) for stateful processing

        Returns:
            Tuple of (output, (h_n, c_n))
        """
        lstm_out, hidden = self.lstm(x, hidden)
        last_output = lstm_out[:, -1, :]
        output = self.fc(last_output)
        output = self.activation(output)
        return output, hidden

    def get_config(self) -> dict:
        """Get model configuration for serialization."""
        return {
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "output_size": self.output_size,
            "bidirectional": self.bidirectional,
        }


class StationarityScoreHead(nn.Module):
    """
    Alternative model head for continuous stationarity score prediction.

    Unlike binary classification, this predicts a continuous score [0, 1]
    for finer-grained decision making.
    """

    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        """Initialize stationarity score predictor."""
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()  # Output range [0, 1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, seq_length, input_size)

        Returns:
            Stationarity score [0, 1] of shape (batch_size, 1)
        """
        lstm_out, _ = self.lstm(x)
        score = self.fc(lstm_out[:, -1, :])
        score = self.sigmoid(score)
        return score


def create_model(
    model_type: str = "lstm",
    input_size: int = 1,
    hidden_size: int = 128,
    num_layers: int = 2,
    output_size: int = 1,
    dropout: float = 0.2,
) -> nn.Module:
    """
    Factory function to create model instances.

    Args:
        model_type: Type of model ("lstm" or "stationarity_score")
        input_size: Input feature dimension
        hidden_size: LSTM hidden size
        num_layers: Number of LSTM layers
        output_size: Output dimension (for LSTM model)
        dropout: Dropout probability

    Returns:
        Instantiated model
    """
    if model_type == "lstm":
        return BoundaryPredictorLSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            output_size=output_size,
            dropout=dropout,
        )
    elif model_type == "stationarity_score":
        return StationarityScoreHead(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    # Test instantiation and forward pass
    print("Testing LSTM model...")

    model = BoundaryPredictorLSTM(input_size=1, hidden_size=128, num_layers=2)
    print(f"Model: {model}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

    # Create dummy input
    batch_size = 32
    seq_length = 1000
    input_size = 1
    dummy_input = torch.randn(batch_size, seq_length, input_size)

    # Forward pass
    output = model(dummy_input)
    print(f"\nInput shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")

    # Test with stationarity score model
    print("\n" + "=" * 50)
    print("Testing Stationarity Score model...")

    score_model = StationarityScoreHead(input_size=1, hidden_size=128, num_layers=2)
    print(f"Model: {score_model}")
    print(f"Number of parameters: {sum(p.numel() for p in score_model.parameters())}")

    score_output = score_model(dummy_input)
    print(f"\nInput shape: {dummy_input.shape}")
    print(f"Output shape: {score_output.shape}")
    print(
        f"Output range: [{score_output.min().item():.4f}, {score_output.max().item():.4f}]"
    )
