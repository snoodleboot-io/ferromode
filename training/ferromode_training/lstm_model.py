"""
LSTM Neural Network architecture for boundary prediction.

Defines the LSTM model that learns to predict signal extensions
for boundary handling in EMD decomposition.
"""

from typing import Tuple, Optional


class LSTMPredictor:
    """
    LSTM boundary prediction model.

    Learns to predict signal extensions for boundary handling.
    Implemented as a wrapper that creates PyTorch LSTM when needed.
    """

    def __init__(
        self,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        input_size: int = 1,
        output_size: int = 10,
    ):
        """
        Initialize LSTM model architecture.

        Args:
            hidden_size: Number of hidden units in LSTM cells (default: 128)
            num_layers: Number of stacked LSTM layers (default: 2)
            dropout: Dropout probability between layers (default: 0.2)
            input_size: Input feature dimension (default: 1 for univariate)
            output_size: Output dimension - number of prediction steps (default: 10)
        """
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.input_size = input_size
        self.output_size = output_size
        self.model = None
        self.device = None

    def build(self) -> "LSTMPredictor":
        """
        Build the PyTorch model.

        Returns:
            self for method chaining
        """
        try:
            import torch
            import torch.nn as nn

            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            class _LSTMNet(nn.Module):
                """Internal PyTorch LSTM network."""

                def __init__(self, hidden_size, num_layers, input_size, output_size):
                    super().__init__()
                    self.lstm = nn.LSTM(
                        input_size=input_size,
                        hidden_size=hidden_size,
                        num_layers=num_layers,
                        batch_first=True,
                        dropout=0.2 if num_layers > 1 else 0.0,
                    )
                    self.fc = nn.Linear(hidden_size, output_size)

                def forward(self, x: torch.Tensor) -> torch.Tensor:
                    """
                    Forward pass.

                    Args:
                        x: Input tensor of shape [batch, seq_len]

                    Returns:
                        Output predictions of shape [batch, output_size]
                    """
                    # x: [batch, seq_len] -> [batch, seq_len, 1]
                    x = x.unsqueeze(-1)
                    lstm_out, _ = self.lstm(x)
                    # Use last LSTM output
                    last_out = lstm_out[:, -1, :]
                    output = torch.tanh(self.fc(last_out))
                    return output

            self.model = _LSTMNet(
                self.hidden_size, self.num_layers, self.input_size, self.output_size
            ).to(self.device)
            return self

        except ImportError:
            raise ImportError("PyTorch not available. Install with: pip install torch")

    def get_model(self):
        """
        Get the underlying PyTorch model.

        Returns:
            PyTorch nn.Module or None if not built
        """
        return self.model

    def get_device(self):
        """
        Get the device the model is on.

        Returns:
            torch.device or None if not built
        """
        return self.device

    def __repr__(self) -> str:
        return (
            f"LSTMPredictor(hidden_size={self.hidden_size}, "
            f"num_layers={self.num_layers}, "
            f"input_size={self.input_size}, "
            f"output_size={self.output_size})"
        )
