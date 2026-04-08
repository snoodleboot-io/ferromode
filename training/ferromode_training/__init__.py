"""
Ferromode LSTM Training Module

Train and validate boundary prediction models for EMD decomposition.
"""

__version__ = "1.0.0"
__author__ = "Ferromode Contributors"

from training.data_generator import generate_synthetic_signals
from training.lstm_model import LSTMPredictor
from training.train import main as train_main
from training.validate import ModelValidator

__all__ = [
    "generate_synthetic_signals",
    "LSTMPredictor",
    "train_main",
    "ModelValidator",
]
