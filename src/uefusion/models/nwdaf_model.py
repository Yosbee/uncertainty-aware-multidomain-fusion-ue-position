"""Checkpoint-compatible network definition; variance uses exp(clamp(log_variance))."""
import math
from typing import Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_model import BasePositioningModel

class NWDAFPositioning(BasePositioningModel):
    """
    NWDAF: Network Data Analytics Function positioning model.

    Uses Bidirectional LSTM to learn mobility patterns from core network
    information (cell state, transition events, dwell time, and supplied
    speed). The speed is an assumed application input, not an AMF measurement.
    """

    def __init__(self, config: Dict):
        super().__init__(config)

        # Model parameters
        input_size = self.model_config['input_size']
        nwdaf_cfg = self.model_config.get('nwdaf', self.model_config.get('lstm', {}))
        hidden_size = nwdaf_cfg.get('hidden_size', 128)
        num_layers = nwdaf_cfg.get('num_layers', 2)
        bidirectional = nwdaf_cfg.get('bidirectional', True)
        dropout = self.model_config.get('dropout', 0.1)

        # LSTM
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )

        lstm_out_size = hidden_size * (2 if bidirectional else 1)

        # Position prediction head
        self.pos_head = nn.Sequential(
            nn.Linear(lstm_out_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.BatchNorm1d(hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 2),  # x, y
        )

        # Uncertainty head (log-variance parameterization for stability)
        self.unc_fc1 = nn.Linear(lstm_out_size, hidden_size // 2)
        self.unc_fc2 = nn.Linear(hidden_size // 2, 2)
        nn.init.constant_(self.unc_fc2.bias, 4.6)  # ~exp(4.6) = 100

    def forward(self, x: torch.Tensor, return_uncertainty: bool = False):
        """
        Forward pass.

        Args:
            x: Input (batch, seq_len, features)
            return_uncertainty: If True, return (pred, uncertainty)

        Returns:
            pred (batch, 2) or (pred, uncertainty) if return_uncertainty=True
        """
        # LSTM encoding
        lstm_out, (hidden, cell) = self.lstm(x)

        # Last output
        h = lstm_out[:, -1, :]  # (batch, lstm_out_size)

        # Position prediction
        pred = self.pos_head(h)

        if return_uncertainty:
            log_var = F.relu(self.unc_fc1(h))
            log_var = self.unc_fc2(log_var)
            log_var = torch.clamp(log_var, min=-4.0, max=10.0)
            variance = torch.exp(log_var)
            return pred, variance

        return pred
