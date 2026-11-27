"""Checkpoint-compatible network definition; variance uses exp(clamp(log_variance))."""
import math
from typing import Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_model import BasePositioningModel

class LSTMPositioning(BasePositioningModel):
    """LSTM-based model for vehicle positioning"""
    
    def __init__(self, config: Dict):
        """
        Initialize LSTM model
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        
        # Model parameters
        input_size = self.model_config['input_size']
        hidden_size = self.model_config.get('hidden_size', 128)
        num_layers = self.model_config['lstm']['num_layers']
        bidirectional = self.model_config['lstm']['bidirectional']
        dropout = self.model_config['dropout']
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Calculate output size based on bidirectionality
        lstm_output_size = hidden_size * (2 if bidirectional else 1)
        
        # Output layers
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(lstm_output_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, 2)  # Output: x, y coordinates
        
        # Batch normalization
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size // 2)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through LSTM
        
        Args:
            x: Input sequences (batch_size, seq_len, features)
            
        Returns:
            Predicted positions (batch_size, 2)
        """
        # Pass through LSTM
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Use the last output
        x = lstm_out[:, -1, :]  # (batch_size, hidden_size * num_directions)
        
        # Apply dropout
        x = self.dropout(x)
        
        # First fully connected layer with batch norm and ReLU
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Second fully connected layer with batch norm and ReLU
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        
        # Output layer
        x = self.fc3(x)
        
        return x
