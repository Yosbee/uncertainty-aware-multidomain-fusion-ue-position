"""Checkpoint-compatible network definition; variance uses exp(clamp(log_variance))."""
import math
from typing import Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_model import BasePositioningModel

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer."""

    def __init__(self, d_model: int, max_len: int = 100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class RANDAFPositioning(BasePositioningModel):
    """
    RANDAF: RAN Data Analytics Function positioning model.

    Uses Transformer architecture to learn radio fingerprint patterns
    across multiple base stations. Outputs position prediction with
    native variance outputs for downstream output fusion.
    """

    def __init__(self, config: Dict):
        super().__init__(config)

        # Model parameters
        input_size = self.model_config['input_size']
        randaf_cfg = self.model_config.get('randaf', self.model_config.get('transformer', {}))
        d_model = randaf_cfg.get('d_model', 128)
        nhead = randaf_cfg.get('nhead', 8)
        num_layers = randaf_cfg.get('num_layers', 4)
        dim_feedforward = randaf_cfg.get('dim_feedforward', 512)
        dropout = self.model_config.get('dropout', 0.1)

        # Input projection
        self.input_projection = nn.Linear(input_size, d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model)

        # Layer normalization (pre-norm)
        self.layer_norm = nn.LayerNorm(d_model)

        # Transformer encoder with GELU activation
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # Position prediction head
        self.pos_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 2),  # x, y
        )

        # Uncertainty head (log-variance parameterization for stability)
        self.unc_fc1 = nn.Linear(d_model, d_model // 4)
        self.unc_fc2 = nn.Linear(d_model // 4, 2)
        # Initialize to predict log(100) ~ 4.6 → variance ~100 (reasonable for meter-scale targets)
        nn.init.constant_(self.unc_fc2.bias, 4.6)

        # Store attention weights for visualization
        self.attention_weights = None

    def forward(self, x: torch.Tensor, return_uncertainty: bool = False):
        """
        Forward pass.

        Args:
            x: Input (batch, seq_len, features)
            return_uncertainty: If True, return (pred, uncertainty)

        Returns:
            pred (batch, 2) or (pred, uncertainty) if return_uncertainty=True
        """
        # Project to d_model
        x = self.input_projection(x)
        x = self.pos_encoder(x)
        x = self.layer_norm(x)

        # Transformer encoding
        x = self.transformer_encoder(x)

        # Last token representation
        h = x[:, -1, :]  # (batch, d_model)

        # Position prediction
        pred = self.pos_head(h)

        if return_uncertainty:
            log_var = self.unc_fc1(h)
            log_var = F.relu(log_var)
            log_var = self.unc_fc2(log_var)
            # Clamp for stability, then exponentiate
            log_var = torch.clamp(log_var, min=-4.0, max=10.0)
            variance = torch.exp(log_var)
            return pred, variance

        return pred

    def get_attention_weights(self) -> Optional[torch.Tensor]:
        return self.attention_weights
