"""Minimal configuration holder used by the released network definitions."""
import torch.nn as nn
class BasePositioningModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model_config = config['model']
