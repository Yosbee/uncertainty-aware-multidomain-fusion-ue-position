from .randaf_model import RANDAFPositioning
from .nwdaf_model import NWDAFPositioning
from .lstm_model import LSTMPositioning

def make_model(name):
    if name == 'ran':
        return RANDAFPositioning({'model': {'input_size': 35, 'dropout': .1, 'randaf': {'d_model': 128, 'nhead': 8, 'num_layers': 4, 'dim_feedforward': 512}}})
    if name == 'core':
        return NWDAFPositioning({'model': {'input_size': 15, 'dropout': .1, 'nwdaf': {'hidden_size': 128, 'num_layers': 2, 'bidirectional': True}}})
    if name in ('central44', 'central42'):
        return LSTMPositioning({'model': {'input_size': 44 if name == 'central44' else 42, 'hidden_size': 128, 'dropout': .1, 'lstm': {'num_layers': 2, 'bidirectional': True}}})
    raise ValueError(name)
