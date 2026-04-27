# Copyright 2024-present
# LTC backbone for Tennessee Eastman Process fault detection

import torch
import torch.nn as nn

from backbone import MammothBackbone, register_backbone
from ncps.torch import LTC
from ncps.wirings import AutoNCP


class BaseTEP_LTC(MammothBackbone):
    """
    Raw LTC network for Tennessee Eastman Process fault detection.
    Uses ODE solver for true continuous-time dynamics.
    """
    
    def __init__(
        self,
        input_size=52,
        num_classes=22,
        hidden_size=128,
        use_ncp_wiring=True
    ):
        super().__init__()
        
        self.input_size = input_size
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        
        # Project input to hidden dimension
        self.input_projection = nn.Linear(input_size, hidden_size)
        
        # LTC layer
        if use_ncp_wiring:
            wiring = AutoNCP(hidden_size, num_classes)
        else:
            from ncps.wirings import FullyConnected
            wiring = FullyConnected(hidden_size, num_classes)
        
        self.ltc = LTC(hidden_size, wiring, batch_first=True)
        
        # Output layer
        self.output_layer = nn.Linear(num_classes, num_classes)
        self.classifier = self.output_layer
        self.hidden_state = None
        
    def forward(self, x, returnt='out'):
        batch_size = x.size(0)
        
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        # Project input
        x = self.input_projection(x)
        
        # Process through LTC
        # Reset hidden state each batch to avoid size mismatch on last batch
        self.hidden_state = None
        ltc_out, self.hidden_state = self.ltc(x)
        
        # Use last timestep
        features = ltc_out[:, -1, :]
        
        if returnt == 'features':
            return features
        
        out = self.output_layer(features)
        
        if returnt == 'out':
            return out
        elif returnt in ['both', 'all']:
            return (out, features)
        
        raise NotImplementedError(f"Unknown return type: {returnt}")


@register_backbone('tepltc')
def tepltc(input_size=52, num_classes=22, hidden_size=128, use_ncp_wiring=True):
    """LTC backbone for TEP dataset."""
    return BaseTEP_LTC(input_size, num_classes, hidden_size, use_ncp_wiring)
