# Copyright 2024-present
# Random Sparse CfC backbone for Tennessee Eastman Process

import torch
import torch.nn as nn

from backbone import MammothBackbone, register_backbone
from ncps.torch import CfC
from ncps.wirings import Random


class BaseTEP_RandomSparse(MammothBackbone):
    """
    CfC with Random Sparse wiring for TEP.
    Baseline to test if AutoNCP structure matters.
    """
    
    def __init__(
        self,
        input_size=52,
        num_classes=22,
        hidden_size=128,
        sparsity_level=0.7
    ):
        super().__init__()
        
        self.input_size = input_size
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        
        # Project input to hidden dimension
        self.input_projection = nn.Linear(input_size, hidden_size)
        
        # Random sparse wiring
        wiring = Random(hidden_size, output_dim=num_classes, sparsity_level=sparsity_level)
        self.cfc = CfC(hidden_size, wiring, batch_first=True)
        
        # Output layer
        self.output_layer = nn.Linear(hidden_size, num_classes)  # Random outputs full hidden_size
        self.hidden_state = None
        
    def forward(self, x, returnt='out'):
        batch_size = x.size(0)
        
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        # Project input
        x = self.input_projection(x)
        
        # Process through CfC
        if self.hidden_state is not None:
            cfc_out, self.hidden_state = self.cfc(x, self.hidden_state)
        else:
            cfc_out, self.hidden_state = self.cfc(x)
        
        # Use last timestep
        features = cfc_out[:, -1, :]
        
        if returnt == 'features':
            return features
        
        out = self.output_layer(features)
        
        if returnt == 'out':
            return out
        elif returnt in ['both', 'all']:
            return (out, features)
        
        raise NotImplementedError(f"Unknown return type: {returnt}")


@register_backbone('tep_random_sparse')
def tep_random_sparse(input_size=52, num_classes=22, hidden_size=128, sparsity_level=0.7):
    """Random Sparse CfC backbone for TEP dataset."""
    return BaseTEP_RandomSparse(input_size, num_classes, hidden_size, sparsity_level)
