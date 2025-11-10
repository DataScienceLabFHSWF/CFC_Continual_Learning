# Copyright 2024-present
# CfC backbone specifically for Tennessee Eastman Process fault detection

import torch
import torch.nn as nn

from backbone import MammothBackbone, register_backbone
from ncps.torch import CfC
from ncps.wirings import AutoNCP, FullyConnected


class BaseTEPCfC(MammothBackbone):
    """
    CfC network for Tennessee Eastman Process fault detection.
    
    Architecture:
    - Input: (batch, seq_len, 52) - time series of 52 process variables
    - CfC layer: Processes temporal dynamics
    - Output: (batch, 22) - classification over 22 fault types (normal + 21 faults)
    """
    
    def __init__(
        self,
        input_size=52,
        num_classes=22,
        hidden_size=128,
        use_ncp_wiring=True,
        use_ltc=False
    ):
        """
        Args:
            input_size: Number of process variables (52 for TEP)
            num_classes: Number of fault classes (22 for TEP)
            hidden_size: Size of CfC hidden state
            use_ncp_wiring: If True, use sparse NCP wiring; else fully-connected
            use_ltc: If True, use LTC (liquid time constants); else CfC
        """
        super().__init__()
        
        if not NCPS_AVAILABLE:
            raise ImportError("ncps package not available. Please install with: poetry run pip install -e ./ncps")
        
        self.input_size = input_size
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.use_ncp_wiring = use_ncp_wiring
        self.use_ltc = use_ltc
        
        # Project input to hidden dimension
        self.input_projection = nn.Linear(input_size, hidden_size)
        
        # CfC/LTC layer
        if use_ncp_wiring:
            # AutoNCP requires output_size < units - 2
            # So we use hidden_size units and hidden_size//2 output
            wiring = AutoNCP(hidden_size, num_classes)
            self.rnn = CfC(hidden_size, wiring, batch_first=True, mixed_memory=not use_ltc)
        else:
            # Fully-connected CfC (for ablation)
            wiring = FullyConnected(hidden_size, num_classes)
            self.rnn = CfC(hidden_size, wiring, batch_first=True, mixed_memory=not use_ltc)
        
        # Output layer (CfC already outputs num_classes, but we add a final linear for flexibility)
        self.output_layer = nn.Linear(num_classes, num_classes)
        
        # Hidden state (persistent across batches for continual learning)
        self.hidden_state = None
        
    def forward(self, x, return_features=False):
        """
        Forward pass through TEP-CfC network.
        
        Args:
            x: Input tensor of shape (batch, seq_len, input_size) or (batch, input_size)
            return_features: If True, return intermediate features
            
        Returns:
            logits: (batch, num_classes)
            features: (batch, hidden_size) if return_features=True
        """
        batch_size = x.size(0)
        
        # Handle both 2D and 3D inputs
        if x.dim() == 2:
            # Input is (batch, input_size), add time dimension
            x = x.unsqueeze(1)  # (batch, 1, input_size)
        
        # x is now (batch, seq_len, input_size)
        seq_len = x.size(1)
        
        # Project input to hidden dimension
        x = self.input_projection(x)  # (batch, seq_len, hidden_size)
        
        # Process through CfC
        # Initialize hidden state if needed
        if self.hidden_state is None or self.hidden_state.size(0) != batch_size:
            self.hidden_state = None
        
        output, self.hidden_state = self.rnn(x, self.hidden_state)
        # output: (batch, seq_len, num_classes)
        
        # Take the last timestep's output
        last_output = output[:, -1, :]  # (batch, num_classes)
        
        # Final linear layer
        logits = self.output_layer(last_output)  # (batch, num_classes)
        
        if return_features:
            # Return hidden state as features
            features = self.hidden_state
            return logits, features
        
        return logits
    
    def reset_hidden(self):
        """Reset hidden state (call between tasks in continual learning)."""
        self.hidden_state = None
    
    def get_params(self):
        """Return number of parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class TEPLSTM(nn.Module):
    """
    LSTM baseline for Tennessee Eastman Process fault detection.
    For comparison with CfC.
    """
    
    def __init__(
        self,
        input_size=52,
        num_classes=22,
        hidden_size=128,
        num_layers=2
    ):
        super().__init__()
        
        self.input_size = input_size
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0
        )
        
        self.output_layer = nn.Linear(hidden_size, num_classes)
        self.hidden_state = None
        
    def forward(self, x, return_features=False):
        batch_size = x.size(0)
        
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        # Initialize hidden state if needed
        if self.hidden_state is None or self.hidden_state[0].size(1) != batch_size:
            self.hidden_state = None
        
        output, self.hidden_state = self.lstm(x, self.hidden_state)
        last_output = output[:, -1, :]
        logits = self.output_layer(last_output)
        
        if return_features:
            return logits, last_output
        
        return logits
    
    def reset_hidden(self):
        self.hidden_state = None
    
    def get_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


@register_backbone('tepcfc')
def tepcfc(input_size: int, output_size: int, **kwargs):
    """CfC backbone for Tennessee Eastman Process fault detection."""
    return BaseTEPCfC(input_size=input_size, num_classes=output_size, **kwargs)


@register_backbone('teplstm')
def teplstm(input_size: int, output_size: int, **kwargs):
    """LSTM baseline for Tennessee Eastman Process."""
    return TEPLSTM(input_size=input_size, num_classes=output_size, **kwargs)
