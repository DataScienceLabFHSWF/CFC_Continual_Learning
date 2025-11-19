# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn

from backbone import MammothBackbone, register_backbone, xavier
from ncps.torch import LTC
from ncps.wirings import AutoNCP


class BaseMNIST_LTC(MammothBackbone):
    """
    Network using Raw LTC (Liquid Time Constant) RNN for MNIST.
    
    This uses the actual ODE solver-based LTC, not the closed-form approximation (CfC).
    It is expected to be slower but potentially more expressive or stable.
    """

    def __init__(self, input_size: int, output_size: int, 
                 use_ncp_wiring: bool = True, 
                 hidden_size: int = 128,
                 chunk_size: int = 28) -> None:
        """
        Instantiates the layers of the network.
        :param input_size: the size of the input data (784 for MNIST)
        :param output_size: the size of the output (number of classes)
        :param use_ncp_wiring: whether to use NCP wiring (True) or fully-connected (False)
        :param hidden_size: size of the hidden state in LTC
        :param chunk_size: size of chunks to split input into (28 for MNIST rows)
        """
        super(BaseMNIST_LTC, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.chunk_size = chunk_size
        
        # Calculate sequence length (number of chunks)
        # For MNIST: 784 / 28 = 28 timesteps (each row is a timestep)
        self.seq_len = input_size // chunk_size
        
        # Project input chunks to a smaller dimension for efficiency
        self.input_projection = nn.Linear(chunk_size, 64)
        
        # Create LTC layer with optional NCP wiring
        if use_ncp_wiring:
            # Use AutoNCP wiring for structured sparse connectivity
            wiring = AutoNCP(hidden_size, hidden_size // 2)
            # LTC uses an ODE solver
            self.ltc = LTC(64, wiring, batch_first=True)
            self.ltc_output_size = hidden_size // 2
        else:
            # Fully connected
            self.ltc = LTC(64, hidden_size, batch_first=True)
            self.ltc_output_size = hidden_size
        
        # Additional feedforward layer after LTC
        self.fc_post_ltc = nn.Linear(self.ltc_output_size, 100)
        
        self._features = nn.Sequential(
            self.fc_post_ltc,
            nn.ReLU(),
        )
        
        self.classifier = nn.Linear(100, self.output_size)
        
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """
        Calls the Xavier parameter initialization function.
        """
        xavier(self.input_projection)
        # LTC has its own initialization
        xavier(self.fc_post_ltc)
        xavier(self.classifier)

    def forward(self, x: torch.Tensor, returnt='out', hx=None) -> torch.Tensor:
        """
        Compute a forward pass.
        """
        batch_size = x.size(0)
        
        # Flatten input
        x = x.view(batch_size, -1)  # (batch_size, input_size)
        
        # Reshape into sequence: (batch_size, seq_len, chunk_size)
        x = x.view(batch_size, self.seq_len, self.chunk_size)
        
        # Project each chunk
        x = self.input_projection(x)  # (batch_size, seq_len, 64)
        
        # Process through LTC RNN
        if hx is not None:
            rnn_out, hx_new = self.ltc(x, hx)
        else:
            rnn_out, hx_new = self.ltc(x)
        
        # Use final timestep output
        final_state = rnn_out[:, -1, :]  # (batch_size, hidden_size)
        
        # Post-processing
        feats = self._features(final_state)  # (batch_size, 100)

        if returnt == 'features':
            return feats

        out = self.classifier(feats)

        if returnt == 'out':
            return out
        elif returnt == 'all':
            return (out, feats)

        raise NotImplementedError("Unknown return type")


@register_backbone('mnistltc')
def mnistltc(input_size: int = 784, output_size: int = 10, 
             hidden_size: int = 256, chunk_size: int = 28, use_ncp_wiring: bool = True):
    """Raw LTC backbone for MNIST with AutoNCP wiring."""
    return BaseMNIST_LTC(input_size, output_size, hidden_size=hidden_size, 
                        chunk_size=chunk_size, use_ncp_wiring=use_ncp_wiring)
