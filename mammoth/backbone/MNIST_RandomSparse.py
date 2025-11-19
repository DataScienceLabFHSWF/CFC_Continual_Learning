# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn

from backbone import MammothBackbone, register_backbone, xavier
from ncps.torch import CfC
from ncps.wirings import Random


class BaseMNIST_RandomSparse(MammothBackbone):
    """
    Network using CfC RNN with Random Sparse Wiring for MNIST.
    
    Purpose:
    - To serve as a baseline for the "Modularity Hypothesis".
    - Compares structured sparsity (AutoNCP) vs. random sparsity.
    - If AutoNCP outperforms this, it suggests the specific wiring matters.
    """

    def __init__(self, input_size: int, output_size: int, 
                 sparsity_level: float = 0.5, 
                 hidden_size: int = 128,
                 chunk_size: int = 28) -> None:
        """
        Instantiates the layers of the network.
        :param input_size: the size of the input data (784 for MNIST)
        :param output_size: the size of the output (number of classes)
        :param sparsity_level: level of sparsity (0.0 = dense, 0.9 = 90% sparse)
        :param hidden_size: size of the hidden state in CfC
        :param chunk_size: size of chunks to split input into (28 for MNIST rows)
        """
        super(BaseMNIST_RandomSparse, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.chunk_size = chunk_size
        
        # Calculate sequence length (number of chunks)
        self.seq_len = input_size // chunk_size
        
        # Project input chunks to a smaller dimension for efficiency
        self.input_projection = nn.Linear(chunk_size, 64)
        
        # Use Random wiring
        # Note: For Random wiring, the output is always 'units' (not output_dim)
        # output_dim is just for marking motor neurons
        wiring = Random(hidden_size, output_dim=output_size, sparsity_level=sparsity_level)
        
        self.cfc = CfC(64, wiring, batch_first=True)
        self.cfc_output_size = hidden_size  # Output is same as units
        
        # Additional feedforward layer after CfC
        self.fc_post_cfc = nn.Linear(self.cfc_output_size, 100)
        
        self._features = nn.Sequential(
            self.fc_post_cfc,
            nn.ReLU(),
        )
        
        self.classifier = nn.Linear(100, self.output_size)
        
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """
        Calls the Xavier parameter initialization function.
        """
        xavier(self.input_projection)
        xavier(self.fc_post_cfc)
        xavier(self.classifier)

    def forward(self, x: torch.Tensor, returnt='out', hx=None) -> torch.Tensor:
        """
        Compute a forward pass.
        """
        batch_size = x.size(0)
        
        # Flatten input
        x = x.view(batch_size, -1)
        
        # Reshape into sequence
        x = x.view(batch_size, self.seq_len, self.chunk_size)
        
        # Project each chunk
        x = self.input_projection(x)
        
        # Process through CfC RNN
        if hx is not None:
            rnn_out, hx_new = self.cfc(x, hx)
        else:
            rnn_out, hx_new = self.cfc(x)
        
        # Use final timestep output
        final_state = rnn_out[:, -1, :]
        
        # Post-processing
        feats = self._features(final_state)

        if returnt == 'features':
            return feats

        out = self.classifier(feats)

        if returnt == 'out':
            return out
        elif returnt == 'all':
            return (out, feats)

        raise NotImplementedError("Unknown return type")


@register_backbone('mnist_random_sparse')
def mnist_random_sparse(input_size: int = 784, output_size: int = 10, 
                        hidden_size: int = 256, chunk_size: int = 28, 
                        sparsity_level: float = 0.7):
    """CfC backbone with Random Sparse wiring."""
    return BaseMNIST_RandomSparse(input_size, output_size, hidden_size=hidden_size, 
                                 chunk_size=chunk_size, sparsity_level=sparsity_level)
