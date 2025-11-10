# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn

from backbone import MammothBackbone, num_flat_features, register_backbone, xavier
from ncps.torch import CfC
from ncps.wirings import AutoNCP


class BaseMNISTcfc(MammothBackbone):
    """
    Network using CfC (Closed-form Continuous-time) RNN for MNIST.
    
    Architecture Philosophy:
    - Treats flattened MNIST image (784 dims) as a sequence by splitting into chunks
    - Uses CfC to process temporal dependencies across these chunks
    - Leverages NCP wiring for sparse, structured connectivity
    - Hidden state carries information across sequence, potentially helping with
      continual learning by maintaining stable representations
    
    Design for Continual Learning:
    - Sparse NCP wiring may create functional modules that specialize per task
    - Continuous-time dynamics provide smooth, stable representations
    - Liquid time constants allow adaptive temporal processing
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
        :param hidden_size: size of the hidden state in CfC
        :param chunk_size: size of chunks to split input into (28 for MNIST rows)
        """
        super(MNISTcfc, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.chunk_size = chunk_size
        
        # Calculate sequence length (number of chunks)
        # For MNIST: 784 / 28 = 28 timesteps (each row is a timestep)
        self.seq_len = input_size // chunk_size
        
        # Project input chunks to a smaller dimension for efficiency
        self.input_projection = nn.Linear(chunk_size, 64)
        
        # Create CfC layer with optional NCP wiring
        if use_ncp_wiring:
            # Use AutoNCP wiring for structured sparse connectivity
            # Creates sensory, inter, command, and motor neurons
            # Note: AutoNCP requires output_size < units - 2
            # So we use hidden_size for total units, and hidden_size for output (will be projected)
            wiring = AutoNCP(hidden_size, hidden_size // 2)  # Output size must be smaller
            self.cfc = CfC(64, wiring, batch_first=True)
            # The CfC output will be hidden_size//2, so we need to adjust
            self.cfc_output_size = hidden_size // 2
        else:
            # Fully connected for ablation comparison
            self.cfc = CfC(64, hidden_size, batch_first=True)
            self.cfc_output_size = hidden_size
        
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
        # Initialize projection layer
        xavier(self.input_projection)
        # CfC has its own initialization
        # Initialize post-processing layers
        xavier(self.fc_post_cfc)
        xavier(self.classifier)

    def forward(self, x: torch.Tensor, returnt='out', hx=None) -> torch.Tensor:
        """
        Compute a forward pass.
        
        Process:
        1. Flatten input image
        2. Split into sequential chunks (e.g., rows for MNIST)
        3. Project each chunk to lower dimension
        4. Process sequence through CfC RNN
        5. Use final hidden state for classification
        
        :param x: input tensor (batch_size, channels, height, width) or (batch_size, input_size)
        :param returnt: return type ('out', 'features', 'all')
        :param hx: hidden state (optional, for continual learning across batches)
        :return: output tensor (output_classes) or features
        """
        batch_size = x.size(0)
        
        # Flatten input
        x = x.view(batch_size, -1)  # (batch_size, input_size)
        
        # Reshape into sequence: (batch_size, seq_len, chunk_size)
        x = x.view(batch_size, self.seq_len, self.chunk_size)
        
        # Project each chunk
        x = self.input_projection(x)  # (batch_size, seq_len, 64)
        
        # Process through CfC RNN
        if hx is not None:
            rnn_out, hx_new = self.cfc(x, hx)  # rnn_out: (batch_size, seq_len, hidden_size)
        else:
            rnn_out, hx_new = self.cfc(x)
        
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


@register_backbone('mnistcfc')
def mnistcfc(input_size: int, output_size: int, **kwargs):
    """CfC backbone for MNIST with AutoNCP wiring."""
    return BaseMNISTcfc(input_size, output_size, **kwargs)
