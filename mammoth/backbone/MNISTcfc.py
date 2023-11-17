# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import avg_pool2d, relu

from backbone import MammothBackbone
from ncps.torch import CfC

# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn

from backbone import MammothBackbone, num_flat_features, xavier
from ncps.torch import CfC

class MNISTcfc(MammothBackbone):
    """
    Network composed of two hidden layers, each containing 100 ReLU activations.
    Designed for the MNIST dataset.
    """

    def __init__(self, input_size: int, output_size: int) -> None:
        """
        Instantiates the layers of the network.
        :param input_size: the size of the input data
        :param output_size: the size of the output
        """
        super(MNISTcfc, self).__init__()

        self.input_size = input_size
        self.output_size = output_size

        #self.fc1 = nn.Linear(self.input_size, 100)
        #self.fc2 = nn.Linear(100, 100)
        self.cfc = CfC(100, 100, batch_first=True, proj_size=None)
        self._features = nn.Sequential(
            self.cfc
        )
        self.classifier = nn.Linear(100, self.output_size)
        self.net = nn.Sequential(self._features, self.classifier)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """
        Calls the Xavier parameter initialization function.
        """
        self.net.apply(xavier)

    def forward(self, x: torch.Tensor, returnt='out') -> torch.Tensor:
        """
        Compute a forward pass.
        :param x: input tensor (batch_size, input_size)
        :return: output tensor (output_size)
        """
        x = x.view(-1, num_flat_features(x))

        feats = self._features(x)

        if returnt == 'features':
            return feats

        out = self.classifier(feats)

        if returnt == 'out':
            return out
        elif returnt == 'all':
            return (out, feats)

        raise NotImplementedError("Unknown return type")
