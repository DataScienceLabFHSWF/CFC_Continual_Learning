# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F

from backbone import MammothBackbone, register_backbone
from ncps.torch import CfC
from ncps.wirings import Random

def conv3x3(in_planes: int, out_planes: int, stride: int=1) -> F.conv2d:
    """
    Instantiates a 3x3 convolutional layer with no bias.
    """
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    """
    The basic block of ResNet.
    """
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int=1) -> None:
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet_RandomSparse(MammothBackbone):
    """
    ResNet network with Random Sparse CfC wiring.
    Purpose: Baseline to test if AutoNCP structure matters or just sparsity.
    """

    def __init__(self, block: BasicBlock, num_blocks: List[int],
                 num_classes: int, nf: int, cfc_hidden_size: int = 256,
                 sparsity_level: float = 0.7) -> None:
        super(ResNet_RandomSparse, self).__init__()
        self.in_planes = nf
        self.block = block
        self.num_classes = num_classes
        self.nf = nf
        self.cfc_hidden_size = cfc_hidden_size
        
        # Convolutional layers
        self.conv1 = conv3x3(3, nf * 1)
        self.bn1 = nn.BatchNorm2d(nf * 1)
        self.layer1 = self._make_layer(block, nf * 1, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, nf * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, nf * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, nf * 8, num_blocks[3], stride=2)
        
        self.feature_dim = nf * 8 * block.expansion
        
        # Random sparse wiring
        wiring = Random(cfc_hidden_size, output_dim=num_classes, sparsity_level=sparsity_level)
        self.cfc = CfC(self.feature_dim, wiring, batch_first=True)
        self.cfc_output_size = cfc_hidden_size  # Random outputs full units
        self.linear = nn.Linear(self.cfc_output_size, num_classes)
        
        self._features = nn.Sequential(
            self.conv1,
            self.bn1,
            nn.ReLU(),
            self.layer1,
            self.layer2,
            self.layer3,
            self.layer4
        )
        self.classifier = self.linear
        self.hidden_state = None

    def _make_layer(self, block: BasicBlock, planes: int,
                    num_blocks: int, stride: int) -> nn.Module:
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, returnt='out', hx=None) -> torch.Tensor:
        batch_size = x.size(0)
        
        # Spatial feature extraction
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, out.shape[2])
        feature = out.view(batch_size, -1)
        
        # Process through CfC with random sparse wiring
        feature_seq = feature.unsqueeze(1)
        
        if hx is not None:
            cfc_out, hx_new = self.cfc(feature_seq, hx)
        else:
            cfc_out, hx_new = self.cfc(feature_seq)
        
        cfc_feature = cfc_out.squeeze(1)
        self.hidden_state = hx_new
        
        if returnt == 'features':
            return cfc_feature
        
        out = self.classifier(cfc_feature)

        if returnt == 'out':
            return out
        elif returnt == 'all':
            return (out, cfc_feature)

        raise NotImplementedError("Unknown return type")


@register_backbone('cnn-random-sparse')
def cnn_random_sparse(num_classes: int, nf: int = 64, cfc_hidden_size: int = 256, sparsity_level: float = 0.7):
    """ResNet18 with Random Sparse CfC wiring."""
    return ResNet_RandomSparse(BasicBlock, [2, 2, 2, 2], num_classes, nf, 
                              cfc_hidden_size=cfc_hidden_size, sparsity_level=sparsity_level)
