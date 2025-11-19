# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F

from backbone import MammothBackbone, register_backbone
from ncps.torch import LTC
from ncps.wirings import AutoNCP

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


class ResNet_LTC(MammothBackbone):
    """
    ResNet network with LTC (Liquid Time Constant) temporal processing.
    Uses ODE solver for continuous-time dynamics.
    """

    def __init__(self, block: BasicBlock, num_blocks: List[int],
                 num_classes: int, nf: int, ltc_hidden_size: int = 256) -> None:
        super(ResNet_LTC, self).__init__()
        self.in_planes = nf
        self.block = block
        self.num_classes = num_classes
        self.nf = nf
        self.ltc_hidden_size = ltc_hidden_size
        
        # Convolutional layers (spatial feature extraction)
        self.conv1 = conv3x3(3, nf * 1)
        self.bn1 = nn.BatchNorm2d(nf * 1)
        self.layer1 = self._make_layer(block, nf * 1, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, nf * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, nf * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, nf * 8, num_blocks[3], stride=2)
        
        # Feature dimension after conv layers
        self.feature_dim = nf * 8 * block.expansion
        
        # LTC as a recurrent processing layer
        wiring = AutoNCP(ltc_hidden_size, ltc_hidden_size // 2)
        self.ltc = LTC(self.feature_dim, wiring, batch_first=True)
        self.ltc_output_size = ltc_hidden_size // 2
        self.linear = nn.Linear(self.ltc_output_size, num_classes)
        
        # For feature extraction interface
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
        
        # Hidden state for LTC
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
        
        # Spatial feature extraction (ResNet)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, out.shape[2])
        feature = out.view(batch_size, -1)
        
        # Process features through LTC
        feature_seq = feature.unsqueeze(1)  # (batch_size, 1, feature_dim)
        
        if hx is not None:
            ltc_out, hx_new = self.ltc(feature_seq, hx)
        else:
            ltc_out, hx_new = self.ltc(feature_seq)
        
        ltc_feature = ltc_out.squeeze(1)
        self.hidden_state = hx_new
        
        if returnt == 'features':
            return ltc_feature
        
        out = self.classifier(ltc_feature)

        if returnt == 'out':
            return out
        elif returnt == 'all':
            return (out, ltc_feature)

        raise NotImplementedError("Unknown return type")


@register_backbone('cnn-ltc')
def cnn_ltc(num_classes: int, nf: int = 64, ltc_hidden_size: int = 256):
    """ResNet18 with LTC temporal processing."""
    return ResNet_LTC(BasicBlock, [2, 2, 2, 2], num_classes, nf, ltc_hidden_size=ltc_hidden_size)
