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

def conv3x3(in_planes: int, out_planes: int, stride: int=1) -> F.conv2d:
    """
    Instantiates a 3x3 convolutional layer with no bias.
    :param in_planes: number of input channels
    :param out_planes: number of output channels
    :param stride: stride of the convolution
    :return: convolutional layer
    """
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    """
    The basic block of ResNet.
    """
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int=1) -> None:
        """
        Instantiates the basic block of the network.
        :param in_planes: the number of input channels
        :param planes: the number of channels (to be possibly expanded)
        """
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
        """
        Compute a forward pass.
        :param x: input tensor (batch_size, input_size)
        :return: output tensor (10)
        """
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(MammothBackbone):
    """
    ResNet network architecture. Designed for complex datasets.
    """

    def __init__(self, block: BasicBlock, num_blocks: List[int],
                 num_classes: int, nf: int, use_cfc: bool = True, 
                 cfc_hidden_size: int = 256) -> None:
        """
        Instantiates the layers of the network.
        
        Architecture Philosophy:
        - ResNet extracts spatial features from images
        - CfC processes these features temporally (useful for video/sequences)
        - For single images in CL: CfC acts as a recurrent readout layer
          that maintains hidden state across batches/tasks
        
        :param block: the basic ResNet block
        :param num_blocks: the number of blocks per layer
        :param num_classes: the number of output classes
        :param nf: the number of filters
        :param use_cfc: whether to use CfC layer (True) or standard linear (False)
        :param cfc_hidden_size: hidden size for CfC layer
        """
        super(ResNet, self).__init__()
        self.in_planes = nf
        self.block = block
        self.num_classes = num_classes
        self.nf = nf
        self.use_cfc = use_cfc
        self.cfc_hidden_size = cfc_hidden_size
        
        # Convolutional layers (spatial feature extraction)
        self.conv1 = conv3x3(3, nf * 1)
        self.bn1 = nn.BatchNorm2d(nf * 1)
        self.layer1 = self._make_layer(block, nf * 1, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, nf * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, nf * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, nf * 8, num_blocks[3], stride=2)
        
        # Feature dimension after conv layers
        self.feature_dim = nf * 8 * block.expansion
        
        if self.use_cfc:
            # CfC as a recurrent processing layer
            # Note: For single images, we can either:
            # 1. Treat each spatial location as a timestep (use feature maps)
            # 2. Use CfC as a stateful readout (process batch as sequence)
            # Here we use option 2: process features with recurrent dynamics
            
            from ncps.wirings import AutoNCP
            # Use NCP wiring for structured sparsity
            # AutoNCP requires output_size < units - 2
            wiring = AutoNCP(cfc_hidden_size, cfc_hidden_size // 2)
            self.cfc = CfC(self.feature_dim, wiring, batch_first=True)
            self.cfc_output_size = cfc_hidden_size // 2
            self.linear = nn.Linear(self.cfc_output_size, num_classes)
        else:
            # Standard feedforward classifier for ablation comparison
            self.cfc = None
            self.cfc_output_size = self.feature_dim
            self.linear = nn.Linear(self.feature_dim, num_classes)
        
        # For feature extraction interface (used by continual learning methods)
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
        
        # Hidden state for CfC (persistent across batches if needed)
        self.hidden_state = None

    def _make_layer(self, block: BasicBlock, planes: int,
                    num_blocks: int, stride: int) -> nn.Module:
        """
        Instantiates a ResNet layer.
        :param block: ResNet basic block
        :param planes: channels across the network
        :param num_blocks: number of blocks
        :param stride: stride
        :return: ResNet layer
        """
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, returnt='out', hx=None) -> torch.Tensor:
        """
        Compute a forward pass.
        
        For CfC version:
        - Extract spatial features with ResNet
        - Process through CfC for temporal/recurrent processing
        - CfC maintains hidden state that could stabilize learning
        
        :param x: input tensor (batch_size, channels, height, width)
        :param returnt: return type (a string among 'out', 'features', 'all')
        :param hx: hidden state for CfC (optional)
        :return: output tensor (output_classes) or features
        """
        batch_size = x.size(0)
        
        # Spatial feature extraction (ResNet)
        out = F.relu(self.bn1(self.conv1(x)))  # 64, 32, 32
        if hasattr(self, 'maxpool'):
            out = self.maxpool(out)
        out = self.layer1(out)  # -> 64, 32, 32
        out = self.layer2(out)  # -> 128, 16, 16
        out = self.layer3(out)  # -> 256, 8, 8
        out = self.layer4(out)  # -> 512, 4, 4
        out = F.avg_pool2d(out, out.shape[2])  # -> 512, 1, 1
        feature = out.view(batch_size, -1)  # (batch_size, feature_dim)
        
        if self.use_cfc:
            # Process features through CfC
            # Reshape to (batch, 1, features) - single timestep per image
            # Alternative: could process batch as sequence (batch_size, seq_len, features)
            feature_seq = feature.unsqueeze(1)  # (batch_size, 1, feature_dim)
            
            if hx is not None:
                cfc_out, hx_new = self.cfc(feature_seq, hx)
            else:
                cfc_out, hx_new = self.cfc(feature_seq)
            
            # Take output from the single timestep
            cfc_feature = cfc_out.squeeze(1)  # (batch_size, cfc_hidden_size)
            
            # Store hidden state for potential use
            self.hidden_state = hx_new
            
            if returnt == 'features':
                return cfc_feature
            
            out = self.classifier(cfc_feature)
        else:
            # Standard feedforward path
            if returnt == 'features':
                return feature
            
            out = self.classifier(feature)

        if returnt == 'out':
            return out
        elif returnt == 'all':
            if self.use_cfc:
                return (out, cfc_feature)
            else:
                return (out, feature)

        raise NotImplementedError("Unknown return type")


def CFCresnet18(nclasses: int, nf: int=64) -> ResNet:
    """
    Instantiates a ResNet18 network with CfC.
    :param nclasses: number of output classes
    :param nf: number of filters
    :return: ResNet network
    """
    return ResNet(BasicBlock, [2, 2, 2, 2], nclasses, nf)


@register_backbone('cnn-cfc')
def cnn_cfc(num_classes: int, nf: int = 64, use_cfc: bool = True, cfc_hidden_size: int = 256):
    """ResNet18 with CfC temporal processing."""
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes, nf, use_cfc=use_cfc, cfc_hidden_size=cfc_hidden_size)
