# Copyright 2024-present
# CfC / MLP backbones for the Steel Plates Faults dataset (static 27-dim
# feature vectors, 6 classes). The CfC variant treats each feature vector as
# a length-1 sequence so the same recurrent cell machinery used for MNIST/
# CIFAR/TEP can be reused; there is no genuine temporal structure to exploit
# here, which is the point of including this benchmark (see paper Sec. 5).

import torch
import torch.nn as nn

from backbone import MammothBackbone, register_backbone

try:
    from ncps.torch import CfC
    from ncps.wirings import AutoNCP
    NCPS_AVAILABLE = True
except ImportError:
    NCPS_AVAILABLE = False


class SteelPlatesMLP(MammothBackbone):
    """Plain MLP baseline for Steel Plates Faults."""

    def __init__(self, input_size=27, num_classes=6, hidden_size=128):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x, returnt='out'):
        if x.dim() == 3:
            x = x.squeeze(1)
        feats = self.relu(self.fc1(x))
        if returnt == 'features':
            return feats
        out = self.classifier(feats)
        if returnt == 'out':
            return out
        elif returnt in ('both', 'all'):
            return out, feats
        raise ValueError(f"Unknown returnt value: {returnt}")


class SteelPlatesCfC(MammothBackbone):
    """CfC backbone for Steel Plates Faults, treating each row as a length-1 sequence."""

    def __init__(self, input_size=27, num_classes=6, hidden_size=64, use_ncp_wiring=True):
        super().__init__()
        if not NCPS_AVAILABLE:
            raise ImportError("ncps package not available. Please install with: poetry run pip install -e ./ncps")

        self.input_projection = nn.Linear(input_size, hidden_size)
        if use_ncp_wiring:
            wiring = AutoNCP(hidden_size, num_classes)
            self.rnn = CfC(hidden_size, wiring, batch_first=True, mixed_memory=True)
            rnn_output_size = num_classes
        else:
            self.rnn = CfC(hidden_size, hidden_size, batch_first=True, mixed_memory=True)
            rnn_output_size = hidden_size

        self.classifier = nn.Linear(rnn_output_size, num_classes)
        self.hidden_state = None

    def get_params(self) -> torch.Tensor:
        return torch.cat([p.contiguous().view(-1) for p in self.parameters()])

    def forward(self, x, returnt='out'):
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (batch, 1, input_size)
        x = self.input_projection(x)
        self.hidden_state = None
        out, self.hidden_state = self.rnn(x, self.hidden_state)
        feats = out[:, -1, :]
        if returnt == 'features':
            return feats
        logits = self.classifier(feats)
        if returnt == 'out':
            return logits
        elif returnt in ('both', 'all'):
            return logits, feats
        raise ValueError(f"Unknown returnt value: {returnt}")


@register_backbone('steelplatesmlp')
def steelplatesmlp(input_size: int = 27, num_classes: int = 6, hidden_size: int = 128):
    """MLP baseline for Steel Plates Faults."""
    return SteelPlatesMLP(input_size=input_size, num_classes=num_classes, hidden_size=hidden_size)


@register_backbone('steelplatescfc')
def steelplatescfc(input_size: int = 27, num_classes: int = 6, hidden_size: int = 64, use_ncp_wiring: bool = True):
    """CfC (AutoNCP) backbone for Steel Plates Faults."""
    return SteelPlatesCfC(input_size=input_size, num_classes=num_classes,
                           hidden_size=hidden_size, use_ncp_wiring=use_ncp_wiring)
