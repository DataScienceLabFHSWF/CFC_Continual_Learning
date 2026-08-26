# Copyright 2024-present
# CfC / MLP backbones for C-MAPSS (sliding windows of 24-dim sensor readings).

import torch
import torch.nn as nn

from backbone import MammothBackbone, register_backbone

try:
    from ncps.torch import CfC
    from ncps.wirings import AutoNCP
    NCPS_AVAILABLE = True
except ImportError:
    NCPS_AVAILABLE = False


class CMAPSSMLP(MammothBackbone):
    """MLP baseline: flattens the (window, features) sequence."""

    def __init__(self, input_size=24, window_size=30, num_classes=12, hidden_size=256):
        super().__init__()
        self.flatten_size = input_size * window_size
        self.fc1 = nn.Linear(self.flatten_size, hidden_size)
        self.relu = nn.ReLU()
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x, returnt='out'):
        x = x.reshape(x.size(0), -1)
        feats = self.relu(self.fc1(x))
        if returnt == 'features':
            return feats
        out = self.classifier(feats)
        if returnt == 'out':
            return out
        elif returnt in ('both', 'all'):
            return out, feats
        raise ValueError(f"Unknown returnt value: {returnt}")


class CMAPSSCfC(MammothBackbone):
    """CfC (AutoNCP) backbone for C-MAPSS sliding windows."""

    def __init__(self, input_size=24, num_classes=12, hidden_size=128, use_ncp_wiring=True):
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
            x = x.unsqueeze(1)
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


@register_backbone('cmapssmlp')
def cmapssmlp(input_size: int = 24, window_size: int = 30, num_classes: int = 12, hidden_size: int = 256):
    return CMAPSSMLP(input_size=input_size, window_size=window_size,
                      num_classes=num_classes, hidden_size=hidden_size)


@register_backbone('cmapsscfc')
def cmapsscfc(input_size: int = 24, num_classes: int = 12, hidden_size: int = 128, use_ncp_wiring: bool = True):
    return CMAPSSCfC(input_size=input_size, num_classes=num_classes,
                      hidden_size=hidden_size, use_ncp_wiring=use_ncp_wiring)
