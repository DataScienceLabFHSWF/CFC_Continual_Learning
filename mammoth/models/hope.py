import torch
from models.utils.continual_model import ContinualModel
from utils.args import ArgumentParser
from backbone.hope import HOPEBackbone

class Hope(ContinualModel):
    NAME = 'hope'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    @staticmethod
    def get_parser(parser) -> ArgumentParser:
        parser.add_argument('--hope_lr', type=float, default=0.001, help='Learning rate for internal updates')
        return parser

    def __init__(self, backbone, loss, args, transform, dataset=None):
        super(Hope, self).__init__(backbone, loss, args, transform, dataset)
        # Ensure backbone is HOPE
        if not isinstance(self.net, HOPEBackbone):
            print("Warning: HOPE model should be used with HOPE backbone. Current backbone:", type(self.net))

    def observe(self, inputs, labels, not_aug_inputs, epoch=None):
        self.opt.zero_grad()
        
        # 1. Forward pass to get features and logits
        outputs, features = self.net(inputs, returnt='all')
        
        loss = self.loss(outputs, labels)
        
        # 2. Backward pass (compute main gradients)
        # We need to retain grad on features to use it as teach signal
        features.retain_grad()
        loss.backward()
        
        # 3. Trigger Internal Updates (Nested Learning)
        # This modifies weights in-place, so it must happen AFTER loss.backward()
        feature_grads = features.grad
        if feature_grads is not None and isinstance(self.net, HOPEBackbone):
            teach_signal = feature_grads.unsqueeze(1)
            self.net(inputs, teach_signal=teach_signal)
            
        # 4. Standard Optimization Step
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
        self.opt.step()
        
        return loss.item()
