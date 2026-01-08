import torch
from models.utils.continual_model import ContinualModel
from utils.args import ArgumentParser, add_rehearsal_args
from utils.buffer import Buffer
from backbone.hope import HOPEBackbone

class Hope(ContinualModel):
    NAME = 'hope'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    @staticmethod
    def get_parser(parser) -> ArgumentParser:
        parser.add_argument('--hope_lr', type=float, default=0.001, help='Learning rate for internal updates')
        parser.add_argument('--alpha', type=float, default=0.5, help='Memory update rate')
        parser.add_argument('--beta', type=float, default=0.5, help='Teacher signal weight')
        
        # Backbone args exposed via Model
        parser.add_argument('--hidden_dim', type=int, default=128, help='HOPE Hidden Dims')
        parser.add_argument('--key_dim', type=int, default=64, help='HOPE Key Dims')
        parser.add_argument('--memory_layers', type=int, default=3, help='HOPE Memory Layers')
        parser.add_argument('--surprise_threshold', type=float, default=0.15, help='Update threshold')
        parser.add_argument('--momentum', type=float, default=0.9, help='Update momentum')
        
        # Add Replay Buffer Args (optional)
        add_rehearsal_args(parser)
        return parser

    def __init__(self, backbone, loss, args, transform, dataset=None):
        super(Hope, self).__init__(backbone, loss, args, transform, dataset)
        # Ensure backbone is HOPE
        if not isinstance(self.net, HOPEBackbone):
            print("Warning: HOPE model should be used with HOPE backbone. Current backbone:", type(self.net))
        
        # Optional Replay Buffer
        self.buffer = Buffer(self.args.buffer_size, self.device) if self.args.buffer_size > 0 else None

    def observe(self, inputs, labels, not_aug_inputs, epoch=None):
        real_batch_size = inputs.shape[0]
        self.opt.zero_grad()
        
        # 1. OPTIONAL: Experience Replay
        if self.buffer is not None and not self.buffer.is_empty():
            buf_inputs, buf_labels = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform, device=self.device)
            inputs = torch.cat((inputs, buf_inputs))
            labels = torch.cat((labels, buf_labels))

        # 2. Forward pass to get features and logits
        outputs, features = self.net(inputs, returnt='all')
        loss = self.loss(outputs, labels)
        
        # 3. Backward pass (compute main gradients)
        # We need to retain grad on features to use it as teach signal
        features.retain_grad()
        loss.backward()
        
        # 4. Trigger Internal Updates (Nested Learning)
        # The key idea: Replay updates the weights via standard SGD, 
        # but SHOULD it also trigger "Plastic" memory updates?
        # A: Yes, providing gradients for old classes during the plastic phase might save tokens.
        feature_grads = features.grad
        if feature_grads is not None and isinstance(self.net, HOPEBackbone):
            teach_signal = feature_grads.unsqueeze(1)
            self.net(inputs, teach_signal=teach_signal)
            
        # 5. Standard Optimization Step
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
        self.opt.step()
        
        # 6. Update Buffer
        if self.buffer is not None:
             self.buffer.add_data(examples=not_aug_inputs, labels=labels[:real_batch_size])
        
        return loss.item()

    def end_task(self, dataset):
        """
        Called at the end of each task. Triggers memory consolidation.
        """
        if hasattr(self.net, 'consolidate'):
             print(f"Ending task {dataset.current_task}. Triggering HOPE consolidation.")
             self.net.consolidate()
