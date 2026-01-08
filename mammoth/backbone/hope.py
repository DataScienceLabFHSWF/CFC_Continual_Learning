import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List, Set
from backbone import MammothBackbone, register_backbone

# --- Titan Memory ---

class TitanMemory(nn.Module):
    """
    Simplified TITAN-style associative memory.
    A neural memory module that can be updated at test time.
    """
    def __init__(self, dim: int, hidden_multiplier: int = 4, layers: int = 2):
        super().__init__()
        self.dim = dim
        hidden = dim * hidden_multiplier
        
        blocks = []
        for layer_idx in range(layers - 1):
            in_dim = dim if layer_idx == 0 else hidden
            blocks.extend([
                nn.Linear(in_dim, hidden),
                nn.GELU()
            ])
        
        # Final layer
        in_dim = hidden if layers > 1 else dim
        blocks.append(nn.Linear(in_dim, dim))
        
        self.net = nn.Sequential(*blocks)
        self.norm = nn.LayerNorm(dim)
        self.grad_clip = 1.0

    def forward(self, query: torch.Tensor) -> torch.Tensor:
        # Query: [Batch, Seq, Dim]
        out = self.net(query)
        return self.norm(out)

    def update(self, key: torch.Tensor, value: torch.Tensor, error_signal: Optional[torch.Tensor] = None, lr: float = 1e-3, surprise_threshold: float = 1e-6):
        """
        Update the memory parameters based on key-value pair and error signal.
        This is a simplified version of Titan's update.
        """
        # In a real Titan implementation, this would update the 'memory' weights.
        # Here we simulate it by updating the MLP weights using the error signal.
        # This requires 'torch.enable_grad()' context if called during inference/forward.
        
        if error_signal is None:
            return

        # STABILITY FIX: Gating threshold to skip updates if signal is noise
        if error_signal.norm() < surprise_threshold:
            return

        # Simple SGD update on the weights
        # We need to compute gradients of the output w.r.t. weights given the error signal.
        # But we don't have the output here.
        # We assume 'update' is called after a forward pass or we re-run forward.
        
        with torch.enable_grad():
            pred = self.net(key)
            
            # STABILITY FIX: Scale signal 
            teach_scale = 0.1
            error_signal = error_signal * teach_scale
            
            # Loss: minimize distance to value (or minimize error_signal if it's a gradient)
            # If error_signal is the gradient of the loss w.r.t. output, we can backward it.
            
            # Assuming error_signal is dL/dOutput
            # We want to update weights W: W_new = W - lr * dL/dW
            # dL/dW = dL/dOutput * dOutput/dW
            
            # We can use torch.autograd.backward
            if error_signal.shape != pred.shape:
                error_signal = error_signal.expand_as(pred)
            
            # Safety check for NaNs
            if torch.isnan(error_signal).any() or torch.isinf(error_signal).any():
                return

            # Check for NaNs in prediction
            if torch.isnan(pred).any() or torch.isinf(pred).any():
                return

            torch.autograd.backward(pred, grad_tensors=error_signal, inputs=list(self.net.parameters()), retain_graph=False)
            
            # Clip gradients
            # STABILITY FIX: Ensure strict clipping
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.grad_clip)
            
            with torch.no_grad():
                for param in self.net.parameters():
                    if param.grad is not None:
                        param.add_(param.grad, alpha=-lr)
                        param.grad.zero_()

# --- CMS (Contextual Memory System) ---

class CMS(nn.Module):
    """
    Contextual Memory System with multiple timescales.
    """
    def __init__(self, dim: int, levels: List[Dict]):
        super().__init__()
        self.dim = dim
        self.levels = levels # List of dicts: {'name': 'fast', 'period': 1}, etc.
        
        self.blocks = nn.ModuleList()
        self.block_map = {}
        
        for level in levels:
            # Each level has a small MLP or memory block
            block = TitanMemory(dim, hidden_multiplier=2, layers=1)
            self.blocks.append(block)
            self.block_map[level['name']] = block
            
        self.step_counter = 0

    def forward(self, x: torch.Tensor, return_intermediates: bool = False) -> torch.Tensor:
        # x: [B, S, D]
        current = x
        inputs = {}
        outputs = {}
        
        for i, level in enumerate(self.levels):
            name = level['name']
            block = self.blocks[i]
            
            inputs[name] = current
            out = block(current)
            outputs[name] = out
            
            # Residual connection
            current = current + out
            
        if return_intermediates:
            return current, inputs, outputs
        return current

    def maybe_update(self, inputs, outputs, teach_signal, lr=1e-3, surprise_threshold=1e-6):
        """
        Check update periods and update blocks if needed.
        """
        self.step_counter += 1
        
        if teach_signal is not None:
            # DEBUG LOGGING (Sample approx every 100 steps)
            if self.step_counter % 100 == 0:
                log_norm = teach_signal.norm().item()
                # print(f"[HOPE] Step {self.step_counter} | Signal Norm: {log_norm:.4f} | Threshold: {surprise_threshold}")

            # STABILITY FIX: Gating threshold
            if teach_signal.norm() < surprise_threshold:
                return

        for i, level in enumerate(self.levels):
            name = level['name']
            period = level['period']
            
            # ALTERNATIVE FIX: Freeze Slow Memories during training (Consolidation Strategy)
            if period >= 100:
                continue
            
            if self.step_counter % period == 0:
                block = self.blocks[i]
                inp = inputs[name].detach()
                
                with torch.enable_grad():
                    pred = block.net(inp) 
                    if teach_signal is not None:
                        # Very simplified: try to align output with teach_signal
                        # If teach_signal is dL/dFeatures, we can use it directly?
                        # Or treat it as a target?
                        # Let's assume teach_signal is a gradient.
                        
                        if teach_signal.shape != pred.shape:
                            teach_signal = teach_signal.expand_as(pred)
                        
                        # STABILITY FIX: Scale signal
                        teach_signal = teach_signal * 0.1

                        # Safety check for NaNs
                        if torch.isnan(teach_signal).any() or torch.isinf(teach_signal).any():
                            continue

                        # Check for NaNs in prediction
                        if torch.isnan(pred).any() or torch.isinf(pred).any():
                            continue
                            
                        # DEBUG LOGGING (Sample approx every 100 steps)
                        if self.step_counter % 100 == 0 and i == 0:
                             print(f"[HOPE] Step {self.step_counter} | Level: {name} | Signal Norm: {teach_signal.norm().item():.4f} | Surprise Thresh: {surprise_threshold}")

                        torch.autograd.backward(pred, grad_tensors=teach_signal, inputs=list(block.parameters()), retain_graph=False)
                        
                        # Clip gradients
                        torch.nn.utils.clip_grad_norm_(block.parameters(), 1.0)

                        with torch.no_grad():
                            for param in block.parameters():
                                if param.grad is not None:
                                    param.add_(param.grad, alpha=-lr)
                                    param.grad.zero_()

    def consolidate(self):
        """
        Consolidate memory: Copy Fast weights to Slow weights.
        """
        # Assume levels are [Fast, Mid, Slow]
        # Copy Fast (idx 0) to Slow (idx -1)
        fast_block = self.blocks[0]
        slow_block = self.blocks[-1]
        
        # Simple weight copy (strict consolidation)
        # In a real hippocampus model, this might be distillation.
        # Here we assume the architecture is identical.
        if len(self.levels) > 1:
             print(f"[HOPE] Consolidating memory: Fast ({self.levels[0]['name']}) -> Slow ({self.levels[-1]['name']})")
             # hard copy
             slow_block.load_state_dict(fast_block.state_dict())

# --- Self Modifier ---

class SelfModifier(nn.Module):
    def __init__(self, dim: int, hidden_multiplier: int = 2):
        super().__init__()
        hidden = dim * hidden_multiplier
        # Input: key, value, error_signal. All dim. -> 3*dim
        self.net = nn.Sequential(
            nn.Linear(dim * 3, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim) # Output delta
        )

    def forward(self, key: torch.Tensor, value: torch.Tensor, error_signal: torch.Tensor) -> torch.Tensor:
        # Concatenate along last dim
        inp = torch.cat([key, value, error_signal], dim=-1)
        return self.net(inp)

# --- HOPE Block ---

class HOPEBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int = 4, surprise_threshold: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.surprise_threshold = surprise_threshold
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        
        self.titan = TitanMemory(dim)
        
        # Define CMS levels
        cms_levels = [
            {'name': 'fast', 'period': 1},
            {'name': 'mid', 'period': 5},
            {'name': 'slow', 'period': 100} 
        ]
        self.cms = CMS(dim, cms_levels)
        
        self.self_mod = SelfModifier(dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, teach_signal: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: [B, S, D]
        
        # 1. Attention
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)
        
        # 2. Titan Memory
        mem_out = self.titan(x)
        combined = x + mem_out
        
        # 3. CMS
        cms_out, cms_inputs, cms_outputs = self.cms(combined, return_intermediates=True)
        out = self.norm2(cms_out)
        
        # 4. Updates (if teach_signal is present)
        if teach_signal is not None and self.training:
            # STABILITY FIX: Pre-normalize teach signal similar to Titan implementation
            # This prevents exploding updates to the memory modules
            with torch.no_grad():
                teach_norm = teach_signal.norm(dim=-1, keepdim=True)
                teach_clip = 1.0
                scale = torch.clamp(teach_norm / teach_clip, min=1.0)
                teach_signal = teach_signal / scale
            
            # Update Titan
            self.titan.update(key=x, value=x, error_signal=teach_signal, surprise_threshold=self.surprise_threshold)
            
            # Update CMS
            self.cms.maybe_update(cms_inputs, cms_outputs, teach_signal, surprise_threshold=self.surprise_threshold)
            
        return out

    def consolidate(self):
        self.cms.consolidate()

# --- HOPE Backbone ---

class HOPEBackbone(MammothBackbone):
    def __init__(self, input_size: int, input_channels: int, num_classes: int, dim: int = 128, depth: int = 2, heads: int = 4, surprise_threshold: float = 1e-6):
        super(HOPEBackbone, self).__init__()
        
        self.input_size = input_size # e.g. 32 for CIFAR
        self.patch_size = 4
        self.dim = dim
        self.num_classes = num_classes
        
        # Patch Embedding
        num_patches = (input_size // self.patch_size) ** 2
        self.patch_embed = nn.Conv2d(input_channels, dim, kernel_size=self.patch_size, stride=self.patch_size)
        
        # Positional Embedding
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, dim) * 0.02)
        
        # HOPE Blocks
        self.blocks = nn.ModuleList([
            HOPEBlock(dim, heads, surprise_threshold=surprise_threshold) for _ in range(depth)
        ])
        
        self.out_dim = dim # For Mammoth compatibility
        self.classifier = nn.Linear(dim, num_classes)

    def consolidate(self):
        """
        Trigger consolidation in all HOPE blocks.
        """
        for block in self.blocks:
             block.consolidate()

    def forward(self, x: torch.Tensor, returnt: str = 'out', teach_signal: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: [B, C, H, W]
        
        # Patch Embed
        x = self.patch_embed(x) # [B, D, H', W']
        x = x.flatten(2).transpose(1, 2) # [B, S, D]
        
        # Add Pos Embed
        x = x + self.pos_embed
        
        # Blocks
        for block in self.blocks:
            x = block(x, teach_signal=teach_signal)
            
        # Pooling (Mean)
        features = x.mean(dim=1) # [B, D]
        
        if returnt == 'features':
            return features
            
        logits = self.classifier(features)
        
        if returnt == 'out':
            return logits
        elif returnt == 'all':
            return logits, features
        return logits

@register_backbone("hope")
def hope(input_size: int = None, input_channels: int = None, num_classes: int = None, dim: int = 128, depth: int = 2, heads: int = 4, surprise_threshold: float = 1e-6) -> MammothBackbone:
    """
    Instantiates the HOPE backbone.
    """
    if input_size is None or input_channels is None or num_classes is None:
        raise ValueError("input_size, input_channels and num_classes must be provided")
    return HOPEBackbone(input_size, input_channels, num_classes, dim=dim, depth=depth, heads=heads, surprise_threshold=surprise_threshold)
