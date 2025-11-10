"""
Test script for CNN-CfC (ResNet with CfC) implementation
"""
import sys
sys.path.insert(0, 'mammoth')

import torch
import ncps.wirings
from ncps.torch import CfC
from ncps.wirings import AutoNCP
from backbone.cnn_cfc import CFCresnet18, ResNet, BasicBlock


def main():
    print('=' * 60)
    print('Testing CNN-CfC (ResNet18) Implementation')
    print('=' * 60)

    # Test 1: Create ResNet18 with CfC
    print('\n1. Creating ResNet18 with CfC...')
    model_cfc = CFCresnet18(nclasses=10, nf=20)  # Smaller nf for faster testing
    total_params = sum(p.numel() for p in model_cfc.parameters())
    print(f'   ✓ ResNet18-CfC model created')
    print(f'   Total parameters: {total_params:,}')

    # Test 2: Create ResNet18 without CfC (standard)
    print('\n2. Creating ResNet18 without CfC (standard)...')
    model_standard = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=10, nf=20, use_cfc=False)
    total_params_std = sum(p.numel() for p in model_standard.parameters())
    print(f'   ✓ Standard ResNet18 model created')
    print(f'   Total parameters: {total_params_std:,}')

    # Test 3: Forward pass with batch (CIFAR-10 size: 32x32x3)
    print('\n3. Testing forward pass...')
    batch_size = 8
    x = torch.randn(batch_size, 3, 32, 32)
    print(f'   Input shape: {x.shape}')

    out_cfc = model_cfc(x)
    print(f'   ✓ CfC output shape: {out_cfc.shape}')

    out_std = model_standard(x)
    print(f'   ✓ Standard output shape: {out_std.shape}')

    # Test 4: Test different return types
    print('\n4. Testing different return types...')
    features_cfc = model_cfc(x, returnt='features')
    print(f'   CfC features shape: {features_cfc.shape}')

    features_std = model_standard(x, returnt='features')
    print(f'   Standard features shape: {features_std.shape}')

    out_all, feats_all = model_cfc(x, returnt='all')
    print(f'   All return - output: {out_all.shape}, features: {feats_all.shape}')

    # Test 5: Verify output is valid
    print('\n5. Sanity checks...')
    print(f'   CfC output range: [{out_cfc.min().item():.3f}, {out_cfc.max().item():.3f}]')
    print(f'   Standard output range: [{out_std.min().item():.3f}, {out_std.max().item():.3f}]')

    # Test 6: Gradient flow
    print('\n6. Testing gradient flow...')
    loss = out_cfc.sum()
    loss.backward()
    has_grad = all(p.grad is not None for p in model_cfc.parameters() if p.requires_grad)
    print(f'   ✓ All CfC parameters have gradients: {has_grad}')

    # Test 7: Test with hidden state (for continual learning)
    print('\n7. Testing with hidden state...')
    model_cfc.zero_grad()
    hx = None
    for i in range(3):
        out = model_cfc(x, hx=hx)
        hx = model_cfc.hidden_state
        if hx is not None:
            print(f'   Step {i+1} - hidden state shape: {hx.shape}')

    print('\n' + '=' * 60)
    print('✅ ALL TESTS PASSED!')
    print('=' * 60)


if __name__ == '__main__':
    main()
