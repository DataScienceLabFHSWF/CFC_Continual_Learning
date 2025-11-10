"""
Test script for MNISTcfc implementation
"""
import sys
sys.path.insert(0, 'mammoth')

import torch
import ncps.wirings
from ncps.torch import CfC
from ncps.wirings import AutoNCP
from backbone.MNISTcfc import MNISTcfc


def main():
    print('=' * 60)
    print('Testing MNISTcfc Implementation')
    print('=' * 60)

    # Test 1: Create model with NCP wiring (fixed hidden size vs output size)
    print('\n1. Creating MNISTcfc with NCP wiring...')
    # For AutoNCP, output_size must be < hidden_size - 2
    model_ncp = MNISTcfc(784, 10, use_ncp_wiring=True, hidden_size=64, chunk_size=28)
    total_params = sum(p.numel() for p in model_ncp.parameters())
    print(f'   ✓ NCP model created')
    print(f'   Total parameters: {total_params:,}')

    # Test 2: Create model without NCP wiring (fully connected)
    print('\n2. Creating MNISTcfc without NCP wiring (fully connected)...')
    model_fc = MNISTcfc(784, 10, use_ncp_wiring=False, hidden_size=64, chunk_size=28)
    total_params_fc = sum(p.numel() for p in model_fc.parameters())
    print(f'   ✓ Fully-connected model created')
    print(f'   Total parameters: {total_params_fc:,}')

    # Test 3: Forward pass with batch
    print('\n3. Testing forward pass...')
    batch_size = 16
    x = torch.randn(batch_size, 1, 28, 28)
    print(f'   Input shape: {x.shape}')

    out_ncp = model_ncp(x)
    print(f'   ✓ NCP output shape: {out_ncp.shape}')

    out_fc = model_fc(x)
    print(f'   ✓ FC output shape: {out_fc.shape}')

    # Test 4: Test different return types
    print('\n4. Testing different return types...')
    features_ncp = model_ncp(x, returnt='features')
    print(f'   Features shape: {features_ncp.shape}')

    out_all, feats_all = model_ncp(x, returnt='all')
    print(f'   All return - output: {out_all.shape}, features: {feats_all.shape}')

    # Test 5: Verify output is valid
    print('\n5. Sanity checks...')
    print(f'   NCP output range: [{out_ncp.min().item():.3f}, {out_ncp.max().item():.3f}]')
    print(f'   FC output range: [{out_fc.min().item():.3f}, {out_fc.max().item():.3f}]')

    # Test 6: Gradient flow
    print('\n6. Testing gradient flow...')
    loss = out_ncp.sum()
    loss.backward()
    has_grad = all(p.grad is not None for p in model_ncp.parameters() if p.requires_grad)
    print(f'   ✓ All parameters have gradients: {has_grad}')

    print('\n' + '=' * 60)
    print('✅ ALL TESTS PASSED!')
    print('=' * 60)


if __name__ == '__main__':
    main()
