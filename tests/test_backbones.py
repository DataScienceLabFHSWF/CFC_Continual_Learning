#!/usr/bin/env python3
"""Quick test to verify all CfC backbones can be instantiated and do forward passes."""

import sys
sys.path.insert(0, 'mammoth')

import torch

def test_backbone(name, backbone_fn, input_shape, expected_output_size):
    """Test a backbone can be created and does a forward pass."""
    print(f"\nTesting {name}...")
    try:
        # Create backbone
        backbone = backbone_fn()
        backbone.eval()
        
        # Create dummy input
        x = torch.randn(2, *input_shape)  # batch_size=2
        
        # Forward pass
        with torch.no_grad():
            out = backbone(x)
        
        # Check output shape
        if out.shape[1] == expected_output_size:
            print(f"  ✓ {name} works! Output shape: {out.shape}")
            return True
        else:
            print(f"  ✗ {name} output shape mismatch: got {out.shape}, expected (2, {expected_output_size})")
            return False
            
    except Exception as e:
        print(f"  ✗ {name} failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("="*60)
    print("CfC Backbone Validation Tests")
    print("="*60)
    
    results = {}
    
    # Test 1: MNIST CfC
    from backbone.MNISTcfc import mnistcfc
    results['mnistcfc'] = test_backbone(
        'mnistcfc',
        lambda: mnistcfc(input_size=784, output_size=10, hidden_size=128, use_ncp_wiring=True, chunk_size=28),
        (784,),
        10
    )
    
    # Test 2: CNN-CfC
    from backbone.cnn_cfc import cnn_cfc
    results['cnn-cfc'] = test_backbone(
        'cnn-cfc',
        lambda: cnn_cfc(num_classes=10, nf=64, cfc_hidden_size=256, use_cfc=True),
        (3, 32, 32),
        10
    )
    
    # Test 3: TEP CfC
    from backbone.TEPcfc import tepcfc
    results['tepcfc'] = test_backbone(
        'tepcfc',
        lambda: tepcfc(num_features=52, num_classes=22, hidden_size=256, use_ncp_wiring=True),
        (50, 52),  # (seq_len, features)
        22
    )
    
    # Test 4: TEP LSTM
    from backbone.TEPcfc import teplstm
    results['teplstm'] = test_backbone(
        'teplstm',
        lambda: teplstm(num_features=52, num_classes=22, hidden_size=256),
        (50, 52),  # (seq_len, features)
        22
    )
    
    # Summary
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    
    for name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {name}")
    
    total = len(results)
    passed = sum(results.values())
    print(f"\nTotal: {passed}/{total} passed")
    
    return 0 if passed == total else 1

if __name__ == '__main__':
    sys.exit(main())
