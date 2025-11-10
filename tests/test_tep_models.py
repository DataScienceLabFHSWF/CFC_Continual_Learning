#!/usr/bin/env python3
"""
Quick test of TEP components before running full experiment.
"""

import torch
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from mammoth.backbone.TEPcfc import TEPCfC, TEPLSTM


def test_tep_models():
    """Test TEP-CfC and TEP-LSTM models."""
    print("="*60)
    print("Testing TEP Models")
    print("="*60)
    
    batch_size = 16
    seq_len = 50
    input_size = 52
    num_classes = 22
    
    # Create dummy input
    x = torch.randn(batch_size, seq_len, input_size)
    print(f"\nInput shape: {x.shape}")
    
    # Test CfC with NCP wiring
    print("\n1. Testing CfC with NCP wiring...")
    model_ncp = TEPCfC(input_size=input_size, num_classes=num_classes, 
                       hidden_size=128, use_ncp_wiring=True)
    print(f"   Parameters: {model_ncp.get_params():,}")
    
    try:
        output_ncp = model_ncp(x)
        print(f"   Output shape: {output_ncp.shape}")
        assert output_ncp.shape == (batch_size, num_classes), "Wrong output shape!"
        print("   ✅ CfC-NCP forward pass successful")
    except Exception as e:
        print(f"   ❌ CfC-NCP failed: {e}")
        return False
    
    # Test CfC fully-connected
    print("\n2. Testing CfC fully-connected...")
    model_full = TEPCfC(input_size=input_size, num_classes=num_classes,
                        hidden_size=128, use_ncp_wiring=False)
    print(f"   Parameters: {model_full.get_params():,}")
    
    try:
        output_full = model_full(x)
        print(f"   Output shape: {output_full.shape}")
        assert output_full.shape == (batch_size, num_classes), "Wrong output shape!"
        print("   ✅ CfC-Full forward pass successful")
    except Exception as e:
        print(f"   ❌ CfC-Full failed: {e}")
        return False
    
    # Test LSTM
    print("\n3. Testing LSTM baseline...")
    model_lstm = TEPLSTM(input_size=input_size, num_classes=num_classes,
                         hidden_size=128)
    print(f"   Parameters: {model_lstm.get_params():,}")
    
    try:
        output_lstm = model_lstm(x)
        print(f"   Output shape: {output_lstm.shape}")
        assert output_lstm.shape == (batch_size, num_classes), "Wrong output shape!"
        print("   ✅ LSTM forward pass successful")
    except Exception as e:
        print(f"   ❌ LSTM failed: {e}")
        return False
    
    # Test gradient flow
    print("\n4. Testing gradient flow...")
    try:
        loss = output_ncp.sum()
        loss.backward()
        
        # Check if gradients exist
        has_grads = any(p.grad is not None for p in model_ncp.parameters())
        assert has_grads, "No gradients computed!"
        print("   ✅ Gradients computed successfully")
    except Exception as e:
        print(f"   ❌ Gradient test failed: {e}")
        return False
    
    # Test hidden state reset
    print("\n5. Testing hidden state management...")
    try:
        model_ncp.reset_hidden()
        output_after_reset = model_ncp(x)
        assert output_after_reset.shape == (batch_size, num_classes)
        print("   ✅ Hidden state reset successful")
    except Exception as e:
        print(f"   ❌ Hidden state test failed: {e}")
        return False
    
    print("\n" + "="*60)
    print("✅ All TEP model tests passed!")
    print("="*60)
    
    # Parameter comparison
    print("\nParameter Comparison:")
    print(f"  CfC-NCP:  {model_ncp.get_params():,}")
    print(f"  CfC-Full: {model_full.get_params():,}")
    print(f"  LSTM:     {model_lstm.get_params():,}")
    
    ncp_reduction = (1 - model_ncp.get_params() / model_full.get_params()) * 100
    print(f"\nNCP parameter reduction: {ncp_reduction:.1f}%")
    
    return True


if __name__ == '__main__':
    success = test_tep_models()
    sys.exit(0 if success else 1)
