
import sys
import os
import torch
import unittest.mock

sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'mammoth'))

# Patch register_backbone in mammoth.backbone to do nothing or accept duplicates
import mammoth.backbone
original_register = mammoth.backbone.register_backbone

def mock_register(name):
    def decorator(cls_or_func):
        # Call original but suppress error? 
        # Actually register_backbone returns a decorator.
        # The decorator usually registers the class.
        try:
            return original_register(name)(cls_or_func)
        except ValueError:
             # Already registered, just return the class/func
             return cls_or_func
    return decorator

mammoth.backbone.register_backbone = mock_register

# Now import TEPcfc
from mammoth.backbone.TEPcfc import BaseTEPCfC

def test():
    print("Instantiating BaseTEPCfC...")
    try:
        model = BaseTEPCfC(input_size=52, num_classes=22, hidden_size=128)
    except Exception as e:
        print(f"Failed to instantiate: {e}")
        # Need ncps probably
        return

    params = model.get_params()
    print(f"get_params returned: {params}")
    print(f"Type: {type(params)}")

    try:
        z = torch.zeros_like(params)
        print("zeros_like worked")
    except Exception as e:
        print(f"zeros_like failed: {e}")

if __name__ == "__main__":
    test()
