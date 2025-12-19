
import sys
import os
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'mammoth'))

try:
    import mammoth.models.hope as hope
    print("Import successful")
    parser = hope.get_parser()
    print("Parser retrieved")
    print("hope_lr in parser:", any(action.dest == 'hope_lr' for action in parser._actions))
except Exception as e:
    print(f"Error: {e}")
