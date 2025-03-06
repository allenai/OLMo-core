# import pickle
# with open('/data/input/amanr/mup/OLMo-core/test.bsh', 'rb') as f:
#     base_shapes = pickle.load(f)
    
# print(type(base_shapes))
# print(base_shapes.keys() if isinstance(base_shapes, dict) else "Not a dictionary")

# test_base_shapes.py
import torch
from mup import load_base_shapes

try:
    # Method 1: Try loading with mup directly
    base_shapes = load_base_shapes('/data/input/amanr/mup/OLMo-core/test.bsh')
    print("Successfully loaded with mup.load_base_shapes")
    print(f"Type: {type(base_shapes)}")
    print(f"Content sample: {str(base_shapes)[:200]}...")
except Exception as e:
    print(f"Error with mup.load_base_shapes: {e}")
    
    # Method 2: Try examining the file manually
    try:
        with open('/data/input/amanr/mup/OLMo-core/test.bsh', 'r') as f:
            content = f.read(200)  # Read first 200 chars
            print("\nFile content (first 200 chars):")
            print(content)
    except Exception as e2:
        print(f"Error reading file: {e2}")