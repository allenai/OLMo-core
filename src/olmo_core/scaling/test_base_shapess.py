# import torch
# from mup import load_base_shapes
# try:
#     base_shapes = load_base_shapes('/data/input/amanr/mup/OLMo-core/test.bsh')
#     print("Successfully loaded with mup.load_base_shapes")
#     print(f"Type: {type(base_shapes)}")
#     print(f"Content sample: {str(base_shapes)[:200]}...")
# except Exception as e:
#     print(f"Error with mup.load_base_shapes: {e}")

#!/usr/bin/env python
"""
Validator script for muP base shapes files.
Tests if a base shapes file can be properly loaded and applied to a model.
"""

import os
import sys
import torch
import argparse
from olmo_core.config import DType
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.data import TokenizerConfig
from olmo_core.nn.transformer import (
    TransformerConfig,
    TransformerDataParallelConfig,
    TransformerDataParallelWrappingStrategy,
)

def validate_base_shapes(base_shapes_path, d_model=768, verbose=True):
    """
    Validate that a base shapes file can be properly loaded and applied.
    
    Args:
        base_shapes_path: Path to the base shapes file
        d_model: Model dimension to use for test model
        verbose: Whether to print detailed information
        
    Returns:
        is_valid: Boolean indicating if the file is valid
    """
    # Import muP functions
    import mup
    from mup import set_base_shapes, get_shapes
    from mup import load_base_shapes  # This is the correct function, not mup.load
    
    # Set up environment variables if needed
    if 'RANK' not in os.environ:
        os.environ['RANK'] = '0'
        os.environ['WORLD_SIZE'] = '1'
        os.environ['LOCAL_RANK'] = '0'
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '29500'
    
    # Check if file exists
    if not os.path.exists(base_shapes_path):
        print(f"ERROR: File does not exist: {base_shapes_path}")
        return False
    
    print(f"Testing base shapes file: {base_shapes_path}")
    print(f"File size: {os.path.getsize(base_shapes_path)} bytes")
    
    # Step 1: Try to load the base shapes
    try:
        print("Loading base shapes...")
        # Use the correct function: load_base_shapes, not load
        base_shapes = load_base_shapes(base_shapes_path)
        print(f"✓ Successfully loaded base shapes with {len(base_shapes)} entries")
        
        if verbose:
            print("\nSample entries from base shapes:")
            for i, (key, shape) in enumerate(base_shapes.items()):
                print(f"  {key}: {shape}")
                if i >= 4:  # Just show first 5
                    remaining = len(base_shapes) - 5
                    print(f"  ... and {remaining} more entries")
                    break
    except Exception as e:
        print(f"ERROR: Failed to load base shapes: {e}")
        return False
    
    # Step 2: Create a test model
    print("\nCreating test model...")
    try:
        # Configure the model
        tokenizer_config = TokenizerConfig.dolma2()
        config = TransformerConfig.olmo2_190M(
            mup=True,
            vocab_size=tokenizer_config.padded_vocab_size(),
            compile=True,
            d_model=d_model,
            dp_config=TransformerDataParallelConfig(
                name=DataParallelType.fsdp,
                param_dtype=DType.bfloat16,
                reduce_dtype=DType.float32,
                wrapping_strategy=TransformerDataParallelWrappingStrategy.blocks,
            ),
        )
        # config.mup = True
        
        # Create the model (use CPU for testing)
        device = torch.device('cuda')
        model = config.build(device=device)
        print(f"✓ Successfully created test model with d_model={d_model}")
        
        # Count parameters
        param_count = sum(p.numel() for p in model.parameters())
        print(f"  Model has {param_count:,} parameters")
    except Exception as e:
        print(f"ERROR: Failed to create test model: {e}")
        return False
    
    # Step 3: Apply base shapes to the model
    print("\nApplying base shapes to model...")
    try:
        set_base_shapes(model, base_shapes)
        print("✓ Successfully applied base shapes to model")
    except Exception as e:
        print(f"ERROR: Failed to apply base shapes: {e}")
        return False
    
    # Step 4: Check if all parameters have infshape attribute
    print("\nChecking parameters for infshape attribute...")
    missing_infshape = []
    total_params = 0
    
    for name, param in model.named_parameters():
        total_params += 1
        if not hasattr(param, 'infshape'):
            missing_infshape.append((name, param.shape))
    
    if missing_infshape:
        print(f"✗ ISSUE: {len(missing_infshape)}/{total_params} parameters missing infshape attribute")
        print("\nSample parameters missing infshape:")
        for i, (name, shape) in enumerate(missing_infshape[:5]):
            print(f"  - {name}: {shape}")
        if len(missing_infshape) > 5:
            print(f"  ... and {len(missing_infshape) - 5} more")
        return False
    else:
        print(f"✓ All {total_params} parameters have infshape attribute")
    
    # Step 5: Try to create a muP optimizer
    print("\nTesting muP optimizer creation...")
    try:
        optimizer = mup.optim.MuSGD(model.parameters(), lr=0.1)
        print("✓ Successfully created muP optimizer")
    except Exception as e:
        print(f"ERROR: Failed to create muP optimizer: {e}")
        return False
    
    print("\n==== SUMMARY ====")
    print(f"Base shapes file '{base_shapes_path}' is valid for use with muP!")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate muP base shapes file")
    parser.add_argument("base_shapes_path", help="Path to the base shapes file to validate")
    parser.add_argument("--d_model", type=int, default=768, help="Model dimension for test model")
    parser.add_argument("--verbose", action="store_true", help="Print verbose output")
    
    args = parser.parse_args()
    
    if validate_base_shapes(args.base_shapes_path, args.d_model, args.verbose):
        sys.exit(0)
    else:
        sys.exit(1)
    