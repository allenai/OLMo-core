import os
import sys
import json
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, List, Tuple, Any
from collections import defaultdict

sys.path.insert(0, '/data/input/amanr/OLMo-core/src')

from olmo_core.distributed.checkpoint import get_checkpoint_metadata, load_keys


def analyze_metadata_structure(checkpoint_path: str):
    metadata = get_checkpoint_metadata(checkpoint_path)
    print(f"Total state dict keys: {len(metadata.state_dict_metadata)}")

    model_keys = []
    optim_keys = []
    other_keys = []
    
    for key in metadata.state_dict_metadata.keys():
        if key.startswith('model.'):
            model_keys.append(key)
        elif key.startswith('optim.'):
            optim_keys.append(key)
        else:
            other_keys.append(key)
    
    print(f"Model keys: {len(model_keys)}")
    print(f"Optimizer keys: {len(optim_keys)}")
    print(f"Other keys: {len(other_keys)}")

    print(f"\nSample model keys:")
    for i, key in enumerate(model_keys[:20]):
        meta = metadata.state_dict_metadata[key]
        print(f"  {key}")
        if hasattr(meta, 'size'):
            print(f"Size: {meta.size}")
        if hasattr(meta, 'chunks'):
            print(f"Chunks: {len(meta.chunks)}")
    
    weight_keys = [k for k in model_keys if 'weight' in k]
    print(f"\nWeight parameter keys: {len(weight_keys)}")
    
    layer_groups = defaultdict(list)
    for key in weight_keys:
        parts = key.split('.')
        if len(parts) >= 3:
            layer_id = '.'.join(parts[1:3])  # Skip 'model.' prefix
            layer_groups[layer_id].append(key)
    
    print(f"\nLayer groups found: {len(layer_groups)}")

    for layer_id, keys in list(layer_groups.items())[:10]:
        print(f"{layer_id}: {len(keys)} keys")
        for key in keys[:5]:
            print(f"{key}")
    
    return metadata, model_keys, weight_keys


def analyze_tensor_sharding(checkpoint_path: str, sample_keys: List[str], max_samples: int = 10):
    sample_keys = sample_keys[:max_samples]
    for key in sample_keys:
        print(f"\nAnalyzing: {key}")
        
        try:
            tensor_data = next(load_keys(checkpoint_path, [key]))
            print(f"Shape: {tensor_data.shape}")
            print(f"Dtype: {tensor_data.dtype}")
            print(f"Device: {tensor_data.device}")
            
            flat_tensor = tensor_data.flatten()
            print(f"  Mean: {flat_tensor.mean().item():.6f}")
            print(f"  Std: {flat_tensor.std().item():.6f}")
            print(f"  Min: {flat_tensor.min().item():.6f}")
            print(f"  Max: {flat_tensor.max().item():.6f}")
            
            # Check for patterns that might indicate identical initialization
            if len(flat_tensor) > 1000:
                # Sample chunks to see if they're similar
                chunk_size = len(flat_tensor) // 64  # Check for 64-way sharding (matching num_nodes=64)
                if chunk_size > 0:
                    chunks = []
                    for i in range(min(64, len(flat_tensor) // chunk_size)):
                        start_idx = i * chunk_size
                        end_idx = start_idx + chunk_size
                        chunks.append(flat_tensor[start_idx:end_idx])
                    
                    if len(chunks) > 1:
                        print(f"  Analyzed {len(chunks)} chunks of size {chunk_size}")

                        for i in range(min(5, len(chunks))):
                            for j in range(i + 1, min(5, len(chunks))):
                                chunk1, chunk2 = chunks[i], chunks[j]
                                cos_sim = F.cosine_similarity(chunk1.unsqueeze(0), chunk2.unsqueeze(0)).item()
                                l2_dist = torch.norm(chunk1 - chunk2).item()
                                identical = torch.equal(chunk1, chunk2)
                                
                                print(f"    Chunk {i} vs {j}: cos_sim={cos_sim:.6f}, l2_dist={l2_dist:.6f}, identical={identical}")
                                
                                if cos_sim > 0.99:
                                    print(f"HIGH SIMILARITY DETECTED!")
                                if identical:
                                    print(f"IDENTICAL CHUNKS DETECTED!")
                                    
        except Exception as e:
            print(f"  Error: {e}")


def check_for_fsdp_sharding_pattern(checkpoint_path: str):
    """
    Check for FSDP-specific sharding patterns where the same tensor might be split across ranks.
    """
    print(f"\nFSDP SHARDING PATTERN ANALYSIS")
    print("="*50)
    
    metadata = get_checkpoint_metadata(checkpoint_path)
    
    # Look for parameters that might be sharded across ranks
    # In FSDP, parameters are typically sharded by flattening all parameters and dividing by rank
    
    # Group parameters by their base name (without any rank/shard identifiers)
    param_base_names = defaultdict(list)
    
    for key in metadata.state_dict_metadata.keys():
        if key.startswith('model.') and ('weight' in key or 'bias' in key):
            # Try to extract base parameter name
            clean_key = key
            
            # Remove common FSDP suffixes/prefixes that might indicate sharding
            # Common patterns: _flat_param, _orig_mod, etc.
            if '_flat_param' in clean_key:
                base_name = clean_key.split('_flat_param')[0]
                param_base_names[base_name].append(key)
            elif '_orig_mod' in clean_key:
                base_name = clean_key.replace('_orig_mod', '')
                param_base_names[base_name].append(key)
            else:
                param_base_names[clean_key].append(key)
    
    print(f"Parameter base names found: {len(param_base_names)}")
    
    # Look for parameters that might have multiple shards
    multi_shard_params = {k: v for k, v in param_base_names.items() if len(v) > 1}
    
    print(f"Parameters with multiple entries: {len(multi_shard_params)}")
    
    if multi_shard_params:
        print("\nMulti-shard parameters:")
        for base_name, keys in list(multi_shard_params.items())[:10]:
            print(f"  {base_name}: {len(keys)} entries")
            for key in keys[:3]:
                print(f"    {key}")
    
    # Also check for flat parameters which are common in FSDP
    flat_params = [k for k in metadata.state_dict_metadata.keys() if 'flat_param' in k]
    print(f"\nFlat parameters found: {len(flat_params)}")
    
    if flat_params:
        print("Sample flat parameters:")
        for key in flat_params[:10]:
            print(f"  {key}")
    
    return multi_shard_params, flat_params


def main():
    """Main function to run the enhanced checkpoint analysis."""
    step0_path = "/data/input/amanr/OLMo-core/step0/model_and_optim"
    
    if not os.path.exists(step0_path):
        print(f"Error: Checkpoint path {step0_path} does not exist")
        return
    
    print("ENHANCED OLMO-CORE CHECKPOINT ANALYSIS")
    print("="*60)
    print(f"Analyzing: {step0_path}")
    print("="*60)
    
    # Step 1: Analyze metadata structure
    metadata, model_keys, weight_keys = analyze_metadata_structure(step0_path)
    
    # Step 2: Check for FSDP sharding patterns
    multi_shard_params, flat_params = check_for_fsdp_sharding_pattern(step0_path)
    
    # Step 3: Analyze actual tensor data
    sample_keys = weight_keys[:10] if weight_keys else model_keys[:10]
    analyze_tensor_sharding(step0_path, sample_keys)
    
    # Step 4: If we found flat parameters, analyze them specifically
    if flat_params:
        print(f"\nANALYZING FLAT PARAMETERS FOR SHARDING PATTERNS")
        print("="*50)
        analyze_tensor_sharding(step0_path, flat_params[:5])
    
    print(f"\nANALYSIS COMPLETE")
    print("="*50)


if __name__ == "__main__":
    main()