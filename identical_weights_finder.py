#!/usr/bin/env python3

import os
import sys
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, List, Tuple, Any
from collections import defaultdict
import numpy as np
import datetime

sys.path.insert(0, '/data/input/amanr/OLMo-core/src')

from olmo_core.distributed.checkpoint import get_checkpoint_metadata, load_keys

log_file = None

def log_and_print(message: str):
    print(message)
    if log_file:
        log_file.write(message + '\n')
        log_file.flush()

def format_tensor_values(tensor: torch.Tensor, max_elements: int = 20) -> str:
    flat = tensor.flatten()
    if len(flat) <= max_elements:
        values = ', '.join(f"{v:.6f}" for v in flat)
        return f"[{values}]"
    else:
        first_few = ', '.join(f"{v:.6f}" for v in flat[:max_elements//2])
        last_few = ', '.join(f"{v:.6f}" for v in flat[-max_elements//2:])
        return f"[{first_few} ... {last_few}]"

def get_tensor_location_info(tensor: torch.Tensor, block_indices: Tuple[int, int], dim: int, block_size: int) -> str:
    i, j = block_indices
    
    if tensor.ndim == 2:
        if dim == 0:  
            return f"rows {i*block_size}-{(i+1)*block_size-1} ≡ rows {j*block_size}-{(j+1)*block_size-1}"
        else: 
            return f"cols {i*block_size}-{(i+1)*block_size-1} ≡ cols {j*block_size}-{(j+1)*block_size-1}"
    else:
        return f"dim {dim}: blocks {i*block_size}-{(i+1)*block_size-1} ≡ blocks {j*block_size}-{(j+1)*block_size-1}"


def check_tensor_patterns(key: str, tensor: torch.Tensor):
    log_and_print(f"\n🔍 Checking patterns in {key} (shape: {tensor.shape})")

    for dim in range(tensor.ndim):
        size = tensor.shape[dim]
        for block_size in [2, 3, 4, 8, 16, 32]:
            if size >= block_size * 2:
                num_blocks = size // block_size
                if num_blocks < 2:
                    continue
                
                blocks = []
                for i in range(num_blocks):
                    slices = [slice(None)] * tensor.ndim
                    slices[dim] = slice(i * block_size, (i + 1) * block_size)
                    blocks.append(tensor[tuple(slices)])

                identical_pairs = []
                similar_pairs = []
                for i in range(len(blocks)):
                    for j in range(i + 1, len(blocks)):
                        if torch.equal(blocks[i], blocks[j]):
                            identical_pairs.append((i, j))
                        else:
                            cos_sim = F.cosine_similarity(blocks[i].flatten().unsqueeze(0), 
                                                        blocks[j].flatten().unsqueeze(0)).item()
                            if cos_sim > 0.85:
                                similar_pairs.append((i, j, cos_sim))
                
                if identical_pairs:
                    log_and_print(f"    🎯 BLOCK PATTERN: {key} - dim {dim}, block_size {block_size}")
                    for i, j in identical_pairs[:5]:  
                        location_info = get_tensor_location_info(tensor, (i, j), dim, block_size)
                        log_and_print(f"      • IDENTICAL: {location_info}")
                        log_and_print(f"        Values: {format_tensor_values(blocks[i])}")
                    if len(identical_pairs) > 5:
                        log_and_print(f"      • ... and {len(identical_pairs) - 5} more pairs")
                
                if similar_pairs:
                    log_and_print(f"    🔸 SIMILAR BLOCKS: {key} - dim {dim}, block_size {block_size}")
                    for i, j, cos_sim in similar_pairs[:3]:  
                        location_info = get_tensor_location_info(tensor, (i, j), dim, block_size)
                        log_and_print(f"      • SIMILAR: {location_info} (similarity: {cos_sim:.6f})")
                        log_and_print(f"        Block {i}: {format_tensor_values(blocks[i])}")
                        log_and_print(f"        Block {j}: {format_tensor_values(blocks[j])}")
                    if len(similar_pairs) > 3:
                        log_and_print(f"      • ... and {len(similar_pairs) - 3} more pairs")
    
    flat = tensor.flatten()
    if len(flat) > 100:
        for period in [2, 3, 4, 8, 16, 32, 64, 128, 256]:
            if len(flat) >= period * 3:
                num_chunks = len(flat) // period
                if num_chunks < 2:
                    continue
                
                chunks = []
                for i in range(num_chunks):
                    start = i * period
                    end = start + period
                    chunks.append(flat[start:end])
                
                identical_chunks = []
                similar_chunks = []
                for i in range(len(chunks)):
                    for j in range(i + 1, min(len(chunks), 10)):
                        if torch.equal(chunks[i], chunks[j]):
                            identical_chunks.append((i, j))
                        else:
                            cos_sim = F.cosine_similarity(chunks[i].unsqueeze(0), chunks[j].unsqueeze(0)).item()
                            if cos_sim > 0.85:
                                similar_chunks.append((i, j, cos_sim))
                
                if identical_chunks:
                    log_and_print(f"    🎯 PERIODIC PATTERN: {key} - period {period}")
                    for i, j in identical_chunks[:3]:  # Show first 3 pairs
                        log_and_print(f"      • IDENTICAL: chunks {i} and {j} (positions {i*period}-{(i+1)*period-1} ≡ {j*period}-{(j+1)*period-1})")
                        log_and_print(f"        Values: {format_tensor_values(chunks[i])}")
                    if len(identical_chunks) > 3:
                        log_and_print(f"      • ... and {len(identical_chunks) - 3} more pairs")
                
                if similar_chunks:
                    log_and_print(f"    🔸 SIMILAR PERIODIC: {key} - period {period}")
                    for i, j, cos_sim in similar_chunks[:2]:  # Show first 2 pairs
                        log_and_print(f"      • SIMILAR: chunks {i} and {j} (similarity: {cos_sim:.6f})")
                        log_and_print(f"        Chunk {i}: {format_tensor_values(chunks[i])}")
                        log_and_print(f"        Chunk {j}: {format_tensor_values(chunks[j])}")
                    if len(similar_chunks) > 2:
                        log_and_print(f"      • ... and {len(similar_chunks) - 2} more pairs")
    
    # Check for block patterns in 2D tensors (common for weight matrices)
    if tensor.ndim == 2:
        rows, cols = tensor.shape
        
        # Check for identical rows
        identical_rows = []
        similar_rows = []
        for i in range(rows):
            for j in range(i + 1, min(rows, 100)):  # Check first 100 rows
                if torch.equal(tensor[i], tensor[j]):
                    identical_rows.append((i, j))
                else:
                    cos_sim = F.cosine_similarity(tensor[i].unsqueeze(0), tensor[j].unsqueeze(0)).item()
                    if cos_sim > 0.85:
                        similar_rows.append((i, j, cos_sim))
        
        if identical_rows:
            log_and_print(f"    🎯 ROW PATTERN: {key} - identical rows found")
            for i, j in identical_rows[:5]:
                log_and_print(f"      • IDENTICAL: rows {i} ≡ {j}")
                log_and_print(f"        Values: {format_tensor_values(tensor[i])}")
            if len(identical_rows) > 5:
                log_and_print(f"      • ... and {len(identical_rows) - 5} more pairs")
        
        if similar_rows:
            log_and_print(f"    🔸 SIMILAR ROWS: {key}")
            for i, j, cos_sim in similar_rows[:3]:
                log_and_print(f"      • SIMILAR: rows {i} ≈ {j} (similarity: {cos_sim:.6f})")
                log_and_print(f"        Row {i}: {format_tensor_values(tensor[i])}")
                log_and_print(f"        Row {j}: {format_tensor_values(tensor[j])}")
            if len(similar_rows) > 3:
                log_and_print(f"      • ... and {len(similar_rows) - 3} more pairs")
        
        # Check for identical columns
        identical_cols = []
        similar_cols = []
        for i in range(cols):
            for j in range(i + 1, min(cols, 100)):  # Check first 100 cols
                if torch.equal(tensor[:, i], tensor[:, j]):
                    identical_cols.append((i, j))
                else:
                    cos_sim = F.cosine_similarity(tensor[:, i].unsqueeze(0), tensor[:, j].unsqueeze(0)).item()
                    if cos_sim > 0.85:
                        similar_cols.append((i, j, cos_sim))
        
        if identical_cols:
            log_and_print(f"    🎯 COLUMN PATTERN: {key} - identical columns found")
            for i, j in identical_cols[:5]:
                log_and_print(f"      • IDENTICAL: columns {i} ≡ {j}")
                log_and_print(f"        Values: {format_tensor_values(tensor[:, i])}")
            if len(identical_cols) > 5:
                log_and_print(f"      • ... and {len(identical_cols) - 5} more pairs")
        
        if similar_cols:
            log_and_print(f"    🔸 SIMILAR COLUMNS: {key}")
            for i, j, cos_sim in similar_cols[:3]:
                log_and_print(f"      • SIMILAR: columns {i} ≈ {j} (similarity: {cos_sim:.6f})")
                log_and_print(f"        Column {i}: {format_tensor_values(tensor[:, i])}")
                log_and_print(f"        Column {j}: {format_tensor_values(tensor[:, j])}")
            if len(similar_cols) > 3:
                log_and_print(f"      • ... and {len(similar_cols) - 3} more pairs")


def check_64_node_sharding_pattern(key: str, tensor: torch.Tensor):
    """
    Check for 64-node sharding patterns within a single tensor.
    This helps identify if data is replicated across nodes.
    """
    flat_tensor = tensor.flatten()
    
    # Check for 64-way sharding (matching num_nodes=64)
    chunk_size = len(flat_tensor) // 64
    if chunk_size == 0:
        return
    
    chunks = []
    for i in range(min(64, len(flat_tensor) // chunk_size)):
        start_idx = i * chunk_size
        end_idx = start_idx + chunk_size
        chunks.append(flat_tensor[start_idx:end_idx])
    
    if len(chunks) < 2:
        return
    
    # Check for identical chunks (nodes)
    identical_nodes = []
    for i in range(len(chunks)):
        for j in range(i + 1, min(len(chunks), 8)):  # Check first 8 chunks
            if torch.equal(chunks[i], chunks[j]):
                identical_nodes.append((i, j))
                log_and_print(f"    🚨 IDENTICAL NODE SHARDS: {key} node_{i} ≡ node_{j}")
    
    # Check for high similarity between nodes
    high_similarity_nodes = []
    for i in range(min(8, len(chunks))):
        for j in range(i + 1, min(8, len(chunks))):
            cos_sim = F.cosine_similarity(chunks[i].unsqueeze(0), chunks[j].unsqueeze(0)).item()
            if cos_sim > 0.99:
                high_similarity_nodes.append((i, j, cos_sim))
                log_and_print(f"    ⚠️  HIGH NODE SIMILARITY: {key} node_{i} ≈ node_{j} (cos_sim: {cos_sim:.6f})")
    
    if identical_nodes:
        log_and_print(f"    🎯 Found {len(identical_nodes)} identical node pairs in {key}")
    if high_similarity_nodes:
        log_and_print(f"    🔸 Found {len(high_similarity_nodes)} highly similar node pairs in {key}")


def find_identical_weights(checkpoint_path: str):
    """
    Find layers and weights that have identical initialization patterns.
    """
    log_and_print("🔍 SEARCHING FOR IDENTICAL WEIGHT INITIALIZATION PATTERNS")
    log_and_print("="*60)
    log_and_print("🔧 Configured for 64-node distributed training analysis")
    log_and_print("="*60)
    
    if not os.path.exists(checkpoint_path):
        log_and_print(f"❌ Error: Checkpoint path {checkpoint_path} does not exist")
        return
    
    # Load checkpoint metadata
    metadata = get_checkpoint_metadata(checkpoint_path)
    
    # Get all weight keys
    weight_keys = [k for k in metadata.state_dict_metadata.keys() 
                   if k.startswith('model.') and ('weight' in k or 'bias' in k)]
    
    log_and_print(f"📊 Found {len(weight_keys)} weight/bias parameters")
    
    # Group weights by layer type and analyze patterns
    layer_groups = defaultdict(list)
    
    for key in weight_keys:
        # Extract layer type (e.g., "transformer.blocks.0.attn.q_proj" -> "attn.q_proj")
        parts = key.split('.')
        if len(parts) >= 4:
            layer_type = '.'.join(parts[-2:])  # Get last two parts
            layer_groups[layer_type].append(key)
    
    log_and_print(f"🔧 Found {len(layer_groups)} different layer types")
    
    # Analyze each layer type for identical patterns
    identical_groups = []
    
    for layer_type, keys in layer_groups.items():
        if len(keys) < 2:
            continue
            
        log_and_print(f"\n🔍 Analyzing {layer_type} ({len(keys)} instances)")
        
        # Load tensors for comparison
        tensors = []
        valid_keys = []
        
        for key in keys[:10]:  # Limit to first 10 for performance
            try:
                tensor_data = next(load_keys(checkpoint_path, [key]))
                tensors.append(tensor_data)
                valid_keys.append(key)
            except Exception as e:
                log_and_print(f"⚠️  Could not load {key}: {e}")
                continue
        
        if len(tensors) < 2:
            continue
            
        # Compare tensors pairwise
        identical_pairs = []
        similar_pairs = []
        
        for i in range(len(tensors)):
            for j in range(i + 1, len(tensors)):
                tensor1, tensor2 = tensors[i], tensors[j]
                key1, key2 = valid_keys[i], valid_keys[j]
                
                # Check if tensors have the same shape
                if tensor1.shape != tensor2.shape:
                    continue
                
                # Check for exact equality
                if torch.equal(tensor1, tensor2):
                    identical_pairs.append((key1, key2))
                    log_and_print(f"🎯 IDENTICAL: {key1} ≡ {key2}")
                    continue
                
                # Check for high similarity
                flat1 = tensor1.flatten()
                flat2 = tensor2.flatten()
                
                cos_sim = F.cosine_similarity(flat1.unsqueeze(0), flat2.unsqueeze(0)).item()
                l2_diff = torch.norm(flat1 - flat2).item()
                
                if cos_sim > 0.999:  # Very high similarity
                    similar_pairs.append((key1, key2, cos_sim, l2_diff))
                    log_and_print(f"🔸 HIGHLY SIMILAR: {key1} ≈ {key2} (cos_sim: {cos_sim:.6f}, l2_diff: {l2_diff:.6f})")
                
                # Check for various patterns within individual tensors
                if len(flat1) > 1000:
                    check_tensor_patterns(key1, tensor1)
                    check_64_node_sharding_pattern(key1, tensor1)
        
        if identical_pairs or similar_pairs:
            identical_groups.append({
                'layer_type': layer_type,
                'identical_pairs': identical_pairs,
                'similar_pairs': similar_pairs
            })
    
    # Print summary
    log_and_print(f"\n📋 SUMMARY OF IDENTICAL/SIMILAR WEIGHTS")
    log_and_print("="*60)
    
    if not identical_groups:
        log_and_print("✅ No identical or highly similar weight patterns found")
        return
    
    total_identical = sum(len(g['identical_pairs']) for g in identical_groups)
    total_similar = sum(len(g['similar_pairs']) for g in identical_groups)
    
    log_and_print(f"🎯 Found {total_identical} identical weight pairs")
    log_and_print(f"🔸 Found {total_similar} highly similar weight pairs")
    
    for group in identical_groups:
        log_and_print(f"\n📍 Layer Type: {group['layer_type']}")
        
        if group['identical_pairs']:
            log_and_print("   🎯 Identical Pairs:")
            for key1, key2 in group['identical_pairs']:
                log_and_print(f"     • {key1}")
                log_and_print(f"     • {key2}")
        
        if group['similar_pairs']:
            log_and_print("   🔸 Similar Pairs:")
            for key1, key2, cos_sim, l2_diff in group['similar_pairs']:
                log_and_print(f"     • {key1} ≈ {key2} (similarity: {cos_sim:.6f})")
    
    return identical_groups


def analyze_weight_statistics(checkpoint_path: str):
    """
    Analyze weight statistics to identify suspicious patterns.
    """
    log_and_print(f"\n📊 WEIGHT STATISTICS ANALYSIS")
    log_and_print("="*60)
    
    metadata = get_checkpoint_metadata(checkpoint_path)
    weight_keys = [k for k in metadata.state_dict_metadata.keys() 
                   if k.startswith('model.') and 'weight' in k]
    
    stats = []
    
    for key in weight_keys[:20]:  # Analyze first 20 weights
        try:
            tensor_data = next(load_keys(checkpoint_path, [key]))
            flat_tensor = tensor_data.flatten()
            
            stat = {
                'key': key,
                'shape': tuple(tensor_data.shape),
                'mean': flat_tensor.mean().item(),
                'std': flat_tensor.std().item(),
                'min': flat_tensor.min().item(),
                'max': flat_tensor.max().item(),
                'zeros': (flat_tensor == 0).sum().item(),
                'total': flat_tensor.numel()
            }
            stats.append(stat)
            
        except Exception as e:
            log_and_print(f"⚠️  Could not analyze {key}: {e}")
    
    # Group by similar statistics
    log_and_print(f"\n📈 Weight Statistics:")
    for stat in stats:
        zero_pct = (stat['zeros'] / stat['total']) * 100
        log_and_print(f"🔹 {stat['key']}")
        log_and_print(f"   Shape: {stat['shape']}, Mean: {stat['mean']:.6f}, Std: {stat['std']:.6f}")
        log_and_print(f"   Min: {stat['min']:.6f}, Max: {stat['max']:.6f}, Zeros: {zero_pct:.2f}%")
    
    # Look for suspicious patterns
    log_and_print(f"\n🚨 SUSPICIOUS PATTERNS:")
    
    # Check for weights with identical statistics
    stat_groups = defaultdict(list)
    for stat in stats:
        # Group by rounded statistics to catch nearly identical patterns
        key = (round(stat['mean'], 4), round(stat['std'], 4), round(stat['min'], 4), round(stat['max'], 4))
        stat_groups[key].append(stat)
    
    for stat_key, group in stat_groups.items():
        if len(group) > 1:
            log_and_print(f"⚠️  {len(group)} weights with identical statistics:")
            for stat in group:
                log_and_print(f"   • {stat['key']}")


def main():
    """Main function to find identical weights."""
    global log_file
    
    # Initialize log file
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"identical_weights_analysis_{timestamp}.txt"
    log_file = open(log_filename, 'w')
    
    step0_path = "/data/input/amanr/step0/model_and_optim"
    
    if not os.path.exists(step0_path):
        log_and_print(f"❌ Error: Checkpoint path {step0_path} does not exist")
        log_file.close()
        return
    
    log_and_print("🔍 IDENTICAL WEIGHTS DETECTION TOOL")
    log_and_print("="*60)
    log_and_print(f"📁 Analyzing: {step0_path}")
    log_and_print(f"📝 Log file: {log_filename}")
    log_and_print("="*60)
    
    try:
        # Step 1: Find identical weights
        identical_groups = find_identical_weights(step0_path)
        
        # Step 2: Analyze weight statistics
        analyze_weight_statistics(step0_path)
        
        log_and_print(f"\n✅ ANALYSIS COMPLETE")
        log_and_print("="*60)
        
    finally:
        log_file.close()


if __name__ == "__main__":
    main()