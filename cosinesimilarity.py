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


def compute_cosine_similarity_matrix(tensors: List[torch.Tensor]) -> torch.Tensor:
    """Compute pairwise cosine similarities between a list of tensors."""
    n = len(tensors)
    similarity_matrix = torch.zeros(n, n)
    
    for i in range(n):
        for j in range(i, n):
            if i == j:
                similarity_matrix[i, j] = 1.0
            else:
                cos_sim = F.cosine_similarity(tensors[i].flatten().unsqueeze(0), 
                                            tensors[j].flatten().unsqueeze(0)).item()
                similarity_matrix[i, j] = cos_sim
                similarity_matrix[j, i] = cos_sim
    
    return similarity_matrix

def analyze_tensor_cosine_similarities(key: str, tensor: torch.Tensor):
    """Analyze cosine similarities across different dimensions of a tensor."""
    log_and_print(f"\n🔍 Analyzing cosine similarities in {key} (shape: {tensor.shape})")
    
    results = {
        'layer_wise': {},
        'block_wise': {},
        'row_wise': None,
        'col_wise': None
    }
    
    # Block-wise analysis across each dimension
    for dim in range(tensor.ndim):
        size = tensor.shape[dim]
        results['block_wise'][dim] = {}
        
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
                
                # Compute cosine similarity matrix for blocks
                if len(blocks) >= 2:
                    similarity_matrix = compute_cosine_similarity_matrix(blocks)
                    
                    # Extract upper triangular part (excluding diagonal)
                    upper_tri = similarity_matrix[torch.triu(torch.ones_like(similarity_matrix), diagonal=1) == 1]
                    avg_cos_sim = upper_tri.mean().item()
                    
                    results['block_wise'][dim][block_size] = {
                        'avg_cosine_similarity': avg_cos_sim,
                        'num_blocks': num_blocks,
                        'similarity_matrix': similarity_matrix
                    }
                    
                    log_and_print(f"    📊 DIM {dim}, BLOCK_SIZE {block_size}: Avg cosine similarity = {avg_cos_sim:.6f} ({num_blocks} blocks)")
    
    # Periodic pattern analysis with cosine similarity
    flat = tensor.flatten()
    if len(flat) > 100:
        periodic_results = {}
        for period in [2, 3, 4, 8, 16, 32, 64, 128, 256]:
            if len(flat) >= period * 3:
                num_chunks = len(flat) // period
                if num_chunks < 2:
                    continue
                
                chunks = []
                for i in range(min(num_chunks, 50)):  # Limit to 50 chunks for performance
                    start = i * period
                    end = start + period
                    chunks.append(flat[start:end])
                
                # Compute average cosine similarity for periodic patterns
                if len(chunks) >= 2:
                    similarity_matrix = compute_cosine_similarity_matrix(chunks)
                    upper_tri = similarity_matrix[torch.triu(torch.ones_like(similarity_matrix), diagonal=1) == 1]
                    avg_cos_sim = upper_tri.mean().item()
                    
                    periodic_results[period] = {
                        'avg_cosine_similarity': avg_cos_sim,
                        'num_chunks': len(chunks)
                    }
                    
                    log_and_print(f"    📊 PERIODIC PATTERN (period {period}): Avg cosine similarity = {avg_cos_sim:.6f} ({len(chunks)} chunks)")
        
        results['periodic'] = periodic_results
    
    # Row-wise and column-wise analysis for 2D tensors
    if tensor.ndim == 2:
        rows, cols = tensor.shape
        
        # Row-wise cosine similarity analysis
        if rows >= 2:
            row_tensors = [tensor[i] for i in range(min(rows, 200))]  # Limit to 200 rows for performance
            row_similarity_matrix = compute_cosine_similarity_matrix(row_tensors)
            upper_tri_rows = row_similarity_matrix[torch.triu(torch.ones_like(row_similarity_matrix), diagonal=1) == 1]
            avg_row_cos_sim = upper_tri_rows.mean().item()
            
            results['row_wise'] = {
                'avg_cosine_similarity': avg_row_cos_sim,
                'num_rows_analyzed': len(row_tensors),
                'similarity_matrix': row_similarity_matrix
            }
            
            log_and_print(f"    📊 ROW-WISE: Avg cosine similarity = {avg_row_cos_sim:.6f} ({len(row_tensors)} rows)")
        
        # Column-wise cosine similarity analysis
        if cols >= 2:
            col_tensors = [tensor[:, i] for i in range(min(cols, 200))]  # Limit to 200 columns for performance
            col_similarity_matrix = compute_cosine_similarity_matrix(col_tensors)
            upper_tri_cols = col_similarity_matrix[torch.triu(torch.ones_like(col_similarity_matrix), diagonal=1) == 1]
            avg_col_cos_sim = upper_tri_cols.mean().item()
            
            results['col_wise'] = {
                'avg_cosine_similarity': avg_col_cos_sim,
                'num_cols_analyzed': len(col_tensors),
                'similarity_matrix': col_similarity_matrix
            }
            
            log_and_print(f"    📊 COLUMN-WISE: Avg cosine similarity = {avg_col_cos_sim:.6f} ({len(col_tensors)} columns)")
    
    return results


def analyze_sharding_cosine_similarity(key: str, tensor: torch.Tensor, num_shards: int = 64):
    """
    Analyze cosine similarity patterns in sharded tensor data.
    """
    flat_tensor = tensor.flatten()
    
    # Check for sharding patterns
    chunk_size = len(flat_tensor) // num_shards
    if chunk_size == 0:
        return None
    
    chunks = []
    for i in range(min(num_shards, len(flat_tensor) // chunk_size)):
        start_idx = i * chunk_size
        end_idx = start_idx + chunk_size
        chunks.append(flat_tensor[start_idx:end_idx])
    
    if len(chunks) < 2:
        return None
    
    # Compute average cosine similarity across shards
    similarity_matrix = compute_cosine_similarity_matrix(chunks)
    upper_tri = similarity_matrix[torch.triu(torch.ones_like(similarity_matrix), diagonal=1) == 1]
    avg_shard_cos_sim = upper_tri.mean().item()
    
    log_and_print(f"    📊 SHARDING PATTERN ({num_shards} shards): Avg cosine similarity = {avg_shard_cos_sim:.6f}")
    
    return {
        'avg_cosine_similarity': avg_shard_cos_sim,
        'num_shards': len(chunks),
        'similarity_matrix': similarity_matrix
    }


def analyze_layer_cosine_similarities(layer_tensors: Dict[str, torch.Tensor]) -> Dict[str, float]:
    """
    Compute average cosine similarities across layers of the same type.
    """
    layer_results = {}
    
    if len(layer_tensors) < 2:
        return layer_results
    
    keys = list(layer_tensors.keys())
    tensors = [layer_tensors[k] for k in keys]
    
    # Compute pairwise cosine similarities between layers
    similarities = []
    for i in range(len(tensors)):
        for j in range(i + 1, len(tensors)):
            if tensors[i].shape == tensors[j].shape:
                cos_sim = F.cosine_similarity(tensors[i].flatten().unsqueeze(0), 
                                            tensors[j].flatten().unsqueeze(0)).item()
                similarities.append(cos_sim)
    
    if similarities:
        avg_layer_cos_sim = np.mean(similarities)
        layer_results['avg_cosine_similarity'] = avg_layer_cos_sim
        layer_results['num_comparisons'] = len(similarities)
        layer_results['layer_keys'] = keys
        
        log_and_print(f"    📊 LAYER-WISE: Avg cosine similarity = {avg_layer_cos_sim:.6f} ({len(similarities)} pairs)")
    
    return layer_results

def analyze_similarity_patterns(layer_similarities, row_similarities, col_similarities, block_similarities):
    """
    Analyze patterns in average cosine similarities to identify similar behaviors.
    """
    log_and_print(f"\n🔍 PATTERN ANALYSIS OF AVERAGE COSINE SIMILARITIES")
    log_and_print("="*60)
    
    # Analyze layer-wise similarities
    if layer_similarities:
        log_and_print(f"\n📊 Layer-wise Similarity Patterns:")
        layer_sims = [sim for _, sim in layer_similarities]
        avg_layer_sim = np.mean(layer_sims)
        std_layer_sim = np.std(layer_sims)
        
        log_and_print(f"   Overall avg: {avg_layer_sim:.6f} ± {std_layer_sim:.6f}")
        
        # Group by similarity ranges
        high_sim_layers = [(name, sim) for name, sim in layer_similarities if sim > avg_layer_sim + std_layer_sim]
        low_sim_layers = [(name, sim) for name, sim in layer_similarities if sim < avg_layer_sim - std_layer_sim]
        
        if high_sim_layers:
            log_and_print(f"   🔴 High similarity layers:")
            for name, sim in high_sim_layers:
                log_and_print(f"     • {name}: {sim:.6f}")
        
        if low_sim_layers:
            log_and_print(f"   🔵 Low similarity layers:")
            for name, sim in low_sim_layers:
                log_and_print(f"     • {name}: {sim:.6f}")
    
    # Analyze row-wise similarities
    if row_similarities:
        log_and_print(f"\n📊 Row-wise Similarity Patterns:")
        row_sims = [sim for _, sim in row_similarities]
        avg_row_sim = np.mean(row_sims)
        std_row_sim = np.std(row_sims)
        
        log_and_print(f"   Overall avg: {avg_row_sim:.6f} ± {std_row_sim:.6f}")
        
        # Find tensors with unusual row similarities
        unusual_rows = [(name, sim) for name, sim in row_similarities 
                       if abs(sim - avg_row_sim) > 2 * std_row_sim]
        
        if unusual_rows:
            log_and_print(f"   ⚠️  Unusual row similarities:")
            for name, sim in unusual_rows:
                log_and_print(f"     • {name}: {sim:.6f}")
    
    # Analyze column-wise similarities
    if col_similarities:
        log_and_print(f"\n📊 Column-wise Similarity Patterns:")
        col_sims = [sim for _, sim in col_similarities]
        avg_col_sim = np.mean(col_sims)
        std_col_sim = np.std(col_sims)
        
        log_and_print(f"   Overall avg: {avg_col_sim:.6f} ± {std_col_sim:.6f}")
        
        # Find tensors with unusual column similarities
        unusual_cols = [(name, sim) for name, sim in col_similarities 
                       if abs(sim - avg_col_sim) > 2 * std_col_sim]
        
        if unusual_cols:
            log_and_print(f"   ⚠️  Unusual column similarities:")
            for name, sim in unusual_cols:
                log_and_print(f"     • {name}: {sim:.6f}")
    
    # Analyze block-wise similarities
    if block_similarities:
        log_and_print(f"\n📊 Block-wise Similarity Patterns:")
        
        # Group by block size
        block_groups = defaultdict(list)
        for name, dim, block_size, sim in block_similarities:
            block_groups[block_size].append((name, dim, sim))
        
        for block_size, entries in block_groups.items():
            sims = [sim for _, _, sim in entries]
            avg_block_sim = np.mean(sims)
            std_block_sim = np.std(sims)
            
            log_and_print(f"   Block size {block_size}: avg = {avg_block_sim:.6f} ± {std_block_sim:.6f}")
            
            # Find unusual block similarities
            unusual_blocks = [(name, dim, sim) for name, dim, sim in entries 
                             if abs(sim - avg_block_sim) > 2 * std_block_sim]
            
            if unusual_blocks:
                log_and_print(f"     ⚠️  Unusual similarities:")
                for name, dim, sim in unusual_blocks:
                    log_and_print(f"       • {name} dim {dim}: {sim:.6f}")

def find_cosine_similarity_patterns(checkpoint_path: str):
    """
    Analyze cosine similarity patterns across layers and weights.
    """
    log_and_print("🔍 ANALYZING COSINE SIMILARITY PATTERNS")
    log_and_print("="*60)
    log_and_print("🔧 Computing average cosine similarities across layers, blocks, rows, and columns")
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
    
    # Analyze each layer type for cosine similarity patterns
    similarity_analysis = []
    all_tensor_results = {}
    
    for layer_type, keys in layer_groups.items():
        if len(keys) < 2:
            continue
            
        log_and_print(f"\n🔍 Analyzing {layer_type} ({len(keys)} instances)")
        
        # Load tensors for comparison
        layer_tensors = {}
        
        for key in keys[:10]:  # Limit to first 10 for performance
            try:
                tensor_data = next(load_keys(checkpoint_path, [key]))
                layer_tensors[key] = tensor_data
            except Exception as e:
                log_and_print(f"⚠️  Could not load {key}: {e}")
                continue
        
        if len(layer_tensors) < 2:
            continue
        
        # Layer-wise cosine similarity analysis
        layer_results = analyze_layer_cosine_similarities(layer_tensors)
        
        # Individual tensor analysis
        for key, tensor in layer_tensors.items():
            if tensor.numel() > 1000:  # Only analyze larger tensors
                tensor_results = analyze_tensor_cosine_similarities(key, tensor)
                sharding_results = analyze_sharding_cosine_similarity(key, tensor)
                
                all_tensor_results[key] = {
                    'tensor_analysis': tensor_results,
                    'sharding_analysis': sharding_results
                }
        
        if layer_results:
            similarity_analysis.append({
                'layer_type': layer_type,
                'layer_results': layer_results,
                'tensor_results': {k: v for k, v in all_tensor_results.items() if k in layer_tensors}
            })
    
    # Print summary and analyze patterns
    log_and_print(f"\n📋 SUMMARY OF COSINE SIMILARITY ANALYSIS")
    log_and_print("="*60)
    
    if not similarity_analysis:
        log_and_print("✅ No similarity patterns found")
        return
    
    # Analyze patterns across different layer types
    layer_similarities = []
    row_similarities = []
    col_similarities = []
    block_similarities = []
    
    for analysis in similarity_analysis:
        layer_type = analysis['layer_type']
        layer_results = analysis['layer_results']
        
        log_and_print(f"\n📍 Layer Type: {layer_type}")
        
        if 'avg_cosine_similarity' in layer_results:
            layer_sim = layer_results['avg_cosine_similarity']
            layer_similarities.append((layer_type, layer_sim))
            log_and_print(f"   📊 Layer-wise avg cosine similarity: {layer_sim:.6f}")
        
        # Aggregate tensor-level results
        for key, results in analysis['tensor_results'].items():
            tensor_analysis = results['tensor_analysis']
            
            if tensor_analysis.get('row_wise'):
                row_sim = tensor_analysis['row_wise']['avg_cosine_similarity']
                row_similarities.append((key, row_sim))
            
            if tensor_analysis.get('col_wise'):
                col_sim = tensor_analysis['col_wise']['avg_cosine_similarity']
                col_similarities.append((key, col_sim))
            
            # Aggregate block-wise results
            block_wise = tensor_analysis.get('block_wise', {})
            for dim, block_results in block_wise.items():
                for block_size, block_data in block_results.items():
                    block_sim = block_data['avg_cosine_similarity']
                    block_similarities.append((key, dim, block_size, block_sim))
    
    # Pattern analysis
    analyze_similarity_patterns(layer_similarities, row_similarities, col_similarities, block_similarities)
    
    return similarity_analysis


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
    
    step0_path = "/data/input/amanr/step10000/model_and_optim"
    
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
        # Step 1: Analyze cosine similarity patterns
        similarity_analysis = find_cosine_similarity_patterns(step0_path)
        
        # Step 2: Analyze weight statistics
        analyze_weight_statistics(step0_path)
        
        log_and_print(f"\n✅ ANALYSIS COMPLETE")
        log_and_print("="*60)
        
    finally:
        log_file.close()


if __name__ == "__main__":
    main()