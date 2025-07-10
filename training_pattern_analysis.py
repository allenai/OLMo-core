#!/usr/bin/env python3
"""
Analyze how the initialization bug pattern survives through training.
Check multiple checkpoints to see how identical sections diverge over time.
"""

import os
import sys
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, List, Tuple, Any
import glob
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

# Add the olmo_core source to path
sys.path.insert(0, '/data/input/amanr/OLMo-core/src')

from olmo_core.distributed.checkpoint import get_checkpoint_metadata, load_keys


def analyze_tensor_chunks(tensor_data: torch.Tensor, num_chunks: int = 64) -> Dict[str, Any]:
    flat_tensor = tensor_data.flatten()
    chunk_size = len(flat_tensor) // num_chunks
    
    # For 64-node distributed training, we expect 64-way sharding
    # Each chunk should represent data from one node/GPU
    
    if chunk_size == 0:
        return {"error": "Tensor too small for chunking"}
    
    chunks = []
    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = start_idx + chunk_size
        if end_idx <= len(flat_tensor):
            chunks.append(flat_tensor[start_idx:end_idx])
    
    if len(chunks) < 2:
        return {"error": "Not enough chunks for comparison"}
    
    # Calculate pairwise similarities
    similarities = []
    l2_distances = []
    identical_count = 0
    
    for i in range(len(chunks)):
        for j in range(i + 1, len(chunks)):
            chunk1, chunk2 = chunks[i], chunks[j]
            cos_sim = F.cosine_similarity(chunk1.unsqueeze(0), chunk2.unsqueeze(0)).item()
            l2_dist = torch.norm(chunk1 - chunk2).item()
            identical = torch.equal(chunk1, chunk2)
            
            similarities.append(cos_sim)
            l2_distances.append(l2_dist)
            if identical:
                identical_count += 1
    
    return {
        "num_chunks": len(chunks),
        "chunk_size": chunk_size,
        "mean_cosine_similarity": sum(similarities) / len(similarities),
        "max_cosine_similarity": max(similarities),
        "min_cosine_similarity": min(similarities),
        "mean_l2_distance": sum(l2_distances) / len(l2_distances),
        "max_l2_distance": max(l2_distances),
        "min_l2_distance": min(l2_distances),
        "identical_pairs": identical_count,
        "total_pairs": len(similarities),
        "identical_ratio": identical_count / len(similarities) if similarities else 0,
        "chunks": chunks,
        "similarities": similarities,
        "l2_distances": l2_distances
    }


def analyze_checkpoint_evolution(checkpoint_paths: List[str], target_param: str = "model.embeddings.weight"):
    """
    Analyze how the initialization bug pattern evolves across multiple checkpoints.
    
    Args:
        checkpoint_paths: List of checkpoint directory paths
        target_param: Parameter name to analyze
    """
    print(f"TRAINING PATTERN EVOLUTION ANALYSIS")
    print(f"Target parameter: {target_param}")
    print(f"Analyzing {len(checkpoint_paths)} checkpoints")
    
    results = []
    
    for i, checkpoint_path in enumerate(checkpoint_paths):
        print(f"\nCheckpoint {i+1}/{len(checkpoint_paths)}: {Path(checkpoint_path).name}")
        
        try:
            # Check if checkpoint exists and has the target parameter
            metadata = get_checkpoint_metadata(checkpoint_path)
            
            if target_param not in metadata.state_dict_metadata:
                print(f"Parameter {target_param} not found in checkpoint")
                continue
            
            # Load the parameter
            tensor_data = next(load_keys(checkpoint_path, [target_param]))
            
            # Analyze chunk patterns
            analysis = analyze_tensor_chunks(tensor_data)
            
            if "error" in analysis:
                print(f"Error: {analysis['error']}")
                continue
            
            # Display results
            print(f"  Shape: {tensor_data.shape}")
            print(f"  Chunks: {analysis['num_chunks']}")
            print(f"  Mean cosine similarity: {analysis['mean_cosine_similarity']:.6f}")
            print(f"  Max cosine similarity: {analysis['max_cosine_similarity']:.6f}")
            print(f"  Min cosine similarity: {analysis['min_cosine_similarity']:.6f}")
            print(f"  Mean L2 distance: {analysis['mean_l2_distance']:.6f}")
            print(f"  Identical pairs: {analysis['identical_pairs']}/{analysis['total_pairs']} ({analysis['identical_ratio']:.2%})")
            
            # Flag concerning patterns
            if analysis['max_cosine_similarity'] > 0.99:
                print(f"HIGH SIMILARITY DETECTED!")
            if analysis['identical_pairs'] > 0:
                print(f"DENTICAL CHUNKS STILL PRESENT!")
            if analysis['mean_cosine_similarity'] > 0.9:
                print(f"ELEVATED SIMILARITY ACROSS CHUNKS!")
            
            # Store results
            results.append({
                "checkpoint": Path(checkpoint_path).name,
                "checkpoint_path": checkpoint_path,
                **analysis
            })
            
        except Exception as e:
            print(f" Error analyzing checkpoint: {e}")
            continue
    
    # Summary analysis
    print(f"\n{'='*60}")
    print(f"EVOLUTION SUMMARY")
    print(f"{'='*60}")
    
    if not results:
        print("No successful analyses to summarize")
        return
    
    print(f"Successfully analyzed: {len(results)} checkpoints")
    
    # Show evolution of key metrics
    print(f"\nEvolution of similarity metrics:")
    print(f"{'Checkpoint':<20} {'Mean Cos Sim':<15} {'Identical Pairs':<15} {'Status'}")
    print("-" * 70)
    
    for result in results:
        status = "BUG" if result['identical_pairs'] > 0 else "OK"
        if result['mean_cosine_similarity'] > 0.9 and result['identical_pairs'] == 0:
            status = "ELEVATED"
        
        print(f"{result['checkpoint']:<20} {result['mean_cosine_similarity']:<15.6f} {result['identical_pairs']:<15} {status}")
    
    # Check for improvement over time
    if len(results) > 1:
        first_result = results[0]
        last_result = results[-1]
        
        print(f"\nChange from first to last checkpoint:")
        print(f"  Mean cosine similarity: {first_result['mean_cosine_similarity']:.6f} → {last_result['mean_cosine_similarity']:.6f}")
        print(f"  Identical pairs: {first_result['identical_pairs']} → {last_result['identical_pairs']}")
        
        if last_result['identical_pairs'] == 0 and first_result['identical_pairs'] > 0:
            print(f"Identical chunks have diverged through training!")
        elif last_result['identical_pairs'] > 0:
            print(f"Identical chunks still persist after training!")
        
        # Calculate divergence rate
        if first_result['identical_pairs'] > 0:
            divergence_rate = (first_result['identical_pairs'] - last_result['identical_pairs']) / first_result['identical_pairs']
            print(f"  Divergence rate: {divergence_rate:.2%}")
    
    # Generate visualizations
    create_divergence_visualizations(results)
    
    # Create detailed pair comparison if we have multiple checkpoints
    if len(results) > 1:
        create_pair_evolution_analysis(results)


def extract_step_number_from_path(path: str) -> int:
    """Extract step number from checkpoint path."""
    try:
        step_part = [part for part in path.split('/') if part.startswith('step')][0]
        return int(step_part.replace('step', ''))
    except:
        return 0


def create_divergence_visualizations(results: List[Dict[str, Any]]):
    """
    Create comprehensive visualizations showing the evolution of training patterns.
    
    Args:
        results: List of analysis results from checkpoints
    """
    if not results:
        print("No results to visualize")
        return
    
    # Extract step numbers and metrics
    steps = [extract_step_number_from_path(r['checkpoint_path']) for r in results]
    identical_pairs = [r['identical_pairs'] for r in results]
    identical_ratios = [r['identical_ratio'] for r in results]
    mean_cos_sim = [r['mean_cosine_similarity'] for r in results]
    max_cos_sim = [r['max_cosine_similarity'] for r in results]
    mean_l2_dist = [r['mean_l2_distance'] for r in results]
    
    # Create timestamp for unique filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create the main divergence plot
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training Pattern Evolution Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: Identical Pairs Divergence
    ax1.plot(steps, identical_pairs, 'ro-', linewidth=2, markersize=8, label='Identical Pairs')
    ax1.fill_between(steps, identical_pairs, alpha=0.3, color='red')
    ax1.set_xlabel('Training Step')
    ax1.set_ylabel('Number of Identical Pairs')
    ax1.set_title('Identical Pairs Divergence Over Training', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Add annotations for key points
    if len(identical_pairs) > 1:
        start_pairs = identical_pairs[0]
        end_pairs = identical_pairs[-1]
        ax1.annotate(f'Start: {start_pairs}', xy=(steps[0], start_pairs), 
                    xytext=(steps[0], start_pairs + max(identical_pairs) * 0.1),
                    arrowprops=dict(arrowstyle='->', color='blue'))
        ax1.annotate(f'End: {end_pairs}', xy=(steps[-1], end_pairs), 
                    xytext=(steps[-1], end_pairs + max(identical_pairs) * 0.1),
                    arrowprops=dict(arrowstyle='->', color='green'))
    
    # Plot 2: Identical Pairs Ratio
    ax2.plot(steps, identical_ratios, 'go-', linewidth=2, markersize=8, label='Identical Ratio')
    ax2.fill_between(steps, identical_ratios, alpha=0.3, color='green')
    ax2.set_xlabel('Training Step')
    ax2.set_ylabel('Identical Pairs Ratio')
    ax2.set_title('Identical Pairs Ratio Over Training', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_ylim(0, max(identical_ratios) * 1.1 if identical_ratios else 1)
    
    # Plot 3: Cosine Similarity Evolution
    ax3.plot(steps, mean_cos_sim, 'bo-', linewidth=2, markersize=6, label='Mean Cosine Similarity')
    ax3.plot(steps, max_cos_sim, 'b^--', linewidth=2, markersize=6, label='Max Cosine Similarity')
    ax3.axhline(y=0.99, color='red', linestyle='--', alpha=0.7, label='High Similarity Threshold')
    ax3.axhline(y=0.9, color='orange', linestyle='--', alpha=0.7, label='Elevated Similarity Threshold')
    ax3.set_xlabel('Training Step')
    ax3.set_ylabel('Cosine Similarity')
    ax3.set_title('Cosine Similarity Evolution', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    ax3.set_ylim(0, 1.05)
    
    # Plot 4: L2 Distance Evolution
    ax4.plot(steps, mean_l2_dist, 'mo-', linewidth=2, markersize=8, label='Mean L2 Distance')
    ax4.fill_between(steps, mean_l2_dist, alpha=0.3, color='magenta')
    ax4.set_xlabel('Training Step')
    ax4.set_ylabel('L2 Distance')
    ax4.set_title('L2 Distance Evolution', fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    plt.tight_layout()
    
    # Save the plot
    output_filename = f"training_pattern_evolution_{timestamp}.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved as: {output_filename}")
    
    # Create a focused divergence plot
    fig2, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Main divergence plot with enhanced styling
    line1 = ax.plot(steps, identical_pairs, 'ro-', linewidth=3, markersize=10, 
                   label='Identical Pairs Count', zorder=3)
    ax.fill_between(steps, identical_pairs, alpha=0.3, color='red', zorder=1)
    
    # Add secondary y-axis for ratio
    ax2 = ax.twinx()
    line2 = ax2.plot(steps, identical_ratios, 'bs-', linewidth=3, markersize=10, 
                    label='Identical Pairs Ratio', zorder=3)
    ax2.fill_between(steps, identical_ratios, alpha=0.2, color='blue', zorder=1)
    
    # Styling
    ax.set_xlabel('Training Step', fontsize=14, fontweight='bold')
    ax.set_ylabel('Number of Identical Pairs', fontsize=14, fontweight='bold', color='red')
    ax2.set_ylabel('Identical Pairs Ratio', fontsize=14, fontweight='bold', color='blue')
    ax.set_title('Identical Pairs Divergence Analysis', fontsize=16, fontweight='bold')
    
    # Color the y-axis labels
    ax.tick_params(axis='y', labelcolor='red')
    ax2.tick_params(axis='y', labelcolor='blue')
    
    # Add grid and legend
    ax.grid(True, alpha=0.3)
    
    # Combine legends
    lines = line1 + line2
    labels = [str(l.get_label()) for l in lines]
    ax.legend(lines, labels, loc='upper right', fontsize=12)
    
    # Add annotations for significant points
    if len(identical_pairs) > 1:
        # Calculate divergence statistics
        total_divergence = identical_pairs[0] - identical_pairs[-1]
        divergence_rate = total_divergence / identical_pairs[0] if identical_pairs[0] > 0 else 0
        
        # Add text box with statistics
        textstr = f'Initial Identical Pairs: {identical_pairs[0]}\n'
        textstr += f'Final Identical Pairs: {identical_pairs[-1]}\n'
        textstr += f'Total Divergence: {total_divergence}\n'
        textstr += f'Divergence Rate: {divergence_rate:.2%}'
        
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=11,
                verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    
    # Save the focused plot
    divergence_filename = f"identical_pairs_divergence_{timestamp}.png"
    plt.savefig(divergence_filename, dpi=300, bbox_inches='tight')
    print(f"Divergence analysis saved as: {divergence_filename}")
    
    # Show summary statistics
    print(f"\nDIVERGENCE ANALYSIS SUMMARY")
    print(f"="*50)
    if identical_pairs:
        print(f"Initial identical pairs: {identical_pairs[0]}")
        print(f"Final identical pairs: {identical_pairs[-1]}")
        if len(identical_pairs) > 1:
            total_divergence = identical_pairs[0] - identical_pairs[-1]
            divergence_rate = total_divergence / identical_pairs[0] if identical_pairs[0] > 0 else 0
            print(f"Total pairs diverged: {total_divergence}")
            print(f"Divergence rate: {divergence_rate:.2%}")
            
            # Find steepest divergence period
            max_drop = 0
            max_drop_period = None
            for i in range(len(identical_pairs) - 1):
                drop = identical_pairs[i] - identical_pairs[i + 1]
                if drop > max_drop:
                    max_drop = drop
                    max_drop_period = (steps[i], steps[i + 1])
            
            if max_drop_period:
                print(f"Steepest divergence: {max_drop} pairs between steps {max_drop_period[0]} and {max_drop_period[1]}")
    
    plt.show()


def create_pair_evolution_analysis(results: List[Dict[str, Any]]):
    """
    Create detailed analysis of how specific tensor chunk pairs evolve over training.
    
    Args:
        results: List of analysis results from checkpoints
    """
    
    # Extract step numbers
    steps = [extract_step_number_from_path(r['checkpoint_path']) for r in results]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Find pairs that were initially identical in the first checkpoint
    first_result = results[0]
    if 'chunks' not in first_result:
        print("No detailed chunk data available for pair analysis")
        return
    
    first_chunks = first_result['chunks']
    if len(first_chunks) < 2:
        print("Not enough chunks for pair analysis")
        return
    
    # Find the first identical pair in the initial checkpoint
    target_pair = None
    for i in range(len(first_chunks)):
        for j in range(i + 1, len(first_chunks)):
            if torch.equal(first_chunks[i], first_chunks[j]):
                target_pair = (i, j)
                break
        if target_pair:
            break
    
    if not target_pair:
        print("No identical pairs found in first checkpoint")
        return
    
    chunk_i, chunk_j = target_pair
    print(f"Tracking evolution of chunk pair ({chunk_i}, {chunk_j})")
    
    # Collect evolution data for this specific pair
    pair_cosine_similarities = []
    pair_l2_distances = []
    chunk_i_norms = []
    chunk_j_norms = []
    chunk_i_means = []
    chunk_j_means = []
    chunk_i_stds = []
    chunk_j_stds = []
    
    for result in results:
        if 'chunks' not in result or len(result['chunks']) <= max(chunk_i, chunk_j):
            print(f"⚠️  Skipping checkpoint {result['checkpoint']} - insufficient chunks")
            continue
        
        chunks = result['chunks']
        chunk_a = chunks[chunk_i]
        chunk_b = chunks[chunk_j]
        
        # Calculate metrics for this specific pair
        cos_sim = F.cosine_similarity(chunk_a.unsqueeze(0), chunk_b.unsqueeze(0)).item()
        l2_dist = torch.norm(chunk_a - chunk_b).item()
        
        pair_cosine_similarities.append(cos_sim)
        pair_l2_distances.append(l2_dist)
        
        # Individual chunk statistics
        chunk_i_norms.append(torch.norm(chunk_a).item())
        chunk_j_norms.append(torch.norm(chunk_b).item())
        chunk_i_means.append(chunk_a.mean().item())
        chunk_j_means.append(chunk_b.mean().item())
        chunk_i_stds.append(chunk_a.std().item())
        chunk_j_stds.append(chunk_b.std().item())
    
    # Create comprehensive pair evolution visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Detailed Evolution Analysis: Chunk Pair ({chunk_i}, {chunk_j})', 
                fontsize=16, fontweight='bold')
    
    # Plot 1: Cosine Similarity Evolution
    ax1.plot(steps[:len(pair_cosine_similarities)], pair_cosine_similarities, 
             'ro-', linewidth=3, markersize=8, label=f'Pair ({chunk_i}, {chunk_j})')
    ax1.axhline(y=1.0, color='green', linestyle='--', alpha=0.7, label='Perfect Similarity')
    ax1.axhline(y=0.99, color='red', linestyle='--', alpha=0.7, label='High Similarity')
    ax1.set_xlabel('Training Step')
    ax1.set_ylabel('Cosine Similarity')
    ax1.set_title('Cosine Similarity Evolution', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_ylim(0, 1.05)
    
    # Add annotations for key points
    if len(pair_cosine_similarities) > 1:
        start_sim = pair_cosine_similarities[0]
        end_sim = pair_cosine_similarities[-1]
        ax1.annotate(f'Start: {start_sim:.6f}', 
                    xy=(steps[0], start_sim), 
                    xytext=(steps[0], start_sim + 0.05),
                    arrowprops=dict(arrowstyle='->', color='blue'))
        ax1.annotate(f'End: {end_sim:.6f}', 
                    xy=(steps[len(pair_cosine_similarities)-1], end_sim), 
                    xytext=(steps[len(pair_cosine_similarities)-1], end_sim + 0.05),
                    arrowprops=dict(arrowstyle='->', color='green'))
    
    # Plot 2: L2 Distance Evolution
    ax2.plot(steps[:len(pair_l2_distances)], pair_l2_distances, 
             'bo-', linewidth=3, markersize=8, label=f'Pair ({chunk_i}, {chunk_j})')
    ax2.fill_between(steps[:len(pair_l2_distances)], pair_l2_distances, alpha=0.3, color='blue')
    ax2.set_xlabel('Training Step')
    ax2.set_ylabel('L2 Distance')
    ax2.set_title('L2 Distance Evolution', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Plot 3: Individual Chunk Norms
    ax3.plot(steps[:len(chunk_i_norms)], chunk_i_norms, 
             'go-', linewidth=2, markersize=6, label=f'Chunk {chunk_i} Norm')
    ax3.plot(steps[:len(chunk_j_norms)], chunk_j_norms, 
             'mo-', linewidth=2, markersize=6, label=f'Chunk {chunk_j} Norm')
    ax3.set_xlabel('Training Step')
    ax3.set_ylabel('L2 Norm')
    ax3.set_title('Individual Chunk Norms', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # Plot 4: Mean and Standard Deviation Evolution
    ax4_twin = ax4.twinx()
    
    # Means on left axis
    line1 = ax4.plot(steps[:len(chunk_i_means)], chunk_i_means, 
                     'g^-', linewidth=2, markersize=6, label=f'Chunk {chunk_i} Mean')
    line2 = ax4.plot(steps[:len(chunk_j_means)], chunk_j_means, 
                     'm^-', linewidth=2, markersize=6, label=f'Chunk {chunk_j} Mean')
    
    # Standard deviations on right axis
    line3 = ax4_twin.plot(steps[:len(chunk_i_stds)], chunk_i_stds, 
                          'gs--', linewidth=2, markersize=6, label=f'Chunk {chunk_i} Std')
    line4 = ax4_twin.plot(steps[:len(chunk_j_stds)], chunk_j_stds, 
                          'ms--', linewidth=2, markersize=6, label=f'Chunk {chunk_j} Std')
    
    ax4.set_xlabel('Training Step')
    ax4.set_ylabel('Mean', color='black')
    ax4_twin.set_ylabel('Standard Deviation', color='gray')
    ax4.set_title('Mean and Standard Deviation Evolution', fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    # Combine legends
    lines = line1 + line2 + line3 + line4
    labels = [str(l.get_label()) for l in lines]
    ax4.legend(lines, labels, loc='upper left', fontsize=10)
    
    plt.tight_layout()
    
    # Save the plot
    pair_filename = f"pair_evolution_analysis_{chunk_i}_{chunk_j}_{timestamp}.png"
    plt.savefig(pair_filename, dpi=300, bbox_inches='tight')
    print(f"📊 Pair evolution analysis saved as: {pair_filename}")
    
    # Create a focused divergence plot for this specific pair
    fig2, ax = plt.subplots(1, 1, figsize=(14, 8))
    
    # Main similarity plot
    ax.plot(steps[:len(pair_cosine_similarities)], pair_cosine_similarities, 
            'ro-', linewidth=4, markersize=12, label='Cosine Similarity', zorder=3)
    ax.fill_between(steps[:len(pair_cosine_similarities)], pair_cosine_similarities, 
                   alpha=0.3, color='red', zorder=1)
    
    # Add secondary y-axis for L2 distance
    ax2 = ax.twinx()
    ax2.plot(steps[:len(pair_l2_distances)], pair_l2_distances, 
             'bs-', linewidth=4, markersize=12, label='L2 Distance', zorder=3)
    ax2.fill_between(steps[:len(pair_l2_distances)], pair_l2_distances, 
                    alpha=0.2, color='blue', zorder=1)
    
    # Styling
    ax.set_xlabel('Training Step', fontsize=14, fontweight='bold')
    ax.set_ylabel('Cosine Similarity', fontsize=14, fontweight='bold', color='red')
    ax2.set_ylabel('L2 Distance', fontsize=14, fontweight='bold', color='blue')
    ax.set_title(f'Chunk Pair ({chunk_i}, {chunk_j}) Divergence Analysis', 
                fontsize=16, fontweight='bold')
    
    # Color the y-axis labels
    ax.tick_params(axis='y', labelcolor='red')
    ax2.tick_params(axis='y', labelcolor='blue')
    
    # Add grid and legend
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', fontsize=12)
    ax2.legend(loc='upper right', fontsize=12)
    
    # Add statistical information
    if len(pair_cosine_similarities) > 1:
        similarity_change = pair_cosine_similarities[-1] - pair_cosine_similarities[0]
        distance_change = pair_l2_distances[-1] - pair_l2_distances[0]
        
        textstr = f'Similarity Change: {similarity_change:+.6f}\n'
        textstr += f'Distance Change: {distance_change:+.6f}\n'
        textstr += f'Initial Similarity: {pair_cosine_similarities[0]:.6f}\n'
        textstr += f'Final Similarity: {pair_cosine_similarities[-1]:.6f}'
        
        props = dict(boxstyle='round', facecolor='lightblue', alpha=0.8)
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=11,
                verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    
    # Save the focused plot
    focused_filename = f"focused_pair_divergence_{chunk_i}_{chunk_j}_{timestamp}.png"
    plt.savefig(focused_filename, dpi=300, bbox_inches='tight')
    print(f"Focused pair analysis saved as: {focused_filename}")
    
    # Print detailed statistics
    print(f"\n PAIR EVOLUTION STATISTICS")
    print(f"="*50)
    print(f"Chunk pair: ({chunk_i}, {chunk_j})")
    print(f"Checkpoints analyzed: {len(pair_cosine_similarities)}")
    
    if len(pair_cosine_similarities) > 1:
        similarity_change = pair_cosine_similarities[-1] - pair_cosine_similarities[0]
        distance_change = pair_l2_distances[-1] - pair_l2_distances[0]
        
        print(f"Initial cosine similarity: {pair_cosine_similarities[0]:.6f}")
        print(f"Final cosine similarity: {pair_cosine_similarities[-1]:.6f}")
        print(f"Similarity change: {similarity_change:+.6f}")
        print(f"Initial L2 distance: {pair_l2_distances[0]:.6f}")
        print(f"Final L2 distance: {pair_l2_distances[-1]:.6f}")
        print(f"Distance change: {distance_change:+.6f}")
        

        if abs(similarity_change) > 0.001:  
            if similarity_change < 0:
                print("Chunks have diverged (similarity decreased)")
            else:
                print("Chunks have converged (similarity increased)")
        else:
            print("Minimal change in similarity")
    
    plt.show()


def find_available_checkpoints(base_path: str = "/data/input/amanr") -> List[str]:
    """
    Find available checkpoints in the directory structure.
    
    Args:
        base_path: Base directory to search for checkpoints
        
    Returns:
        List of checkpoint paths sorted by step number
    """
    print("SCANNING FOR AVAILABLE CHECKPOINTS")
    print("="*50)
    
    checkpoint_paths = []
    
    # Look for step directories
    step_dirs = glob.glob(os.path.join(base_path, "**/step*"), recursive=True)
    
    for step_dir in step_dirs:
        # Check if it contains a model_and_optim subdirectory
        model_optim_dir = os.path.join(step_dir, "model_and_optim")
        if os.path.exists(model_optim_dir):
            checkpoint_paths.append(model_optim_dir)
            print(f"  Found: {model_optim_dir}")
    
    # Sort by step number
    checkpoint_paths.sort(key=extract_step_number_from_path)
    
    print(f"\nFound {len(checkpoint_paths)} checkpoints")
    return checkpoint_paths


def main():
    """Main function to run the training pattern analysis."""
    print("OLMO-CORE TRAINING PATTERN EVOLUTION ANALYSIS")
    print("="*70)
    
    # Find available checkpoints
    checkpoint_paths = find_available_checkpoints()
    
    if not checkpoint_paths:
        print("No checkpoints found!")
        return
    
    # Analyze every 10th checkpoint or at most 10 checkpoints
    selected_checkpoints = []
    if len(checkpoint_paths) <= 10:
        selected_checkpoints = checkpoint_paths
    else:
        # Sample checkpoints: step0, then every 10k steps, then final
        step0_checkpoints = [p for p in checkpoint_paths if 'step0' in p or 'step1000' in p]
        step10k_checkpoints = [p for p in checkpoint_paths if any(f'step{i}0000' in p for i in range(1, 10))]
        final_checkpoints = checkpoint_paths[-3:]  # Last 3 checkpoints
        
        selected_checkpoints = step0_checkpoints + step10k_checkpoints + final_checkpoints
        # Remove duplicates while preserving order
        seen = set()
        selected_checkpoints = [p for p in selected_checkpoints if not (p in seen or seen.add(p))]
    
    print(f"\nSelected {len(selected_checkpoints)} checkpoints for analysis")
    
    # Analyze the evolution
    analyze_checkpoint_evolution(selected_checkpoints)


if __name__ == "__main__":
    main()