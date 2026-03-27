"""
Post-training routing analysis script for A+C experiments.

Analyzes routing dynamics across checkpoints to study how data repetition
affects MoE routing behavior. Implements analysis for the 6 routing phenomena
described in the experiment plan:

1. Routing decision ossification speed
2. Router weight extremity (tracked during training via router metrics)
3. Expert "expert-ification" (expert knockout analysis)
4. Expert co-activation patterns
5. Load imbalance (tracked during training via router metrics)
6. Load balancing settings interaction (sweep dimension, not analyzed here)

Usage:
    python routing_analysis.py --checkpoint_dir /path/to/checkpoints \
        --analysis ossification knockout co_activation \
        --data_mix OLMoE_mix_0824 \
        --output_dir /path/to/results
"""

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)


def load_model_from_checkpoint(checkpoint_path: str, device: str = "cpu"):
    """Load a model from an OLMo-core checkpoint directory."""
    from olmo_core.nn.transformer import TransformerConfig
    from olmo_core.train import TrainerConfig

    config_path = os.path.join(checkpoint_path, "config.json")
    if not os.path.exists(config_path):
        # Try parent directory for config
        config_path = os.path.join(os.path.dirname(checkpoint_path), "config.json")

    if os.path.exists(config_path):
        with open(config_path) as f:
            config_dict = json.load(f)
        model_config = TransformerConfig.from_dict(config_dict.get("model", config_dict))
    else:
        raise FileNotFoundError(f"No config found at {config_path}")

    model = model_config.build(init_device=device)

    # Load weights
    weights_path = os.path.join(checkpoint_path, "model.pt")
    if os.path.exists(weights_path):
        state_dict = torch.load(weights_path, map_location=device)
        model.load_state_dict(state_dict)
    else:
        log.warning(f"No model weights found at {weights_path}")

    model.eval()
    return model


def get_routing_decisions(model, input_ids: torch.Tensor) -> Dict[int, Dict[str, torch.Tensor]]:
    """
    Run a forward pass and collect routing decisions from all MoE layers.

    Returns a dict mapping layer_idx -> {
        "expert_indices": (batch, seq, top_k) for each router group,
        "expert_weights": (batch, seq, top_k) for each router group,
        "logits": (batch, seq, num_experts) for each router group,
    }
    """
    routing_info = {}

    # Register hooks on router forward methods
    hooks = []
    for block_key, block in model.blocks.items():
        if not hasattr(block, 'feed_forward_moe'):
            continue

        layer_idx = int(block_key)
        routing_info[layer_idx] = {}
        moe = block.feed_forward_moe

        for router_idx, router in enumerate(moe.routers_list):
            key = f"router_{router_idx}"
            routing_info[layer_idx][key] = {}

            def make_hook(info_dict, r_idx, r):
                def hook_fn(module, input, output):
                    expert_weights, expert_indices, batch_size_per_expert, _ = output
                    info_dict[f"router_{r_idx}"] = {
                        "expert_indices": expert_indices.detach().cpu(),
                        "expert_weights": expert_weights.detach().cpu(),
                    }
                    # Also capture logits
                    with torch.no_grad():
                        x = input[0]
                        logits = module.get_expert_logits(x).detach().cpu()
                        info_dict[f"router_{r_idx}"]["logits"] = logits
                return hook_fn

            h = router.register_forward_hook(make_hook(routing_info[layer_idx], router_idx, router))
            hooks.append(h)

    with torch.no_grad():
        model(input_ids)

    for h in hooks:
        h.remove()

    return routing_info


def analyze_ossification(
    checkpoint_dirs: List[str],
    eval_input_ids: torch.Tensor,
    device: str = "cpu",
) -> Dict[str, list]:
    """
    Phenomenon #1: Routing decision ossification speed.

    For each consecutive pair of checkpoints, compute the fraction of tokens
    whose top-1 expert assignment stays the same.
    """
    log.info("=== Analyzing routing decision ossification ===")

    results = {"steps": [], "stability_rates": {}}
    prev_decisions = None

    for ckpt_dir in sorted(checkpoint_dirs):
        step = _extract_step(ckpt_dir)
        log.info(f"  Loading checkpoint at step {step}...")

        model = load_model_from_checkpoint(ckpt_dir, device=device)
        routing = get_routing_decisions(model, eval_input_ids.to(device))
        del model
        torch.cuda.empty_cache() if device != "cpu" else None

        # Extract top-1 decisions per layer per router
        current_decisions = {}
        for layer_idx, layer_routing in routing.items():
            for router_key, router_data in layer_routing.items():
                key = f"layer{layer_idx}_{router_key}"
                # Top-1 expert for each token
                current_decisions[key] = router_data["expert_indices"][:, :, 0]

        if prev_decisions is not None:
            results["steps"].append(step)
            for key in current_decisions:
                if key not in results["stability_rates"]:
                    results["stability_rates"][key] = []
                # Fraction of tokens routed to same expert as previous checkpoint
                same = (current_decisions[key] == prev_decisions[key]).float().mean().item()
                results["stability_rates"][key].append(same)
                log.info(f"    {key}: stability = {same:.4f}")

        prev_decisions = current_decisions

    return results


def analyze_expert_knockout(
    checkpoint_path: str,
    eval_input_ids: torch.Tensor,
    labels: torch.Tensor,
    device: str = "cpu",
) -> Dict[str, Dict[int, float]]:
    """
    Phenomenon #3: Expert "expert-ification".

    Disable each expert one at a time and measure the increase in loss.
    More specialized experts -> larger loss increase when knocked out.
    """
    log.info("=== Analyzing expert knockout ===")

    model = load_model_from_checkpoint(checkpoint_path, device=device)

    # Get baseline loss
    with torch.no_grad():
        baseline_output = model(eval_input_ids.to(device), labels=labels.to(device))
        if hasattr(baseline_output, 'loss'):
            baseline_loss = baseline_output.loss.item()
        else:
            baseline_loss = baseline_output.item()
    log.info(f"  Baseline loss: {baseline_loss:.4f}")

    results = {"baseline_loss": baseline_loss, "knockout_losses": {}}

    for block_key, block in model.blocks.items():
        if not hasattr(block, 'feed_forward_moe'):
            continue

        layer_idx = int(block_key)
        moe = block.feed_forward_moe

        for router_idx, (router, experts) in enumerate(
            zip(moe.routers_list, moe.experts_list)
        ):
            key = f"layer{layer_idx}_router{router_idx}"
            results["knockout_losses"][key] = {}

            num_experts = router.num_experts
            for expert_id in range(num_experts):
                # Zero out the expert's weights temporarily
                original_weights = {}
                expert_mlp = experts.mlp
                for name, param in expert_mlp.named_parameters():
                    if param.dim() >= 2 and param.shape[0] >= num_experts:
                        original_weights[name] = param.data[expert_id].clone()
                        param.data[expert_id].zero_()

                with torch.no_grad():
                    ko_output = model(eval_input_ids.to(device), labels=labels.to(device))
                    if hasattr(ko_output, 'loss'):
                        ko_loss = ko_output.loss.item()
                    else:
                        ko_loss = ko_output.item()

                # Restore weights
                for name, param in expert_mlp.named_parameters():
                    if name in original_weights:
                        param.data[expert_id] = original_weights[name]

                loss_increase = ko_loss - baseline_loss
                results["knockout_losses"][key][expert_id] = {
                    "loss": ko_loss,
                    "loss_increase": loss_increase,
                }

            log.info(f"  {key}: max loss increase = "
                     f"{max(v['loss_increase'] for v in results['knockout_losses'][key].values()):.4f}")

    del model
    return results


def analyze_co_activation(
    checkpoint_path: str,
    eval_input_ids: torch.Tensor,
    device: str = "cpu",
) -> Dict[str, Dict[str, float]]:
    """
    Phenomenon #4: Expert co-activation patterns.

    For top-k > 1, measure how often pairs of experts are co-activated
    for the same token. Compute a co-activation matrix and summary stats.
    """
    log.info("=== Analyzing expert co-activation patterns ===")

    model = load_model_from_checkpoint(checkpoint_path, device=device)
    routing = get_routing_decisions(model, eval_input_ids.to(device))
    del model

    results = {}

    for layer_idx, layer_routing in routing.items():
        for router_key, router_data in layer_routing.items():
            key = f"layer{layer_idx}_{router_key}"
            indices = router_data["expert_indices"]  # (batch, seq, top_k)
            top_k = indices.shape[-1]

            if top_k < 2:
                log.info(f"  {key}: top_k=1, skipping co-activation analysis")
                continue

            # Flatten batch and seq dims
            flat_indices = indices.view(-1, top_k)  # (N, top_k)
            num_experts = indices.max().item() + 1

            # Build co-activation matrix
            coact_matrix = torch.zeros(num_experts, num_experts)
            for i in range(top_k):
                for j in range(i + 1, top_k):
                    pairs = torch.stack([flat_indices[:, i], flat_indices[:, j]], dim=-1)
                    for ei, ej in pairs:
                        coact_matrix[ei, ej] += 1
                        coact_matrix[ej, ei] += 1

            # Normalize
            total = coact_matrix.sum()
            if total > 0:
                coact_matrix = coact_matrix / total

            # Summary stats
            # Entropy of co-activation distribution (lower = more rigid patterns)
            flat_coact = coact_matrix[coact_matrix > 0]
            coact_entropy = -(flat_coact * flat_coact.log()).sum().item()

            # Top-k most common pairs
            upper_tri = torch.triu(coact_matrix, diagonal=1)
            top_pairs_flat = upper_tri.flatten().topk(min(10, upper_tri.numel()))
            top_pair_indices = []
            for idx in top_pairs_flat.indices:
                row = idx.item() // num_experts
                col = idx.item() % num_experts
                top_pair_indices.append((row, col, upper_tri[row, col].item()))

            results[key] = {
                "co_activation_entropy": coact_entropy,
                "top_pairs": top_pair_indices,
                "co_activation_matrix": coact_matrix.numpy().tolist(),
            }
            log.info(f"  {key}: co-activation entropy = {coact_entropy:.4f}")

    return results


def _extract_step(checkpoint_dir: str) -> int:
    """Extract the training step number from a checkpoint directory name."""
    basename = os.path.basename(checkpoint_dir.rstrip('/'))
    # Common patterns: "step-1000", "step_1000", "1000"
    for part in basename.split('-'):
        try:
            return int(part)
        except ValueError:
            continue
    for part in basename.split('_'):
        try:
            return int(part)
        except ValueError:
            continue
    return 0


def find_checkpoints(save_dir: str) -> List[str]:
    """Find all checkpoint directories sorted by step."""
    checkpoints = []
    for entry in os.listdir(save_dir):
        full_path = os.path.join(save_dir, entry)
        if os.path.isdir(full_path) and any(
            f.endswith('.pt') for f in os.listdir(full_path) if os.path.isfile(os.path.join(full_path, f))
        ):
            checkpoints.append(full_path)

    return sorted(checkpoints, key=_extract_step)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Post-training routing analysis for A+C experiments")
    parser.add_argument("--checkpoint_dir", type=str, required=True,
                        help="Directory containing training checkpoints")
    parser.add_argument("--analysis", nargs="+",
                        choices=["ossification", "knockout", "co_activation", "all"],
                        default=["all"],
                        help="Which analyses to run")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to save results (defaults to checkpoint_dir/routing_analysis)")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device to run analysis on")
    parser.add_argument("--num_eval_batches", type=int, default=4,
                        help="Number of evaluation batches to use")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size for evaluation")
    parser.add_argument("--seq_length", type=int, default=2048,
                        help="Sequence length for evaluation")

    args = parser.parse_args()

    analyses = set(args.analysis)
    if "all" in analyses:
        analyses = {"ossification", "knockout", "co_activation"}

    output_dir = args.output_dir or os.path.join(args.checkpoint_dir, "routing_analysis")
    os.makedirs(output_dir, exist_ok=True)

    # Create dummy eval data (in practice, load from the actual validation set)
    log.info("Creating evaluation data...")
    eval_input_ids = torch.randint(0, 50257, (args.batch_size, args.seq_length))
    labels = eval_input_ids.clone()
    labels[:, :-1] = eval_input_ids[:, 1:]
    labels[:, -1] = -100

    checkpoints = find_checkpoints(args.checkpoint_dir)
    if not checkpoints:
        log.error(f"No checkpoints found in {args.checkpoint_dir}")
        exit(1)
    log.info(f"Found {len(checkpoints)} checkpoints")

    all_results = {}

    if "ossification" in analyses:
        results = analyze_ossification(checkpoints, eval_input_ids, device=args.device)
        all_results["ossification"] = results
        with open(os.path.join(output_dir, "ossification.json"), "w") as f:
            json.dump(results, f, indent=2)

    if "knockout" in analyses:
        # Use the last checkpoint for knockout
        results = analyze_expert_knockout(
            checkpoints[-1], eval_input_ids, labels, device=args.device
        )
        all_results["knockout"] = results
        with open(os.path.join(output_dir, "knockout.json"), "w") as f:
            json.dump(results, f, indent=2, default=str)

    if "co_activation" in analyses:
        # Use the last checkpoint for co-activation
        results = analyze_co_activation(checkpoints[-1], eval_input_ids, device=args.device)
        all_results["co_activation"] = results
        with open(os.path.join(output_dir, "co_activation.json"), "w") as f:
            json.dump(results, f, indent=2, default=str)

    log.info(f"Results saved to {output_dir}")
