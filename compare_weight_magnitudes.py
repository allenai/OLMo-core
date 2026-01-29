"""
Compare weight magnitudes between two OLMo-core model checkpoints.
Outputs a TSV summary of differences between model weights.
"""

import argparse
import os
from dataclasses import dataclass
from typing import Dict, Optional

import torch

from olmo_core.distributed.checkpoint import get_checkpoint_metadata, load_keys
from olmo_core.io import join_path, normalize_path


@dataclass
class ParameterDiff:
    """Statistics about the difference between two parameter tensors."""
    name: str
    max_diff: float
    mean_diff: float
    l2_norm: float
    num_elements: int
    shape: tuple
    any_exceeds_threshold: bool = False
    mean_exceeds_threshold: bool = False


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare weight magnitudes between two OLMo-core model checkpoints"
    )
    parser.add_argument(
        "--model1",
        type=str,
        required=True,
        help="Path to first model checkpoint",
    )
    parser.add_argument(
        "--model2",
        type=str,
        required=True,
        help="Path to second model checkpoint",
    )
    parser.add_argument(
        "--min_diff",
        type=float,
        default=None,
        help="Report all parameters where any element exceeds this magnitude difference",
    )
    parser.add_argument(
        "--top_n",
        type=int,
        default=20,
        help="Number of top parameters to report by max difference (default: 20)",
    )
    return parser.parse_args()


def load_checkpoint_state_dict(checkpoint_dir: str) -> Dict[str, torch.Tensor]:
    """Load model state dict from an OLMo-core distributed checkpoint."""
    checkpoint_dir = normalize_path(checkpoint_dir)

    # Try model_and_optim subdirectory first (standard checkpoint structure)
    model_dir = join_path(checkpoint_dir, "model_and_optim")
    try:
        metadata = get_checkpoint_metadata(model_dir)
    except FileNotFoundError:
        # Fall back to base directory
        model_dir = checkpoint_dir
        metadata = get_checkpoint_metadata(model_dir)

    # Get all model parameter keys from the metadata
    all_keys = list(metadata.state_dict_metadata.keys())
    model_keys = [k for k in all_keys if k.startswith("model.")]

    if not model_keys:
        raise ValueError(f"No model parameters found in checkpoint at {checkpoint_dir}")

    print(f"  Found {len(model_keys)} model parameters")

    # Load all model parameters
    state_dict = {}
    for key, tensor in zip(model_keys, load_keys(model_dir, model_keys)):
        # Remove "model." prefix for cleaner comparison
        clean_key = key[6:] if key.startswith("model.") else key
        state_dict[clean_key] = tensor

    return state_dict


def compute_parameter_diff(
    name: str,
    tensor1: torch.Tensor,
    tensor2: torch.Tensor,
    min_diff: Optional[float] = None,
) -> ParameterDiff:
    """Compute difference statistics between two parameter tensors."""
    diff = (tensor1.float() - tensor2.float()).abs()

    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    l2_norm = torch.linalg.vector_norm(diff).item()

    any_exceeds = False
    mean_exceeds = False
    if min_diff is not None:
        any_exceeds = max_diff > min_diff
        mean_exceeds = mean_diff > min_diff

    return ParameterDiff(
        name=name,
        max_diff=max_diff,
        mean_diff=mean_diff,
        l2_norm=l2_norm,
        num_elements=tensor1.numel(),
        shape=tuple(tensor1.shape),
        any_exceeds_threshold=any_exceeds,
        mean_exceeds_threshold=mean_exceeds,
    )


def compare_models(
    state_dict1: Dict[str, torch.Tensor],
    state_dict2: Dict[str, torch.Tensor],
    min_diff: Optional[float] = None,
) -> list[ParameterDiff]:
    """Compare two model state dicts and return difference statistics."""
    # Verify both models have the same parameters
    keys1 = set(state_dict1.keys())
    keys2 = set(state_dict2.keys())

    if keys1 != keys2:
        only_in_1 = keys1 - keys2
        only_in_2 = keys2 - keys1
        error_msg = "Model architectures do not match!\n"
        if only_in_1:
            error_msg += f"  Parameters only in model1: {sorted(only_in_1)[:10]}{'...' if len(only_in_1) > 10 else ''}\n"
        if only_in_2:
            error_msg += f"  Parameters only in model2: {sorted(only_in_2)[:10]}{'...' if len(only_in_2) > 10 else ''}\n"
        raise ValueError(error_msg)

    diffs = []
    for name in sorted(state_dict1.keys()):
        tensor1 = state_dict1[name]
        tensor2 = state_dict2[name]

        if tensor1.shape != tensor2.shape:
            raise ValueError(
                f"Shape mismatch for {name}: {tensor1.shape} vs {tensor2.shape}"
            )

        diff = compute_parameter_diff(name, tensor1, tensor2, min_diff)
        diffs.append(diff)

    return diffs


def get_model_name(path: str) -> str:
    """Extract a short model name from the path."""
    parts = path.rstrip('/').split('/')
    for part in reversed(parts):
        if part and not part.startswith('step'):
            return part
    return os.path.basename(path.rstrip('/'))


def write_diff_report(
    diffs: list[ParameterDiff],
    output_path: str,
    model1_path: str,
    model2_path: str,
    top_n: int,
    min_diff: Optional[float],
):
    """Write the diff report as a TSV file."""
    with open(output_path, 'w') as f:
        # Write header comment
        f.write(f"# Weight magnitude comparison\n")
        f.write(f"# Model 1: {model1_path}\n")
        f.write(f"# Model 2: {model2_path}\n")
        f.write(f"# Total parameters compared: {len(diffs)}\n")
        f.write(f"#\n")

        # Section 1: Top N parameters by max difference
        f.write(f"# === TOP {top_n} PARAMETERS BY MAX DIFFERENCE ===\n")
        f.write("parameter_name\tmax_diff\tmean_diff\tl2_norm\tnum_elements\tshape\n")

        sorted_by_max = sorted(diffs, key=lambda d: d.max_diff, reverse=True)
        for diff in sorted_by_max[:top_n]:
            f.write(
                f"{diff.name}\t"
                f"{diff.max_diff:.6e}\t"
                f"{diff.mean_diff:.6e}\t"
                f"{diff.l2_norm:.6e}\t"
                f"{diff.num_elements}\t"
                f"{diff.shape}\n"
            )

        # Section 2: Parameters exceeding min_diff threshold (if specified)
        if min_diff is not None:
            exceeding = [d for d in diffs if d.any_exceeds_threshold]
            f.write(f"#\n")
            f.write(f"# === PARAMETERS EXCEEDING THRESHOLD (min_diff={min_diff}) ===\n")
            f.write(f"# Total: {len(exceeding)} parameters\n")
            f.write(f"# (* = mean also exceeds threshold)\n")
            f.write("parameter_name\tmax_diff\tmean_diff\tl2_norm\tnum_elements\tshape\tmean_exceeds\n")

            sorted_exceeding = sorted(exceeding, key=lambda d: d.max_diff, reverse=True)
            for diff in sorted_exceeding:
                marker = "*" if diff.mean_exceeds_threshold else ""
                f.write(
                    f"{marker}{diff.name}\t"
                    f"{diff.max_diff:.6e}\t"
                    f"{diff.mean_diff:.6e}\t"
                    f"{diff.l2_norm:.6e}\t"
                    f"{diff.num_elements}\t"
                    f"{diff.shape}\t"
                    f"{diff.mean_exceeds_threshold}\n"
                )

        # Section 3: Summary statistics
        f.write(f"#\n")
        f.write(f"# === SUMMARY STATISTICS ===\n")

        all_max_diffs = [d.max_diff for d in diffs]
        all_mean_diffs = [d.mean_diff for d in diffs]

        f.write(f"# Global max difference: {max(all_max_diffs):.6e}\n")
        f.write(f"# Average of max differences: {sum(all_max_diffs)/len(all_max_diffs):.6e}\n")
        f.write(f"# Average of mean differences: {sum(all_mean_diffs)/len(all_mean_diffs):.6e}\n")

        # Count parameters with zero difference
        zero_diff = sum(1 for d in diffs if d.max_diff == 0)
        f.write(f"# Parameters with zero difference: {zero_diff}\n")


def main():
    args = parse_args()

    print(f"Loading model 1: {args.model1}")
    state_dict1 = load_checkpoint_state_dict(args.model1)
    print(f"  Loaded {len(state_dict1)} parameters")

    print(f"Loading model 2: {args.model2}")
    state_dict2 = load_checkpoint_state_dict(args.model2)
    print(f"  Loaded {len(state_dict2)} parameters")

    print("Comparing models...")
    diffs = compare_models(state_dict1, state_dict2, args.min_diff)

    # Generate output filename
    model1_name = get_model_name(args.model1)
    model2_name = get_model_name(args.model2)
    output_path = f"{model1_name}_{model2_name}.diff"

    print(f"Writing report to {output_path}")
    write_diff_report(
        diffs,
        output_path,
        args.model1,
        args.model2,
        args.top_n,
        args.min_diff,
    )

    # Print summary to console
    sorted_by_max = sorted(diffs, key=lambda d: d.max_diff, reverse=True)
    print(f"\nTop 5 parameters by max difference:")
    for diff in sorted_by_max[:5]:
        print(f"  {diff.name}: max={diff.max_diff:.6e}, mean={diff.mean_diff:.6e}")

    if args.min_diff is not None:
        exceeding = [d for d in diffs if d.any_exceeds_threshold]
        mean_exceeding = [d for d in diffs if d.mean_exceeds_threshold]
        print(f"\nParameters exceeding threshold ({args.min_diff}):")
        print(f"  Any element exceeds: {len(exceeding)}")
        print(f"  Mean exceeds: {len(mean_exceeding)}")

    print(f"\nFull report written to: {output_path}")


if __name__ == "__main__":
    main()
