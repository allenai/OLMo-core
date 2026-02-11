"""
Track qnorm and knorm weight changes over training checkpoints.

Given a base folder containing checkpoints at step0, step2000, ..., step34000,
this script:
  1. Compares consecutive checkpoints' qnorm/knorm weights (% increase/decrease/same).
  2. Computes the L2 norm of each qnorm/knorm parameter at every checkpoint.
  3. Writes two TSV output files: one for comparative metrics, one for magnitudes.
"""

import argparse
import os
from typing import Dict, List

import torch

from olmo_core.distributed.checkpoint import get_checkpoint_metadata, load_keys
from olmo_core.io import join_path, normalize_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Track qnorm/knorm weight changes across training checkpoints"
    )
    parser.add_argument(
        "--base_folder",
        type=str,
        required=True,
        help="Path to folder containing stepN checkpoint subdirectories",
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=0.01,
        help="Tolerance for considering a weight unchanged (default: 0.01)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for results (default: same as base_folder)",
    )
    return parser.parse_args()


def load_qknorm_state_dict(checkpoint_dir: str) -> Dict[str, torch.Tensor]:
    """Load only qnorm/knorm parameters from an OLMo-core distributed checkpoint."""
    checkpoint_dir = normalize_path(checkpoint_dir)

    model_dir = join_path(checkpoint_dir, "model_and_optim")
    try:
        metadata = get_checkpoint_metadata(model_dir)
    except FileNotFoundError:
        model_dir = checkpoint_dir
        metadata = get_checkpoint_metadata(model_dir)

    all_keys = list(metadata.state_dict_metadata.keys())
    qknorm_keys = [k for k in all_keys if k.startswith("model.") and ("q_norm" in k or "k_norm" in k)]

    if not qknorm_keys:
        raise ValueError(f"No qnorm/knorm parameters found in checkpoint at {checkpoint_dir}")

    print(f"  Found {len(qknorm_keys)} qnorm/knorm parameters")

    state_dict = {}
    for key, tensor in zip(qknorm_keys, load_keys(model_dir, qknorm_keys)):
        clean_key = key[6:] if key.startswith("model.") else key
        state_dict[clean_key] = tensor

    return state_dict


def compare_qknorm(
    state_dict_a: Dict[str, torch.Tensor],
    state_dict_b: Dict[str, torch.Tensor],
    tol: float,
) -> List[dict]:
    """Compare qnorm/knorm weights between two checkpoints.

    Returns per-parameter stats: % increased, % decreased, % same (within tol).
    Comparison is B - A, so 'increased' means the weight grew from A to B.
    """
    results = []
    for name in sorted(state_dict_a.keys()):
        t_a = state_dict_a[name].float()
        t_b = state_dict_b[name].float()
        diff = t_b - t_a
        n = t_a.numel()

        increased = (diff > tol).sum().item()
        decreased = (diff < -tol).sum().item()
        same = n - increased - decreased

        results.append({
            "name": name,
            "pct_increased": 100.0 * increased / n,
            "pct_decreased": 100.0 * decreased / n,
            "pct_same": 100.0 * same / n,
            "max_increase": diff.max().item(),
            "max_decrease": diff.min().item(),
            "mean_change": diff.mean().item(),
        })
    return results


def compute_norms(state_dict: Dict[str, torch.Tensor]) -> List[dict]:
    """Compute L2 norm and mean absolute value of each qnorm/knorm parameter."""
    results = []
    for name in sorted(state_dict.keys()):
        t = state_dict[name].float()
        results.append({
            "name": name,
            "l2_norm": torch.linalg.vector_norm(t).item(),
            "mean_abs": t.abs().mean().item(),
            "max_abs": t.abs().max().item(),
            "min_abs": t.abs().min().item(),
        })
    return results


def get_run_name(base_folder: str) -> str:
    """Extract a short run name from the base folder path."""
    parts = base_folder.rstrip("/").split("/")
    for part in reversed(parts):
        if part:
            return part
    return "run"


def write_comparative_report(
    all_comparisons: List[dict],
    output_path: str,
    base_folder: str,
    tol: float,
):
    """Write the step-to-step comparative metrics TSV."""
    with open(output_path, "w") as f:
        f.write(f"# qnorm/knorm step-to-step comparison\n")
        f.write(f"# Base folder: {base_folder}\n")
        f.write(f"# Tolerance: {tol}\n")
        f.write(f"# Comparison direction: stepB - stepA (increased = weight grew)\n")
        f.write(f"#\n")
        f.write(
            "step_a\tstep_b\tparameter_name\t"
            "pct_increased\tpct_decreased\tpct_same\t"
            "max_increase\tmax_decrease\tmean_change\n"
        )

        for entry in all_comparisons:
            step_a = entry["step_a"]
            step_b = entry["step_b"]
            for param in entry["params"]:
                f.write(
                    f"{step_a}\t{step_b}\t{param['name']}\t"
                    f"{param['pct_increased']:.2f}\t"
                    f"{param['pct_decreased']:.2f}\t"
                    f"{param['pct_same']:.2f}\t"
                    f"{param['max_increase']:.6e}\t"
                    f"{param['max_decrease']:.6e}\t"
                    f"{param['mean_change']:.6e}\n"
                )


def write_magnitude_report(
    all_norms: List[dict],
    output_path: str,
    base_folder: str,
):
    """Write the per-step magnitude TSV."""
    with open(output_path, "w") as f:
        f.write(f"# qnorm/knorm parameter magnitudes over training\n")
        f.write(f"# Base folder: {base_folder}\n")
        f.write(f"#\n")
        f.write("step\tparameter_name\tl2_norm\tmean_abs\tmax_abs\tmin_abs\n")

        for entry in all_norms:
            step = entry["step"]
            for param in entry["params"]:
                f.write(
                    f"{step}\t{param['name']}\t"
                    f"{param['l2_norm']:.6e}\t"
                    f"{param['mean_abs']:.6e}\t"
                    f"{param['max_abs']:.6e}\t"
                    f"{param['min_abs']:.6e}\n"
                )


def main():
    args = parse_args()
    base_folder = args.base_folder.rstrip("/")
    tol = args.tol

    steps = list(range(0, 34001, 2000))

    # Verify checkpoint directories exist
    missing = []
    for step in steps:
        ckpt_dir = join_path(base_folder, f"step{step}")
        if not os.path.isdir(ckpt_dir):
            missing.append(step)
    if missing:
        print(f"Warning: missing checkpoint directories for steps: {missing}")
        steps = [s for s in steps if s not in missing]

    print(f"Processing {len(steps)} checkpoints: step{steps[0]} to step{steps[-1]}")

    all_norms = []
    all_comparisons = []
    prev_state_dict = None
    prev_step = None

    for step in steps:
        ckpt_dir = join_path(base_folder, f"step{step}")
        print(f"\nLoading step{step}: {ckpt_dir}")
        state_dict = load_qknorm_state_dict(ckpt_dir)

        # Compute norms at this step
        norms = compute_norms(state_dict)
        all_norms.append({"step": step, "params": norms})

        # Compare with previous step
        if prev_state_dict is not None:
            print(f"  Comparing step{prev_step} -> step{step}")
            comparison = compare_qknorm(prev_state_dict, state_dict, tol)
            all_comparisons.append({
                "step_a": prev_step,
                "step_b": step,
                "params": comparison,
            })

        prev_state_dict = state_dict
        prev_step = step

    # Determine output paths
    run_name = get_run_name(base_folder)
    output_dir = args.output_dir or base_folder
    os.makedirs(output_dir, exist_ok=True)

    comp_path = os.path.join(output_dir, f"{run_name}_qknorm_comparisons.tsv")
    mag_path = os.path.join(output_dir, f"{run_name}_qknorm_magnitudes.tsv")

    print(f"\nWriting comparative report to {comp_path}")
    write_comparative_report(all_comparisons, comp_path, base_folder, tol)

    print(f"Writing magnitude report to {mag_path}")
    write_magnitude_report(all_norms, mag_path, base_folder)

    # Print a brief summary
    print(f"\nDone. Processed {len(steps)} checkpoints, {len(all_comparisons)} comparisons.")
    if all_norms:
        param_names = [p["name"] for p in all_norms[0]["params"]]
        print(f"Tracked {len(param_names)} qnorm/knorm parameters.")


if __name__ == "__main__":
    main()
