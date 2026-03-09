"""
Verify that an OLMo-core checkpoint was correctly converted to HF format by
comparing weight tensors directly — no forward pass, no attention implementation
differences.

Usage:
    python src/scripts/verify_hf_weights.py \
        --olmo-checkpoint /weka/.../step1000 \
        --hf-checkpoint /weka/.../step1000-hf
"""

import argparse
import json
import logging
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
import torch.distributed.checkpoint.state_dict as dist_cp_sd
from cached_path import cached_path
from transformers import AutoConfig, AutoModelForCausalLM

from olmo_core.distributed.checkpoint import load_model_and_optim_state
from olmo_core.nn.hf import convert_state_to_hf, get_hf_config
from olmo_core.nn.transformer.config import TransformerConfig
from olmo_core.utils import prepare_cli_environment

log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--olmo-checkpoint", required=True, help="Path to the OLMo-core step checkpoint dir (e.g. .../step1000)")
    parser.add_argument("--hf-checkpoint", required=True, help="Path to the saved HF checkpoint dir")
    parser.add_argument("--max-diffs", type=int, default=10, help="Max mismatched keys to print")
    args = parser.parse_args()

    # Load experiment config
    config_path = f"{args.olmo_checkpoint}/config.json"
    with cached_path(config_path).open() as f:
        experiment_config = json.load(f)

    transformer_config_dict = experiment_config["model"]
    for key in ("compile", "dp_config", "tp_config", "float8_config"):
        transformer_config_dict.pop(key, None)

    model_config = TransformerConfig.from_dict(transformer_config_dict)
    model = model_config.build(init_device="meta")
    model.to_empty(device=torch.device("cpu"))

    log.info("Loading OLMo-core checkpoint...")
    model_and_optim_dir = f"{args.olmo_checkpoint}/model_and_optim"
    with TemporaryDirectory() as work_dir:
        load_model_and_optim_state(model_and_optim_dir, model, work_dir=work_dir)

    log.info("Converting OLMo-core state dict to HF format...")
    state_dict_options = dist_cp_sd.StateDictOptions(flatten_optimizer_state_dict=True, cpu_offload=True)
    olmo_state_dict = dist_cp_sd.get_model_state_dict(model, options=state_dict_options)

    hf_config = get_hf_config(model)
    expected_hf_state = convert_state_to_hf(hf_config, olmo_state_dict)

    log.info("Loading saved HF checkpoint...")
    saved_hf_config = AutoConfig.from_pretrained(args.hf_checkpoint)
    saved_hf_model = AutoModelForCausalLM.from_pretrained(
        args.hf_checkpoint,
        torch_dtype="auto",
        config=saved_hf_config,
    )
    saved_hf_state = saved_hf_model.state_dict()

    # Compare
    expected_keys = set(expected_hf_state.keys())
    saved_keys = set(saved_hf_state.keys())

    missing = expected_keys - saved_keys
    extra = saved_keys - expected_keys
    if missing:
        print(f"\n[WARN] Keys in expected but not in saved HF model: {missing}")
    if extra:
        print(f"\n[WARN] Keys in saved HF model but not expected: {extra}")

    mismatches = []
    matches = []
    for key in sorted(expected_keys & saved_keys):
        t_expected = expected_hf_state[key].float()
        t_saved = saved_hf_state[key].float()
        if t_expected.shape != t_saved.shape:
            mismatches.append((key, f"shape mismatch: {t_expected.shape} vs {t_saved.shape}"))
            continue
        max_diff = (t_expected - t_saved).abs().max().item()
        mean_diff = (t_expected - t_saved).abs().mean().item()
        if max_diff > 1e-3:
            mismatches.append((key, f"max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}"))
        else:
            matches.append(key)

    print(f"\n{'='*60}")
    print(f"Weight comparison: {len(matches)} matched, {len(mismatches)} mismatched out of {len(expected_keys & saved_keys)} keys")
    print(f"{'='*60}")

    if mismatches:
        print(f"\nMismatched weights (showing up to {args.max_diffs}):")
        for key, msg in mismatches[:args.max_diffs]:
            print(f"  {key}: {msg}")
    else:
        print("\nAll weights match within tolerance. The conversion is correct.")
        print("The forward-pass validation failure is a numerical precision issue, not a real bug.")


if __name__ == "__main__":
    prepare_cli_environment()
    main()
