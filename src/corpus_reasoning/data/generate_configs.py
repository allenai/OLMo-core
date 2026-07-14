"""Generate Axolotl training configs from templates.

This script documents the config architecture and can regenerate all training
configs from a small set of base templates + experiment-specific overrides.
Existing configs in configs/ are the source of truth — this script is for
understanding the patterns and creating new experiment configs quickly.

Config architecture:
  - Two base models: Llama-3.2-1B, Qwen3.5-0.8B-Base
  - Two attention types: "std" (standard causal) and "chunked" (document-isolated)
  - Two adapter types: "lora" (LoRA r=16) and "fullft" (full fine-tuning)
  - Three query positions: "after", "before", "both"
  - Multiple datasets: NQ, HotpotQA, multi-HotpotQA, etc.

Standard attention configs use flash_attention + sample_packing (fast, via Axolotl).
Chunked attention configs disable both (uses custom train_chunked.py with 4D masks).

Usage:
    # List all defined experiments
    python scripts/generate_configs.py --list

    # Generate a specific config
    python scripts/generate_configs.py --name hotpotqa_std_qboth_qwen_lora

    # Generate all configs (dry run)
    python scripts/generate_configs.py --all --dry-run

    # Generate all configs (write to configs/)
    python scripts/generate_configs.py --all

    # Create a new experiment by overriding base template fields
    python scripts/generate_configs.py --base qwen_lora \\
        --set dataset_path=data/my_data.jsonl output_dir=./outputs/my-experiment \\
        --output configs/my_experiment.yml
"""

import argparse
import copy
import sys
from pathlib import Path

# Try to use yaml for clean output; fall back to manual formatting
try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False


# ═══════════════════════════════════════════════════════════════════════════════
# Base templates: shared fields for each (model, adapter) combination.
# Experiment definitions override only the fields that differ.
# ═══════════════════════════════════════════════════════════════════════════════

BASE_TEMPLATES = {
    # Llama 3.2 1B with LoRA — standard attention (flash + packing)
    "llama_lora_std": {
        "base_model": "NousResearch/Llama-3.2-1B",
        "val_set_size": 0.0,
        "wandb_project": "corpus-reasoning",
        "wandb_run_id": None,
        "wandb_watch": None,
        "wandb_log_model": None,
        "sequence_len": 8192,
        "micro_batch_size": 1,
        "gradient_accumulation_steps": 8,
        "num_epochs": 1,
        "learning_rate": 5e-4,
        "adapter": "lora",
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "lora_target_modules": ["gate_proj", "down_proj", "up_proj", "q_proj", "v_proj", "k_proj", "o_proj"],
        "sample_packing": True,
        "eval_sample_packing": True,
        "optimizer": "adamw_8bit",
        "lr_scheduler": "cosine",
        "bf16": "auto",
        "tf32": False,
        "gradient_checkpointing": True,
        "flash_attention": True,
        "logging_steps": 1,
        "saves_per_epoch": 0,
        "save_strategy": "no",
        "evals_per_epoch": 0,
        "warmup_ratio": 0.1,
        "weight_decay": 0.0,
        "deepspeed": "configs/deepspeed_zero1.json",
        "special_tokens": {"pad_token": "<|end_of_text|>"},
    },

    # Llama 3.2 1B with LoRA — chunked attention (no flash, no packing)
    "llama_lora_chunked": {
        "base_model": "NousResearch/Llama-3.2-1B",
        "val_set_size": 0.0,
        "wandb_project": "corpus-reasoning",
        "wandb_run_id": None,
        "wandb_watch": None,
        "wandb_log_model": None,
        "sequence_len": 8192,
        "micro_batch_size": 1,
        "gradient_accumulation_steps": 8,
        "num_epochs": 1,
        "learning_rate": 5e-4,
        "adapter": "lora",
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "lora_target_modules": ["gate_proj", "down_proj", "up_proj", "q_proj", "v_proj", "k_proj", "o_proj"],
        "sample_packing": False,
        "eval_sample_packing": False,
        "optimizer": "adamw_8bit",
        "lr_scheduler": "cosine",
        "bf16": "auto",
        "tf32": True,
        "gradient_checkpointing": True,
        "flash_attention": False,
        "logging_steps": 1,
        "saves_per_epoch": 0,
        "save_strategy": "no",
        "evals_per_epoch": 0,
        "warmup_ratio": 0.1,
        "weight_decay": 0.0,
        "deepspeed": "configs/deepspeed_zero1.json",
        "special_tokens": {"pad_token": "<|end_of_text|>"},
    },

    # Qwen 3.5 0.8B with LoRA — standard attention
    "qwen_lora_std": {
        "base_model": "Qwen/Qwen3.5-0.8B-Base",
        "val_set_size": 0.0,
        "wandb_project": "corpus-reasoning",
        "sequence_len": 8192,
        "micro_batch_size": 1,
        "gradient_accumulation_steps": 8,
        "num_epochs": 1,
        "learning_rate": 5e-4,
        "adapter": "lora",
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "lora_target_modules": ["gate_proj", "down_proj", "up_proj", "q_proj", "v_proj", "k_proj", "o_proj"],
        "sample_packing": False,
        "eval_sample_packing": False,
        "optimizer": "adamw_8bit",
        "lr_scheduler": "cosine",
        "bf16": "auto",
        "tf32": False,
        "gradient_checkpointing": True,
        "flash_attention": True,
        "logging_steps": 1,
        "saves_per_epoch": 0,
        "save_strategy": "no",
        "evals_per_epoch": 0,
        "warmup_ratio": 0.1,
        "weight_decay": 0.0,
        "deepspeed": "configs/deepspeed_zero1.json",
        "special_tokens": {"pad_token": "<|endoftext|>"},
    },

    # Qwen 3.5 0.8B with LoRA — chunked attention
    "qwen_lora_chunked": {
        "base_model": "Qwen/Qwen3.5-0.8B-Base",
        "val_set_size": 0.0,
        "wandb_project": "corpus-reasoning",
        "sequence_len": 8192,
        "micro_batch_size": 1,
        "gradient_accumulation_steps": 8,
        "num_epochs": 1,
        "learning_rate": 5e-4,
        "adapter": "lora",
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "lora_target_modules": ["gate_proj", "down_proj", "up_proj", "q_proj", "v_proj", "k_proj", "o_proj"],
        "sample_packing": False,
        "eval_sample_packing": False,
        "optimizer": "adamw_8bit",
        "lr_scheduler": "cosine",
        "bf16": "auto",
        "tf32": True,
        "gradient_checkpointing": True,
        "flash_attention": False,
        "logging_steps": 1,
        "saves_per_epoch": 0,
        "save_strategy": "no",
        "evals_per_epoch": 0,
        "warmup_ratio": 0.1,
        "weight_decay": 0.0,
        "deepspeed": "configs/deepspeed_zero1.json",
        "special_tokens": {"pad_token": "<|endoftext|>"},
    },

    # Qwen 3.5 0.8B full fine-tuning — standard attention
    "qwen_fullft_std": {
        "base_model": "Qwen/Qwen3.5-0.8B-Base",
        "val_set_size": 0.0,
        "wandb_project": "corpus-reasoning",
        "sequence_len": 8192,
        "micro_batch_size": 1,
        "gradient_accumulation_steps": 8,
        "num_epochs": 1,
        "learning_rate": 1e-5,
        # No adapter/lora fields for full fine-tuning
        "sample_packing": False,
        "eval_sample_packing": False,
        "optimizer": "adamw_8bit",
        "lr_scheduler": "cosine",
        "bf16": "auto",
        "tf32": True,
        "gradient_checkpointing": True,
        "flash_attention": False,
        "query_position": "both",
        "standard_attention": True,
        "train_on_inputs": False,
        "logging_steps": 1,
        "saves_per_epoch": 0,
        "save_strategy": "no",
        "evals_per_epoch": 0,
        "warmup_ratio": 0.1,
        "weight_decay": 0.0,
        "deepspeed": "configs/deepspeed_zero2.json",
        "special_tokens": {"pad_token": "<|endoftext|>"},
    },

    # Llama 3.2 1B full fine-tuning — standard attention
    "llama_fullft_std": {
        "base_model": "NousResearch/Llama-3.2-1B",
        "val_set_size": 0.0,
        "wandb_project": "corpus-reasoning",
        "sequence_len": 8192,
        "micro_batch_size": 1,
        "gradient_accumulation_steps": 8,
        "num_epochs": 1,
        "learning_rate": 1e-5,
        "sample_packing": False,
        "eval_sample_packing": False,
        "optimizer": "adamw_8bit",
        "lr_scheduler": "cosine",
        "bf16": "auto",
        "tf32": True,
        "gradient_checkpointing": True,
        "flash_attention": False,
        "query_position": "both",
        "standard_attention": True,
        "train_on_inputs": False,
        "logging_steps": 1,
        "saves_per_epoch": 0,
        "save_strategy": "no",
        "evals_per_epoch": 0,
        "warmup_ratio": 0.1,
        "weight_decay": 0.0,
        "deepspeed": "configs/deepspeed_zero2.json",
        "special_tokens": {"pad_token": "<|end_of_text|>"},
    },
}


# ═══════════════════════════════════════════════════════════════════════════════
# Experiment definitions: only the fields that differ from the base template.
# Format: (config_filename, base_template_name, overrides_dict)
# ═══════════════════════════════════════════════════════════════════════════════

EXPERIMENTS = [
    # ── NQ standard attention (Llama) ──
    ("nq_std_qafter_lora", "llama_lora_std", {
        "datasets": [{"path": "data/nq_train_k20_random_50000.jsonl", "type": "alpaca", "ds_type": "json"}],
        "dataset_prepared_path": "data/prepared_nq_std_qafter_lora",
        "output_dir": "./outputs/nq-std-qafter-lora",
        "wandb_name": "std_qafter_nq50k_lora",
    }),
    ("nq_std_qbefore_lora", "llama_lora_std", {
        "datasets": [{"path": "data/nq_train_k20_random_50000_qbefore.jsonl", "type": "alpaca", "ds_type": "json"}],
        "dataset_prepared_path": "data/prepared_nq_std_qbefore_lora",
        "output_dir": "./outputs/nq-std-qbefore-lora",
        "wandb_name": "std_qbefore_nq50k_lora",
    }),
    ("nq_std_qboth_lora", "llama_lora_std", {
        "datasets": [{"path": "data/nq_train_k20_random_50000_qboth.jsonl", "type": "alpaca", "ds_type": "json"}],
        "dataset_prepared_path": "data/prepared_nq_std_qboth_lora",
        "output_dir": "./outputs/nq-std-qboth-lora",
        "wandb_name": "std_qboth_nq50k_lora",
    }),

    # ── NQ chunked attention (Llama) ──
    ("nq_chunked_qafter_lora", "llama_lora_chunked", {
        "query_position": "after",
        "datasets": [{"path": "data/nq_train_k20_random_50000.jsonl", "type": "alpaca", "ds_type": "json"}],
        "dataset_prepared_path": "data/prepared_nq_chunked_qafter_lora",
        "output_dir": "./outputs/nq-chunked-qafter-lora",
        "wandb_name": "chunked_qafter_nq50k_lora",
    }),
    ("nq_chunked_qbefore_lora", "llama_lora_chunked", {
        "query_position": "before",
        "datasets": [{"path": "data/nq_train_k20_random_50000.jsonl", "type": "alpaca", "ds_type": "json"}],
        "dataset_prepared_path": "data/prepared_nq_chunked_qbefore_lora",
        "output_dir": "./outputs/nq-chunked-qbefore-lora",
        "wandb_name": "chunked_qbefore_nq50k_lora",
    }),
    ("nq_chunked_qboth_lora", "llama_lora_chunked", {
        "query_position": "both",
        "datasets": [{"path": "data/nq_train_k20_random_50000_qboth.jsonl", "type": "alpaca", "ds_type": "json"}],
        "dataset_prepared_path": "data/prepared_nq_chunked_qboth_lora",
        "output_dir": "./outputs/nq-chunked-qboth-lora",
        "wandb_name": "chunked_qboth_nq50k_lora",
    }),

    # ── HotpotQA standard attention (Llama, k=20) ──
    ("hotpotqa_std_qafter_lora", "llama_lora_std", {
        "sample_packing": False, "eval_sample_packing": False,
        "datasets": [{"path": "data/hotpotqa_train_k20_shuffled_bridge_72991.jsonl", "type": "alpaca", "ds_type": "json"}],
        "dataset_prepared_path": "data/prepared_hotpotqa_std_qafter_lora",
        "output_dir": "./outputs/hotpotqa-std-qafter-lora",
        "wandb_name": "std_qafter_hotpotqa73k_lora",
    }),
    ("hotpotqa_std_qbefore_lora", "llama_lora_std", {
        "sample_packing": False, "eval_sample_packing": False,
        "datasets": [{"path": "data/hotpotqa_train_k20_shuffled_bridge_72991_qbefore.jsonl", "type": "alpaca", "ds_type": "json"}],
        "dataset_prepared_path": "data/prepared_hotpotqa_std_qbefore_lora",
        "output_dir": "./outputs/hotpotqa-std-qbefore-lora",
        "wandb_name": "std_qbefore_hotpotqa73k_lora",
    }),
    ("hotpotqa_std_qboth_lora", "llama_lora_std", {
        "sample_packing": False, "eval_sample_packing": False,
        "datasets": [{"path": "data/hotpotqa_train_k20_shuffled_bridge_72991_qboth.jsonl", "type": "alpaca", "ds_type": "json"}],
        "dataset_prepared_path": "data/prepared_hotpotqa_std_qboth_lora",
        "output_dir": "./outputs/hotpotqa-std-qboth-lora",
        "wandb_name": "std_qboth_hotpotqa73k_lora",
    }),

    # ── HotpotQA chunked attention (Llama, k=20) ──
    ("hotpotqa_chunked_qafter_lora", "llama_lora_chunked", {
        "query_position": "after",
        "datasets": [{"path": "data/hotpotqa_train_k20_shuffled_bridge_72991.jsonl", "type": "alpaca", "ds_type": "json"}],
        "dataset_prepared_path": "data/prepared_hotpotqa_chunked_qafter_lora",
        "output_dir": "./outputs/hotpotqa-chunked-qafter-lora",
        "wandb_name": "chunked_qafter_hotpotqa73k_lora",
    }),
    ("hotpotqa_chunked_qbefore_lora", "llama_lora_chunked", {
        "query_position": "before",
        "datasets": [{"path": "data/hotpotqa_train_k20_shuffled_bridge_72991.jsonl", "type": "alpaca", "ds_type": "json"}],
        "dataset_prepared_path": "data/prepared_hotpotqa_chunked_qbefore_lora",
        "output_dir": "./outputs/hotpotqa-chunked-qbefore-lora",
        "wandb_name": "chunked_qbefore_hotpotqa73k_lora",
    }),
    ("hotpotqa_chunked_qboth_lora", "llama_lora_chunked", {
        "query_position": "both",
        "datasets": [{"path": "data/hotpotqa_train_k20_shuffled_bridge_72991.jsonl", "type": "alpaca", "ds_type": "json"}],
        "dataset_prepared_path": "data/prepared_hotpotqa_chunked_qboth_lora",
        "output_dir": "./outputs/hotpotqa-chunked-qboth-lora",
        "wandb_name": "chunked_qboth_hotpotqa73k_lora",
    }),

    # ── HotpotQA Qwen LoRA (k=20, 2.5k examples) ──
    ("hotpotqa_std_qboth_qwen_lora", "qwen_lora_std", {
        "datasets": [{"path": "data/hotpotqa_train_k20_shuffled_bridge_2500_qboth.jsonl", "type": "alpaca", "ds_type": "json"}],
        "dataset_prepared_path": "data/prepared_hotpotqa_std_qboth_qwen_lora",
        "output_dir": "./outputs/hotpotqa-std-qboth-qwen-lora",
        "wandb_name": "std_qboth_hotpotqa2500_qwen_lora",
    }),

    # ── HotpotQA Qwen full FT variants ──
    ("hotpotqa_std_qboth_qwen_fullft", "qwen_fullft_std", {
        "datasets": [{"path": "data/hotpotqa_train_k20_shuffled_retrieval_bridge_5000_qboth.jsonl", "type": "alpaca", "ds_type": "json"}],
        "output_dir": "./outputs/hotpotqa-std-qboth-qwen-fullft",
        "wandb_name": "std_qboth_hotpotqa_k20_5k_qwen_fullft",
    }),
    ("hotpotqa_k20_std_qboth_qwen_fullft_wd01", "qwen_fullft_std", {
        "weight_decay": 0.1,
        "datasets": [{"path": "data/hotpotqa_train_k20_shuffled_retrieval_bridge_5000_qboth.jsonl", "type": "alpaca", "ds_type": "json"}],
        "output_dir": "./outputs/hotpotqa-k20-std-qboth-qwen-fullft-wd01",
        "wandb_name": "std_qboth_hotpotqa_k20_5k_qwen_fullft_wd01",
    }),

    # ── HotpotQA Qwen chunked + dummy token ablations ──
    ("hotpotqa_k20_chunked_qbefore_qwen_lora_ad32", "qwen_lora_chunked", {
        "query_position": "before",
        "standard_attention": False,
        "train_on_inputs": False,
        "before_dummy": 0,
        "after_dummy": 32,
        "datasets": [{"path": "data/hotpotqa_train_k20_shuffled_retrieval_bridge_5000_ad32.jsonl", "type": "alpaca", "ds_type": "json"}],
        "output_dir": "./outputs/hotpotqa-k20-chunked-qbefore-qwen-lora-ad32",
        "wandb_name": "chunked_qbefore_hotpotqa_k20_5k_qwen_lora_ad32",
    }),

    # ── Context size ablations (k=2, k=4, k=10) ──
    ("hotpotqa_k2_std_qboth_qwen_fullft", "qwen_fullft_std", {
        "datasets": [{"path": "data/hotpotqa_train_k2_shuffled_retrieval_bridge_5000_qboth.jsonl", "type": "alpaca", "ds_type": "json"}],
        "output_dir": "./outputs/hotpotqa-k2-std-qboth-qwen-fullft",
        "wandb_name": "std_qboth_hotpotqa_k2_5k_qwen_fullft",
    }),
    ("hotpotqa_k4_std_qboth_qwen_lora", "qwen_lora_std", {
        "datasets": [{"path": "data/hotpotqa_train_k4_shuffled_retrieval_bridge_5000_qboth.jsonl", "type": "alpaca", "ds_type": "json"}],
        "dataset_prepared_path": "data/prepared_hotpotqa_k4_std_qboth_qwen_lora",
        "output_dir": "./outputs/hotpotqa-k4-std-qboth-qwen-lora",
        "wandb_name": "std_qboth_hotpotqa_k4_5k_qwen_lora",
    }),
    ("hotpotqa_k4_std_qboth_qwen_fullft", "qwen_fullft_std", {
        "datasets": [{"path": "data/hotpotqa_train_k4_shuffled_retrieval_bridge_5000_qboth.jsonl", "type": "alpaca", "ds_type": "json"}],
        "output_dir": "./outputs/hotpotqa-k4-std-qboth-qwen-fullft",
        "wandb_name": "std_qboth_hotpotqa_k4_5k_qwen_fullft",
    }),
]


def build_config(base_name, overrides):
    """Merge a base template with experiment-specific overrides."""
    if base_name not in BASE_TEMPLATES:
        raise ValueError(f"Unknown base template: {base_name}. "
                         f"Available: {list(BASE_TEMPLATES.keys())}")
    config = copy.deepcopy(BASE_TEMPLATES[base_name])
    config.update(overrides)
    return config


def format_yaml(config):
    """Format config dict as YAML string."""
    if HAS_YAML:
        # Custom representer to output lists inline for lora_target_modules
        class CustomDumper(yaml.SafeDumper):
            pass

        def list_representer(dumper, data):
            # Use flow style for short lists (like lora_target_modules)
            if all(isinstance(item, str) for item in data) and len(data) > 3:
                return dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=True)
            return dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=False)

        CustomDumper.add_representer(list, list_representer)
        return yaml.dump(config, Dumper=CustomDumper, default_flow_style=False, sort_keys=False)
    else:
        # Fallback: simple manual YAML formatting
        lines = []
        for k, v in config.items():
            if isinstance(v, dict):
                lines.append(f"{k}:")
                for dk, dv in v.items():
                    lines.append(f"  {dk}: {_format_value(dv)}")
            elif isinstance(v, list) and v and isinstance(v[0], dict):
                lines.append(f"{k}:")
                for item in v:
                    first = True
                    for dk, dv in item.items():
                        prefix = "  - " if first else "    "
                        lines.append(f"{prefix}{dk}: {_format_value(dv)}")
                        first = False
            elif isinstance(v, list):
                lines.append(f"{k}: [{', '.join(str(x) for x in v)}]")
            else:
                lines.append(f"{k}: {_format_value(v)}")
        return "\n".join(lines) + "\n"


def _format_value(v):
    if v is None:
        return "null"
    if isinstance(v, bool):
        return str(v).lower()
    if isinstance(v, str):
        if v in ("no",) or " " in v or v.startswith("<"):
            return f"'{v}'"
        return v
    return str(v)


def write_config(name, config, output_dir, dry_run=False):
    """Write a config to a YAML file."""
    path = Path(output_dir) / f"{name}.yml"
    content = f"# Generated by scripts/generate_configs.py — edit the generator, not this file.\n"
    content += format_yaml(config)

    if dry_run:
        print(f"  [dry-run] Would write {path}")
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    print(f"  Wrote {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate Axolotl training configs from templates")
    parser.add_argument("--list", action="store_true",
                        help="List all defined experiments")
    parser.add_argument("--all", action="store_true",
                        help="Generate all experiment configs")
    parser.add_argument("--name", type=str,
                        help="Generate a specific experiment config by name")
    parser.add_argument("--base", type=str,
                        help="Base template name for --set mode")
    parser.add_argument("--set", nargs="+", metavar="KEY=VALUE",
                        help="Override fields (used with --base)")
    parser.add_argument("--output", type=str,
                        help="Output file path (used with --base)")
    parser.add_argument("--output-dir", type=str, default="configs",
                        help="Output directory for generated configs (default: configs/)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be generated without writing files")
    args = parser.parse_args()

    if args.list:
        print(f"Base templates ({len(BASE_TEMPLATES)}):")
        for name in BASE_TEMPLATES:
            t = BASE_TEMPLATES[name]
            model = t["base_model"].split("/")[-1]
            adapter = t.get("adapter", "fullft")
            attn = "chunked" if not t.get("flash_attention", True) else "std"
            print(f"  {name:30s}  {model:20s}  {adapter:6s}  {attn}")

        print(f"\nExperiments ({len(EXPERIMENTS)}):")
        for name, base, overrides in EXPERIMENTS:
            ds = overrides.get("datasets", [{}])[0].get("path", "?")
            ds = ds.split("/")[-1][:50] if ds != "?" else "?"
            print(f"  {name:55s}  base={base:20s}  data={ds}")
        return

    if args.base:
        # Ad-hoc config generation from base + overrides
        overrides = {}
        if args.set:
            for kv in args.set:
                k, v = kv.split("=", 1)
                # Try to parse as number or bool
                if v.lower() in ("true", "false"):
                    v = v.lower() == "true"
                else:
                    try:
                        v = float(v) if "." in v else int(v)
                    except ValueError:
                        pass
                overrides[k] = v
        config = build_config(args.base, overrides)
        if args.output:
            name = Path(args.output).stem
            write_config(name, config, Path(args.output).parent, args.dry_run)
        else:
            print(format_yaml(config))
        return

    if args.name:
        for name, base, overrides in EXPERIMENTS:
            if name == args.name:
                config = build_config(base, overrides)
                write_config(name, config, args.output_dir, args.dry_run)
                return
        print(f"Error: experiment '{args.name}' not found. Use --list to see available experiments.")
        sys.exit(1)

    if args.all:
        print(f"Generating {len(EXPERIMENTS)} configs...")
        for name, base, overrides in EXPERIMENTS:
            config = build_config(base, overrides)
            write_config(name, config, args.output_dir, args.dry_run)
        print("Done.")
        return

    parser.print_help()


if __name__ == "__main__":
    main()
