"""Generate attention-pattern sweep configs for HotpotQA and Contradiction.

Emits 8 configs per task family, all varying only in attention_pattern (+ its
params). The per-family template captures dataset / task / qpos / seq_len /
lr / adapter choices.

Usage:
    python -m scripts.utils.generate_sweep_configs

Overwrites existing files. Files that already exist with identical content are
a no-op.
"""

from pathlib import Path

import yaml


CONFIGS_DIR = Path("configs")


# The 8 attention-pattern variants for each task family. "std" is sent to the
# axolotl path (standard_attention: true); everything else goes through the
# chunked trainer.
PATTERN_VARIANTS = [
    # tag,              pattern,              extra_keys
    ("std",             "standard",           {"standard_attention": True}),
    ("chunked",         "chunked",            {"standard_attention": False}),
    ("docwin_k1",       "doc_window",         {"standard_attention": False, "doc_window_k": 1}),
    ("docwin_k2",       "doc_window",         {"standard_attention": False, "doc_window_k": 2}),
    ("docwin_k4",       "doc_window",         {"standard_attention": False, "doc_window_k": 4}),
    ("anchor",          "last_token_anchor",  {"standard_attention": False}),
    ("tokwin_w512",     "token_window",       {"standard_attention": False, "token_window_w": 512}),
    ("bigbird",         "bigbird",            {"standard_attention": False, "doc_window_k": 1,
                                               "num_random_doc_edges": 2, "random_seed": 42}),
]


COMMON_LORA_BLOCK = {
    "adapter": "lora",
    "lora_r": 32,
    "lora_alpha": 64,
    "lora_dropout": 0.05,
    "lora_target_modules": [
        "gate_proj", "down_proj", "up_proj",
        "q_proj", "v_proj", "k_proj", "o_proj",
    ],
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
    "deepspeed": "configs/deepspeed_zero2.json",
    "special_tokens": {"pad_token": "<|endoftext|>"},
}


TEMPLATES = {
    "hpqa": {
        "name_prefix": "hpqa_k100_hn98",
        "name_suffix": "qboth_2500_qwen08_lora",
        "data_path": "data/hotpotqa_train_k100_bridge_hn98_2500.jsonl",
        "task": "qa",
        "query_position": "both",
        "sequence_len": 16384,
        "learning_rate": 0.0005,
        "num_epochs": 1,
        "gradient_accumulation_steps": 4,
        "dataset_processes": 32,
        "wandb_project": "corpus-reasoning",
        "output_prefix": "hpqa-k100-hn98-2500-qwen08-lora",
    },
    "contradiction": {
        "name_prefix": "contradiction_pubmed_n100_k3",
        "name_suffix": "qboth_qwen08_lora",
        "data_path": "data/contradiction_train_pubmed_both_n100_k3.jsonl",
        "task": "contradiction",
        "query_position": "both",
        "sequence_len": 16384,
        "learning_rate": 0.0001,
        "num_epochs": 1,
        "gradient_accumulation_steps": 4,
        "dataset_processes": 32,
        "wandb_project": "corpus-reasoning",
        "output_prefix": "contradiction-pubmed-n100-k3-qwen08-lora",
    },
}


def build_config(family: str, variant_tag: str, pattern_name: str,
                 extra: dict) -> dict:
    t = TEMPLATES[family]

    # Base block common to every config.
    cfg = {
        "base_model": "Qwen/Qwen3.5-0.8B-Base",
        "datasets": [{"path": t["data_path"], "type": "alpaca", "ds_type": "json"}],
        "val_set_size": 0.0,
        "output_dir": f"./outputs/checkpoints/{t['output_prefix']}-{variant_tag}",
        "wandb_project": t["wandb_project"],
        "wandb_name": f"{t['name_prefix']}_{variant_tag}_{t['name_suffix']}",
        "task": t["task"],
        "query_position": t["query_position"],
        "train_on_inputs": False,
        "sequence_len": t["sequence_len"],
        "dataset_processes": t["dataset_processes"],
        "micro_batch_size": 1,
        "gradient_accumulation_steps": t["gradient_accumulation_steps"],
        "num_epochs": t["num_epochs"],
        "learning_rate": t["learning_rate"],
        "before_dummy": 0,
        "after_dummy": 0,
    }

    # Pattern-specific keys (standard_attention, attention_pattern, doc_window_k, ...)
    if pattern_name != "standard":
        cfg["attention_pattern"] = pattern_name
    cfg.update(extra)

    # LoRA + training mechanics.
    cfg.update(COMMON_LORA_BLOCK)
    return cfg


def main():
    CONFIGS_DIR.mkdir(exist_ok=True)
    written = []
    for family in ("hpqa", "contradiction"):
        t = TEMPLATES[family]
        for variant_tag, pattern_name, extra in PATTERN_VARIANTS:
            cfg = build_config(family, variant_tag, pattern_name, extra)
            fname = f"{t['name_prefix']}_{variant_tag}_{t['name_suffix']}.yml"
            path = CONFIGS_DIR / fname
            path.write_text(yaml.safe_dump(cfg, sort_keys=False))
            written.append(path)
            print(f"  wrote {path}")
    print(f"\nGenerated {len(written)} configs.")


if __name__ == "__main__":
    main()
