import json
import argparse
from typing import Dict, Any, List, Tuple, Set
from pprint import pformat


def get_nested_value(config: Dict[str, Any], key_path: str) -> Any:
    """
    Safely get a value from nested dictionary using dot notation
    """
    try:
        parts = key_path.split('.')
        current = config
        for part in parts:
            current = current[part]
        return current
    except (KeyError, TypeError):
        return None

def get_all_keys(d: Dict[str, Any], prefix: str = "") -> Set[str]:
    """
    Get all possible keys in a nested dictionary using dot notation
    """
    keys = set()
    for k, v in d.items():
        new_key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            keys.update(get_all_keys(v, new_key))
        else:
            keys.add(new_key)
    return keys

def get_mapped_keys(key_mapping: Dict[str, str]) -> Set[str]:
    """
    Get all keys that are part of the mapping (both old and new)
    """
    mapped_keys = set()
    for old_key, new_key in key_mapping.items():
        mapped_keys.add(old_key)
        mapped_keys.add(new_key)
    return mapped_keys

def compare_configs(old_config: Dict[str, Any], new_config: Dict[str, Any], 
                   key_mapping: Dict[str, str]) -> Tuple[List[Tuple[str, Any, Any]], List[str], Dict[str, Any], Dict[str, Any]]:
    """
    Compare two configs based on the provided key mapping and find unmapped values
    
    Returns:
        Tuple containing:
        - List of differences as (key, old_value, new_value)
        - List of keys that couldn't be found in either config
        - Dictionary of unmapped values in old config
        - Dictionary of unmapped values in new config
    """
    differences = []
    missing_keys = []
    
    # Compare mapped keys
    for old_key, new_key in key_mapping.items():
        old_value = get_nested_value(old_config, old_key)
        new_value = get_nested_value(new_config, new_key)
        
        if old_value is None and new_value is None:
            missing_keys.append((old_key, new_key))
            continue
            
        if old_value != new_value:
            differences.append((old_key, old_value, new_value))
    
    # Get all keys from both configs
    old_keys = get_all_keys(old_config)
    new_keys = get_all_keys(new_config)
    mapped_keys = get_mapped_keys(key_mapping)
    
    # Find unmapped keys
    unmapped_old_keys = old_keys - mapped_keys
    unmapped_new_keys = new_keys - mapped_keys
    
    # Get values for unmapped keys
    unmapped_old_values = {k: get_nested_value(old_config, k) for k in unmapped_old_keys}
    unmapped_new_values = {k: get_nested_value(new_config, k) for k in unmapped_new_keys}
            
    return differences, missing_keys, unmapped_old_values, unmapped_new_values

def print_config_comparison(differences: List[Tuple[str, Any, Any]], 
                          missing_keys: List[str],
                          unmapped_old_values: Dict[str, Any],
                          unmapped_new_values: Dict[str, Any],
                          key_mapping: Dict[str, str]) -> None:
    """
    Print the comparison results in a readable format
    """
    print("=" * 80)
    print("=== Differences in Mapped Keys ===")
    print("=" * 80 + "\n")
    
    if not differences:
        print("No differences found in mapped keys that exist in both configs.\n")
    else:
        for key, old_val, new_val in sorted(differences):
            print(f"Key: {key} // {key_mapping[key]}")
            print(f"  Old value: {pformat(old_val, indent=4)}")
            print(f"  New value: {pformat(new_val, indent=4)}")
            print()
    
    if missing_keys:
        print("=" * 80)
        print("=== Missing Mapped Keys ===")
        print("=" * 80 + "\n")
        print("The following mapped keys were not found in one or both configs:")
        for old_key, new_key in sorted(missing_keys):
            print(f"  - Old config key: {old_key}")
            print(f"    New config key: {new_key}")
            print()
    
    print("=" * 80)
    print("=== Unmapped Values in Old Config ===")
    print("=" * 80 + "\n")

    print(f"Unmapped values in old config: {[key for key in sorted(unmapped_old_values.keys()) if not key.startswith('_')]}")
    # for key in sorted(unmapped_old_values.keys()):
    #     print(f"Key: {key}")
    #     print(f"Value: {pformat(unmapped_old_values[key], indent=4)}")
    #     print()
    
    print("=" * 80)
    print("=== Unmapped Values in New Config ===")
    print("=" * 80 + "\n")

    print(f"Unmapped values in new config: {[key for key in sorted(unmapped_new_values.keys()) if not key.startswith('_')]}")
    # for key in sorted(unmapped_new_values.keys()):
    #     print(f"Key: {key}")
    #     print(f"Value: {pformat(unmapped_new_values[key], indent=4)}")
    #     print()

# Example usage
def main(old_config: Dict[str, Any], new_config: Dict[str, Any]) -> None:
    
    # Key mapping from previous artifact
    key_mapping = {

        # model

        "model.value.embedding_size": "model.value.vocab_size",
        "model.value.d_model": "model.value.d_model",
        "model.value.n_layers": "model.value.n_layers",
        "model.value.n_heads": "model.value.block.attention.n_heads",
        "model.value.embedding_size": "model.value.vocab_size",
        "model.value.max_sequence_length": "dataset.value.sequence_length",
        # "model.value.pad_token_id": "model.value.tokenizer.pad_token_id",
        # "model.value.eos_token_id": "model.value.tokenizer.eos_token_id",

        # optimizer
        "optimizer.value.learning_rate": "optim.value.lr",
        "optimizer.value.weight_decay": "optim.value.weight_decay",
        "optimizer.value.betas": "optim.value.betas",
        "optimizer.value.eps": "optim.value.eps",

        # dataset
        "tokenizer.value.identifier": "dataset.value.tokenizer.identifier",
        "data.value.seed": "data_loader.value.seed",

        # batch size
        "global_train_batch_size.value": "data_loader.value.global_batch_size",
        "eval_interval": "trainer.callbacks.downstream_evaluator.eval_interval",

        # scheduler
        "scheduler.value.alpha_f": "trainer.value.callbacks.lr_scheduler.scheduler.alpha_f",
        "scheduler.value.t_warmup": "trainer.value.callbacks.lr_scheduler.scheduler.warmup_steps",

        # trainer
        "auxiliary_loss_multiplier.value": "trainer.value.z_loss_multiplier",
        "max_grad_norm.value": "trainer.value.callbacks.grad_clipper.max_grad_norm",

        # gc
        "gen1_gc_interval.value": "trainer.value.callbacks.garbage_collector.gc_interval",

        # misc
        "_wandb.value.python_version": "_wandb.value.python_version",
        "_wandb.value.cli_version": "_wandb.value.cli_version",
        "_wandb.value.framework": "_wandb.value.framework",
    }
    
    differences, missing_keys, unmapped_old_values, unmapped_new_values = compare_configs(
        old_config, new_config, key_mapping)
    print_config_comparison(differences, missing_keys, unmapped_old_values, unmapped_new_values, key_mapping)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("old_config_path", type=str)
    parser.add_argument("new_config_path", type=str)

    args = parser.parse_args()

    old_config = json.load(open(args.old_config_path))
    new_config = json.load(open(args.new_config_path))

    main(old_config, new_config)
