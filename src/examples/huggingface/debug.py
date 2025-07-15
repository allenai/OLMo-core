import json
from pathlib import Path

def debug_key_mapping():
    config_path = Path("/data/input/amanr/step1000002/config.json")
    if not config_path.exists():
        print("Error: config.json not found")
        return
    
    with open(config_path) as f:
        config = json.load(f)
    
    n_layers = config["model"]["n_layers"]

    from convert_checkpoint_to_hf import create_old_olmo_to_olmo_core_mapping
    
    key_mapping = create_old_olmo_to_olmo_core_mapping(n_layers)
    
    print(f"\nCreated mapping with {len(key_mapping)} entries:")
    for i, (model_key, checkpoint_key) in enumerate(key_mapping.items()):
        if i < 10:
            print(f"  {model_key} == {checkpoint_key}")
        else:
            break
            
    for model_key, checkpoint_key in key_mapping.items():
        if "blocks.0." in model_key:
            print(f"  {model_key} == {checkpoint_key}")

if __name__ == "__main__":
    debug_key_mapping()