import json
from pathlib import Path

from cached_path import cached_path
from olmo_core.data.tokenizer import TokenizerConfig
from olmo_core.distributed.checkpoint import load_model_and_optim_state, save_model_and_optim_state
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.io import join_path
from olmo_core.nn.transformer.config import TransformerConfig

if __name__ == "__main__":
    ckpt = "/weka/oe-training-default/benjaminm/runs/stage2_hnet_v4_global_2dot6e-5_fixed_bsx4_local_5.4e-5_zero_bos-fula-100k/step100000"

    config_path = join_path(ckpt, "config.json")
    with cached_path(config_path).open() as f:
        config_dict = json.load(f)
        del config_dict["model"]["add_boundary_predictor"]
        del config_dict["model"]["teacher_config"]["add_boundary_predictor"]
        config_dict["model"]["local_encoder"]["boundary_predictor"] = "dtp"
    try:
        # Avoid loading the entire experiment config b/c we don't care about validation outside
        # of the transformer config and the tokenizer config
        transformer_config = TransformerConfig.from_dict(config_dict["model"])
        tokenizer_config = TokenizerConfig.from_dict(config_dict["dataset"]["tokenizer"])
    except KeyError as e:
        raise OLMoConfigurationError(
            f"Failed to load config from checkpoint at {config_path}: missing required field {e}"
        ) from e

    model = transformer_config.build()

    key_mapping = {}
    extend_key_mapping = {}

    prefixes_to_duplicate = []

    for prefix in prefixes_to_duplicate:
        extend_key_mapping.update({
            key: key.replace(f"{prefix}", f"teacher.{prefix}")
            for key in model.state_dict().keys() if key.startswith(prefix)
        })

    key_mapping |= {k: k.replace("local_encoder.", "").replace("boundary_predictor_module", "boundary_predictor") for k in model.state_dict().keys() if "boundary_predictor_module" in k}

    incompatible_keys = load_model_and_optim_state(join_path(ckpt, "model_and_optim"), model, key_mapping=key_mapping, extend_key_mapping=extend_key_mapping)

    out_dir = join_path(ckpt, "converted")
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    json.dump(config_dict, open(join_path(out_dir, "config.json"), "w"))
    save_model_and_optim_state(join_path(out_dir, "model_and_optim"), model)