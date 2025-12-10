import json
from pathlib import Path

from cached_path import cached_path
from olmo_core.data.tokenizer import TokenizerConfig
from olmo_core.distributed.checkpoint import load_model_and_optim_state, save_model_and_optim_state
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.io import join_path
from olmo_core.nn.transformer.config import TransformerConfig

if __name__ == "__main__":
    ckpt = "/weka/oe-training-default/benjaminm/runs/reverse_stage1_hnet_v8-e1d4-w-ee-no-sm-dmcs_100k-xlstm-ffw1.5-igate-bias10-seq4096/step75000"

    config_path = join_path(ckpt, "config.json")
    with cached_path(config_path).open() as f:
        config_dict = json.load(f)
    try:
        transformer_config = TransformerConfig.from_dict(config_dict["model"])
    except KeyError as e:
        raise OLMoConfigurationError(
            f"Failed to load config from checkpoint at {config_path}: missing required field {e}"
        ) from e

    model = transformer_config.build()

    incompatible_keys = load_model_and_optim_state(join_path(ckpt, "model_and_optim"), model)  # type: ignore

    out_dir = join_path(ckpt, "converted")
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # store teacher
    config_dict["dataset"]["tokenizer"] = TokenizerConfig.dolma2().as_config_dict()
    config_dict["model"] = config_dict["model"]["teacher_config"]

    model.teacher.blocks = model.blocks  # type: ignore

    json.dump(config_dict, open(join_path(out_dir, "config.json"), "w"))
    save_model_and_optim_state(join_path(out_dir, "model_and_optim"), model.teacher)  # type: ignore