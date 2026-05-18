"""
Example script to convert an OLMo Core model checkpoint to a HuggingFace model checkpoint.

Supports both standard architectures (olmo2, olmo3) and hybrid (GDN + attention) architectures.
Hybrid models are saved as raw ``config.json`` + ``model.safetensors`` rather than using
``save_pretrained()``.

The conversion logic lives in :mod:`olmo_core.nn.hf.convert_checkpoint`; this script is a
thin CLI wrapper.

Usage::

    # Standard model
    python convert_checkpoint_to_hf.py -i /path/to/checkpoint -o /path/to/output

    # Hybrid model (auto-detected)
    python convert_checkpoint_to_hf.py -i /path/to/hybrid-checkpoint -o /path/to/output
"""

from argparse import ArgumentParser

import torch

from olmo_core.config import DType
from olmo_core.nn.hf import convert_checkpoint_to_hf, load_config
from olmo_core.utils import prepare_cli_environment


def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "-i",
        "--checkpoint-input-path",
        type=str,
        required=True,
        help="Local or remote directory containing the OLMo Core checkpoint.",
    )

    parser.add_argument(
        "-o",
        "--huggingface-output-dir",
        type=str,
        required=True,
        help="Local or remote directory where the converted checkpoint should be saved.",
    )
    parser.add_argument(
        "-s",
        "--max-sequence-length",
        type=int,
        help="Max sequence length supported by the model. If not set, the model_max_length of the tokenizer will be used.",
    )
    parser.add_argument(
        "-t",
        "--tokenizer",
        help="Identifier of the HuggingFace tokenizer to save the model with. If not set, the tokenizer from the experiment config will be used, or no tokenizer will be saved if not present in the experiment config.",
    )
    parser.add_argument(
        "--skip-validation",
        dest="validate",
        action="store_false",
        help="If set, validation to check that the converted model matches the original model is skipped.",
    )
    parser.add_argument(
        "--debug",
        dest="debug",
        action="store_true",
        help="If set, debug information of validation is output.",
    )
    parser.add_argument(
        "--device",
        type=torch.device,
        help="The device on which conversion occurs. Defaults to CPU.",
    )
    parser.add_argument(
        "--dtype",
        help="The torch dtype that model weights should be saved as. Defaults to bfloat16 due to https://github.com/allenai/olmo-cookbook/issues/60.",
        type=DType,
        default=DType.bfloat16,
    )
    parser.add_argument(
        "--validation-device",
        type=torch.device,
        help="The device on which validation occurs. Defaults to `device`.",
    )
    parser.add_argument(
        "--validation-sliding-window",
        help="If set, overrides the model's sliding window size during validation. Useful for checking that sliding window is correctly implemented.",
        type=int,
    )
    parser.add_argument(
        "--moe-capacity-factor",
        type=float,
        help="The MoE capacity factor. Higher capacity factor can decrease validation false negatives but may cause out of memory errors.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    experiment_config = load_config(args.checkpoint_input_path)
    if experiment_config is None:
        raise RuntimeError("Experiment config not found, cannot convert to HF checkpoint")

    transformer_config_dict = experiment_config["model"]
    tokenizer_config_dict = experiment_config.get("dataset", {}).get("tokenizer")

    assert transformer_config_dict is not None
    assert tokenizer_config_dict is not None

    convert_checkpoint_to_hf(
        original_checkpoint_path=args.checkpoint_input_path,
        output_path=args.huggingface_output_dir,
        transformer_config_dict=transformer_config_dict,
        tokenizer_config_dict=tokenizer_config_dict,
        dtype=args.dtype,
        max_sequence_length=args.max_sequence_length,
        tokenizer_id=args.tokenizer,
        validate=args.validate,
        debug=args.debug,
        device=args.device,
        moe_capacity_factor=args.moe_capacity_factor,
        validation_device=args.validation_device or args.device,
        validation_sliding_window=args.validation_sliding_window,
    )


if __name__ == "__main__":
    prepare_cli_environment()
    main()
