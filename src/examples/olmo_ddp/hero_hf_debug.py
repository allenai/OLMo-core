"""Read-only layerwise diagnostic for an unpublished hero conversion."""

import argparse
import logging
import os

import torch

from hero_hf_convert import reference_config
from hero_hf_download import SCRATCH, TARGETS
from hero_hf_reference_ops import install, self_test
from olmo_core.config import DType
from olmo_core.nn.hf.config import _register_olmo3moe_auto_classes
from olmo_core.nn.hf.convert_checkpoint import (
    _load_ddp_optimizer_model_state,
    load_config,
    validate_conversion,
)
from olmo_core.nn.transformer.config import TransformerConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("emo", "non-emo"), required=True)
    parser.add_argument("--step", type=int, choices=tuple(TARGETS.values()), required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    root = SCRATCH / args.arm / f"step{args.step}"
    if (root / "conversion-success.json").exists():
        raise RuntimeError("Diagnostic is restricted to unpublished conversions")
    _register_olmo3moe_auto_classes()
    self_test()
    install()
    os.environ["OLMO_HF_MOE_CORE_REFERENCE"] = "1"
    config = load_config(root / "olmo-core")
    model = TransformerConfig.from_dict(reference_config(config["model"])).build(init_device="meta")
    model.to_empty(device="cpu")
    _load_ddp_optimizer_model_state(
        root / "olmo-core/model_and_optim",
        model,
        work_dir=str(root / "load-work"),
        return_state_dict=False,
    )
    torch.manual_seed(17)
    validate_conversion(
        root / "hf.partial",
        model,
        config["dataset"]["tokenizer"]["vocab_size"],
        debug=True,
        dtype=DType.bfloat16,
        device=torch.device("cuda"),
    )


if __name__ == "__main__":
    main()
