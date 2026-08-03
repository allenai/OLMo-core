"""Configuration invariants for the s002 Molmo2 stage-2 launcher."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from olmo_core.distributed.parallel import DataParallelType


def _load_stage2_module():
    path = Path(__file__).parents[2] / "scripts" / "train" / "Molmo2-Stage2.py"
    spec = importlib.util.spec_from_file_location("molmo2_stage2", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_s002_stage2_uses_olmo_ddp_ep8_and_trains_all_components():
    stage2 = _load_stage2_module()
    config = stage2._build_train_module_config(
        sequence_length=64,
        rank_microbatch_size=64,
        ep_degree=8,
        compile_model=False,
    )

    assert config.dp_config.name == DataParallelType.ddp
    assert config.ep_config.degree == 8
    assert config.freeze_params is None
    assert config.vision_activation_checkpointing
    assert config.connector_activation_checkpointing
    assert config.response_logits_only
    assert not config.compile_model

    overrides = {tuple(group.params): group.opts for group in config.optim.group_overrides}
    assert overrides[("*connector.*",)]["lr"] == stage2.CONNECTOR_LR
    assert overrides[("*vision.*",)]["lr"] == stage2.VISION_LR
    assert config.optim.lr == stage2.LLM_LR


def test_s002_stage2_production_defaults():
    stage2 = _load_stage2_module()

    assert stage2.NUM_NODES == 2
    assert stage2.EP_DEGREE == 8
    assert stage2.GLOBAL_BATCH_INSTANCES == 128
    assert stage2.RANK_MICROBATCH_INSTANCES == 1
    assert stage2.MAX_STEPS == 30_000
    assert stage2.DEFAULT_LOAD_PATH.startswith("/weka/oe-training-default/rustin/")
