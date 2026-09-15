"""Build the real attention modules: dry_run's parameter counting misses invalid options."""

import importlib
from pathlib import Path

import pytest


@pytest.mark.parametrize("arm", ["dense", "compressive"])
def test_contradiction_attention_builds(arm, monkeypatch, tmp_path):
    family = Path(__file__).resolve().parents[2] / "scripts/train/memexpress/sft_xlong256k"
    monkeypatch.syspath_prepend(str(family))
    base = importlib.import_module("_qwen35_xlong5_dolci25_256k_common")
    recipe = importlib.import_module("_qwen35_contradiction_256k_common")
    from olmo_core.internal.experiment import CliContext, SubCmd

    monkeypatch.setattr(base, "get_root_dir", lambda _: str(tmp_path))
    monkeypatch.setattr(base, "get_work_dir", lambda _: str(tmp_path))
    monkeypatch.setattr(base, "build_launch_config", lambda **_: None)
    config = recipe.build_contradiction_experiment(
        CliContext(
            script=__file__, cmd=SubCmd.train, run_name="test", cluster="test", overrides=[]
        ),
        arm=arm,
    )
    attention = config.model.block["attn"].sequence_mixer.build(
        config.model.d_model, layer_idx=3, n_layers=config.model.n_layers, init_device="meta"
    )
    if arm == "compressive":
        assert attention.block_size == 64
        assert attention.mem_freq == 63
        assert config.dataset[0].num_landmarks == 1
        assert (
            config.dataset[0].sequence_length // attention.block_size * attention.mem_freq >= 262144
        )
    assert config.trainer.max_duration.value == 3
    assert config.trainer.max_duration.unit == "epochs"
    assert config.train_module.state_dict_load_opts is None
    assert config.trainer.load_optim_state is False
    assert config.trainer.load_trainer_state is False
