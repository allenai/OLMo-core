"""
Tests for the convert_moe_v2_checkpoint_config.py script.
"""

import importlib.util
import json
from pathlib import Path

from click.testing import CliRunner

spec = importlib.util.spec_from_file_location(
    "convert_moe_v2_checkpoint_config", "src/scripts/convert_moe_v2_checkpoint_config.py"
)
if spec is None or spec.loader is None:
    raise ImportError("Could not load convert_moe_v2_checkpoint_config.py")
convert_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(convert_module)


def _legacy_config() -> dict:
    """A config.json nesting every legacy MoE-v2 class path the script should rewrite."""
    return {
        "_CLASS_": "olmo_core.nn.transformer.config.MoEFusedV2TransformerConfig",
        "d_model": 128,
        "block": {
            "_CLASS_": "olmo_core.nn.moe.v2.block.MoEFusedV2TransformerBlockConfig",
            "layers": [
                {"_CLASS_": "olmo_core.nn.moe.v2.block.OLMoDDPTransformerBlockConfig"},
            ],
        },
        "train_module": {
            "_CLASS_": (
                "olmo_core.train.train_module.transformer.config."
                "MoEV2TransformerTrainModuleConfig"
            ),
            "optim": {
                "_CLASS_": "olmo_core.optim.moe_optimizer.MoEFusedV2OptimizerConfig",
            },
        },
        # An already-canonical entry must be left untouched.
        "model": {"_CLASS_": "olmo_core.nn.ddp.model.OLMoDDPModel"},
    }


def test_rewrite_config_dict_maps_all_legacy_paths():
    new, changes = convert_module.rewrite_config_dict(_legacy_config())

    assert new["_CLASS_"] == "olmo_core.nn.transformer.config.OLMoDDPModelConfig"
    assert new["block"]["_CLASS_"] == "olmo_core.nn.ddp.block.OLMoDDPTransformerBlockConfig"
    assert (
        new["block"]["layers"][0]["_CLASS_"]
        == "olmo_core.nn.ddp.block.OLMoDDPTransformerBlockConfig"
    )
    assert (
        new["train_module"]["_CLASS_"]
        == "olmo_core.train.train_module.transformer.config.OLMoDDPTrainModuleConfig"
    )
    assert (
        new["train_module"]["optim"]["_CLASS_"]
        == "olmo_core.optim.moe_optimizer.OLMoDDPOptimizerConfig"
    )
    # The already-canonical entry is unchanged and did not count as a rewrite.
    assert new["model"]["_CLASS_"] == "olmo_core.nn.ddp.model.OLMoDDPModel"

    assert len(changes) == 5
    # No legacy paths survive anywhere in the output.
    assert "MoEFusedV2" not in json.dumps(new)
    assert "MoEV2" not in json.dumps(new)
    assert ".moe.v2." not in json.dumps(new)


def test_rewrite_config_dict_maps_canonical_module_aliases():
    # Aliases that lived in the *canonical* modules (not the moe.v2 shim paths) must also be
    # rewritten — otherwise the alias name would be left dangling once it's removed.
    config = {
        "_CLASS_": "olmo_core.nn.ddp.model.MoEFusedV2Transformer",
        "block": {"_CLASS_": "olmo_core.nn.ddp.block.MoEFusedV2TransformerBlockConfig"},
        "train_module": {
            "_CLASS_": (
                "olmo_core.train.train_module.transformer.ddp_train_module."
                "MoEV2TransformerTrainModule"
            ),
        },
    }
    new, changes = convert_module.rewrite_config_dict(config)
    assert new["_CLASS_"] == "olmo_core.nn.ddp.model.OLMoDDPModel"
    assert new["block"]["_CLASS_"] == "olmo_core.nn.ddp.block.OLMoDDPTransformerBlockConfig"
    assert (
        new["train_module"]["_CLASS_"]
        == "olmo_core.train.train_module.transformer.ddp_train_module.OLMoDDPTrainModule"
    )
    assert len(changes) == 3
    assert "MoEFusedV2" not in json.dumps(new)
    assert "MoEV2" not in json.dumps(new)


def test_rewrite_config_dict_maps_package_level_export_paths():
    # Configs could record the class under a package-level ``__init__`` re-export path rather than
    # the concrete module. Leaf-name matching handles those too.
    config = {
        "optim": {"_CLASS_": "olmo_core.optim.MoEFusedV2OptimizerConfig"},
        "train_module": {
            "_CLASS_": (
                "olmo_core.train.train_module.transformer.MoEV2TransformerTrainModuleConfig"
            ),
        },
        "model": {"_CLASS_": "olmo_core.nn.moe.v2.MoEFusedV2Transformer"},
        "model_config": {"_CLASS_": "olmo_core.nn.transformer.MoEFusedV2TransformerConfig"},
    }
    new, changes = convert_module.rewrite_config_dict(config)
    assert new["optim"]["_CLASS_"] == "olmo_core.optim.moe_optimizer.OLMoDDPOptimizerConfig"
    assert (
        new["train_module"]["_CLASS_"]
        == "olmo_core.train.train_module.transformer.config.OLMoDDPTrainModuleConfig"
    )
    assert new["model"]["_CLASS_"] == "olmo_core.nn.ddp.model.OLMoDDPModel"
    assert new["model_config"]["_CLASS_"] == "olmo_core.nn.transformer.config.OLMoDDPModelConfig"
    assert len(changes) == 4
    assert "MoEFusedV2" not in json.dumps(new)
    assert "MoEV2" not in json.dumps(new)


def test_rewrite_config_dict_maps_shim_exported_moe_subconfigs():
    # The deleted moe.v2.block shim re-exported these MoE sub-configs; a config recorded under that
    # module path must be normalized to their defining modules.
    config = {
        "router": {"_CLASS_": "olmo_core.nn.moe.v2.block.MoERouterConfigV2"},
        "routed_experts": {"_CLASS_": "olmo_core.nn.moe.v2.block.RoutedExpertsConfig"},
        "shared_experts": {"_CLASS_": "olmo_core.nn.moe.v2.block.SharedExpertsConfig"},
    }
    new, changes = convert_module.rewrite_config_dict(config)
    assert new["router"]["_CLASS_"] == "olmo_core.nn.moe.v2.router.MoERouterConfigV2"
    assert (
        new["routed_experts"]["_CLASS_"] == "olmo_core.nn.moe.v2.routed_experts.RoutedExpertsConfig"
    )
    assert (
        new["shared_experts"]["_CLASS_"] == "olmo_core.nn.moe.v2.shared_experts.SharedExpertsConfig"
    )
    assert len(changes) == 3


def test_all_rewrite_targets_are_importable():
    # Every canonical target must actually resolve, so a migrated config loads.
    import importlib

    for target in set(convert_module._CANONICAL_BY_LEAF_NAME.values()):
        module_path, _, class_name = target.rpartition(".")
        module = importlib.import_module(module_path)
        assert hasattr(module, class_name), f"{target} does not resolve"


def test_rewrite_config_dict_noop_on_canonical():
    canonical = {"_CLASS_": "olmo_core.nn.ddp.model.OLMoDDPModel", "d_model": 64}
    new, changes = convert_module.rewrite_config_dict(canonical)
    assert new == canonical
    assert changes == []


def test_convert_checkpoint_config_rewrites_in_place(tmp_path: Path):
    ckpt = tmp_path / "step10000"
    ckpt.mkdir()
    (ckpt / "config.json").write_text(json.dumps(_legacy_config()))

    n = convert_module.convert_checkpoint_config(str(ckpt))
    assert n == 5

    migrated = json.loads((ckpt / "config.json").read_text())
    assert migrated["_CLASS_"] == "olmo_core.nn.transformer.config.OLMoDDPModelConfig"
    assert "MoEFusedV2" not in json.dumps(migrated)


def test_convert_checkpoint_config_dry_run_leaves_file_untouched(tmp_path: Path):
    ckpt = tmp_path / "step10000"
    ckpt.mkdir()
    original = json.dumps(_legacy_config())
    (ckpt / "config.json").write_text(original)

    n = convert_module.convert_checkpoint_config(str(ckpt), dry_run=True)
    assert n == 5
    assert (ckpt / "config.json").read_text() == original


def test_cli_dry_run(tmp_path: Path):
    ckpt = tmp_path / "step10000"
    ckpt.mkdir()
    original = json.dumps(_legacy_config())
    (ckpt / "config.json").write_text(original)

    result = CliRunner().invoke(convert_module.main, ["--dry-run", str(ckpt)])
    assert result.exit_code == 0, result.output
    assert (ckpt / "config.json").read_text() == original
