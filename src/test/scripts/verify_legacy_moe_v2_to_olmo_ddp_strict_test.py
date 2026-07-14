import importlib.util
import json
import sys
from pathlib import Path

import torch


SCRIPT = Path(__file__).parents[2] / "scripts/verify_legacy_moe_v2_to_olmo_ddp_strict.py"
SPEC = importlib.util.spec_from_file_location("verify_legacy_moe_v2_to_olmo_ddp_strict", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_main_accepts_and_exactly_verifies_all_moe_checkpoint(tmp_path, monkeypatch) -> None:
    source = tmp_path / "source"
    target = tmp_path / "target"
    for checkpoint in (source, target):
        model_dir = checkpoint / "model_and_optim"
        model_dir.mkdir(parents=True)
        (model_dir / ".metadata").touch()

    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "model": {
                    "d_model": 4,
                    "n_layers": 1,
                    "block": {"name": "moe_fused_v2"},
                }
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "strict.json"
    source_state = {
        "module.blocks.0.routed_experts.w_down.main": torch.tensor(
            [0.0, -0.0, 1.0], dtype=torch.float32
        )
    }

    def load_main_tensors(checkpoint_dir, **_kwargs):
        assert checkpoint_dir in {
            source / "model_and_optim",
            target / "model_and_optim",
        }
        return {key: tensor.clone() for key, tensor in source_state.items()}

    monkeypatch.setattr(MODULE, "_load_main_tensors", load_main_tensors)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT),
            str(source),
            str(target),
            "--config",
            str(config),
            "--output",
            str(output),
        ],
    )

    MODULE.main()

    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["status"] == "STRICT_TENSOR_MATCH"
    assert report["bitwise_equal"] is True
    assert report["dense_layers"] == []
    assert report["transformed_target_tensor_count"] == 0
    assert report["unchanged_tensor_count"] == 1
