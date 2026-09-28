"""End-to-end test of the LoRA merge script against a real distributed checkpoint.

The point is the round trip through the on-disk format, not the arithmetic (covered in
``src/test/nn/lora_test.py``): the script reads checkpoint *keys*, and the thing that would
break it is a key-space assumption, not a matmul.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from olmo_core.distributed.checkpoint import load_keys, save_state_dict
from olmo_core.nn.lora import LoRAConfig, apply_lora, merge_lora_

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "src" / "scripts" / "merge_lora_checkpoint.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("merge_lora_checkpoint", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["merge_lora_checkpoint"] = module
    spec.loader.exec_module(module)
    return module


class _Tiny(nn.Module):
    def __init__(self, d=8):
        super().__init__()
        self.lm = nn.Module()
        self.lm.blocks = nn.ModuleDict({"0": nn.Module()})
        self.lm.blocks["0"].attention = nn.Module()
        self.lm.blocks["0"].attention.w_q = nn.Linear(d, d, bias=False)
        self.lm.blocks["0"].attention.w_out = nn.Linear(d, d, bias=False)

    def forward(self, x):
        return self.lm.blocks["0"].attention.w_out(self.lm.blocks["0"].attention.w_q(x))


def _cfg():
    return LoRAConfig(
        rank=2,
        alpha=4.0,
        target_modules=["lm.blocks.*.attention.w_q", "lm.blocks.*.attention.w_out"],
    )


def _write_checkpoint(step_dir: Path, model: nn.Module, alpha: float) -> None:
    state = {f"model.{k}": v for k, v in model.state_dict().items()}
    save_state_dict(step_dir / "model_and_optim", state)
    (step_dir / "config.json").write_text(
        json.dumps({"train_module": {"lora": {"rank": 2, "alpha": alpha}}, "model": {}})
    )


@pytest.fixture
def script():
    return _load_script()


def test_merge_matches_an_in_memory_merge(tmp_path, script):
    torch.manual_seed(0)
    model = _Tiny()
    apply_lora(model, _cfg())
    for name, p in model.named_parameters():
        if name.endswith(".lora_B"):
            with torch.no_grad():
                p.normal_(std=0.2)

    src = tmp_path / "step10"
    _write_checkpoint(src, model, alpha=4.0)

    dst = tmp_path / "step10-merged"
    script.merge(src, dst, alpha=None, save_overwrite=False)

    merge_lora_(model)
    expected = model.state_dict()

    keys = [f"model.{k}" for k in expected]
    got = dict(zip(keys, load_keys(dst / "model_and_optim", keys)))
    assert set(got) == set(keys)
    for k, v in expected.items():
        torch.testing.assert_close(got[f"model.{k}"], v, rtol=1e-6, atol=1e-7)


def test_merged_checkpoint_has_no_adapter_keys_and_keeps_config(tmp_path, script):
    torch.manual_seed(0)
    model = _Tiny()
    apply_lora(model, _cfg())
    src = tmp_path / "step10"
    _write_checkpoint(src, model, alpha=4.0)

    dst = tmp_path / "step10-merged"
    script.merge(src, dst, alpha=None, save_overwrite=False)

    assert not any(k.endswith((".lora_A", ".lora_B")) for k in script._model_keys(dst))
    assert (dst / "config.json").is_file()
    reference = {f"model.{k}" for k in _Tiny().state_dict()}
    assert set(script._model_keys(dst)) == reference


def test_check_flags_unmerged_and_clears_merged(tmp_path, script, capsys):
    torch.manual_seed(0)
    model = _Tiny()
    apply_lora(model, _cfg())
    src = tmp_path / "step10"
    _write_checkpoint(src, model, alpha=4.0)

    assert script.check(src) == 1
    assert "UNMERGED" in capsys.readouterr().out

    dst = tmp_path / "step10-merged"
    script.merge(src, dst, alpha=None, save_overwrite=False)
    assert script.check(dst) == 0
    assert "OK" in capsys.readouterr().out


def test_merging_a_plain_checkpoint_is_an_error(tmp_path, script):
    src = tmp_path / "step10"
    _write_checkpoint(src, _Tiny(), alpha=4.0)
    with pytest.raises(RuntimeError, match="no LoRA adapters"):
        script.merge(src, tmp_path / "out", alpha=None, save_overwrite=False)


def test_missing_alpha_is_an_error_rather_than_a_guess(tmp_path, script):
    torch.manual_seed(0)
    model = _Tiny()
    apply_lora(model, _cfg())
    src = tmp_path / "step10"
    _write_checkpoint(src, model, alpha=4.0)
    (src / "config.json").write_text(json.dumps({"train_module": {}}))
    with pytest.raises(RuntimeError, match="--alpha"):
        script.merge(src, tmp_path / "out", alpha=None, save_overwrite=False)
