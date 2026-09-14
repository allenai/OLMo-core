"""The public launcher preserves decode options and disambiguates result paths."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("periodic", [False, True])
@pytest.mark.parametrize("no_topk", [False, True])
def test_landmark_eval_flags_and_output_tag(monkeypatch, periodic, no_topk):
    path = (
        Path(__file__).resolve().parents[2]
        / "scripts/train/memexpress/singletask_ladder/run_q4b_beaker_multirung_eval.py"
    )
    spec = importlib.util.spec_from_file_location("eval_launcher", path)
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)
    monkeypatch.setattr(launcher, "get_root_dir", lambda cluster: "/weka/test")
    monkeypatch.setattr(
        launcher, "build_launch_config", lambda **kwargs: SimpleNamespace(**kwargs, env_vars=[])
    )
    kwargs = dict(
        run_name="model",
        task="contra",
        variant="compressive",
        cluster="ai2/jupiter",
        step="560",
        ckpt="/weka/step560",
        results_dir="/weka/eval",
        prompt_format="chat",
        query_position="both",
        ngpu=2,
        max_test=600,
        max_length=131072,
        batch_size=1,
        priority="urgent",
        ladder_version="v3",
        xlong=False,
        xlong_only=False,
        xlong_rungs="",
        cot_mode="none",
        landmark_top_k_blocks=None,
        landmark_nonselected_mass=None,
        landmark_periodic_output=periodic,
        landmark_disable_top_k=no_topk,
        eval_tag="comparison",
    )
    config = launcher.build_eval_launch_config(**kwargs)
    cmd = config.cmd[-1]
    assert f"LANDMARK_PERIODIC_OUTPUT={int(periodic)}" in cmd
    assert f"LANDMARK_DISABLE_TOP_K={int(no_topk)}" in cmd
    assert ("periodic-lm" in config.name) == periodic
    assert ("no-topk" in config.name) == no_topk
    assert "periodic-lm" in cmd if periodic else "periodic-lm" not in cmd
    if periodic:
        with pytest.raises(ValueError, match="landmark/compressive"):
            launcher.build_eval_launch_config(**dict(kwargs, variant="dense"))
    if no_topk:
        with pytest.raises(ValueError, match="conflicts"):
            launcher.build_eval_launch_config(**dict(kwargs, landmark_top_k_blocks=10))
