"""Focused CPU regressions for single-node SFT specs and the real training entrypoints."""

import copy
import os
import runpy
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from unittest.mock import patch


def check_specs():
    """Single-replica specs drop only the multi-replica rendezvous options."""
    from olmoe3_qkgain_control import training_spec
    from olmoe3_qkgain_plan import ARMS, Run

    template = {"tasks": [{
        "resources": {"gpuCount": 8}, "context": {}, "envVars": [], "datasets": [],
        "synchronizedStartTimeout": 900_000_000_000,
    }]}
    original = copy.deepcopy(template)
    for arm in ARMS:
        task = training_spec(template, Run(arm, "sft"), "test", ["test-host"])["tasks"][0]
        assert "synchronizedStartTimeout" not in task
        assert task["replicas"] == 1 and task["resources"]["gpuCount"] == 8
        assert task["leaderSelection"] is False
        assert task["context"]["minRuntime"] == "6h"
        assert task["context"]["priority"] == "urgent"
    task = training_spec(template, Run("7to1-split", "lc"), "test", ["test-host"])["tasks"][0]
    assert task["synchronizedStartTimeout"] == 900_000_000_000
    assert task["replicas"] == 8 and task["leaderSelection"] is True
    assert template == original
    print("SFT_SINGLE_NODE_SPECS_VERIFIED", flush=True)


def check_callbacks():
    """Import-qualified custom callbacks survive the same config merge used by training."""
    from olmo_core.config import Config
    from olmo_core.train.callbacks import Callback
    from olmoe3_qkgain_train import Audit, Finish

    @dataclass
    class Holder(Config):
        callbacks: dict[str, Callback] = field(default_factory=dict)

    # A local holder has no importable name; decode it explicitly, while preserving
    # the real module-qualified names on its nested callbacks.
    source = Holder({"audit": Audit(run_id="test"), "finish": Finish(run_id="test")})
    encoded = source.as_dict(include_class_name=True)
    encoded.pop(Config.CLASS_NAME_FIELD)
    result = Holder.from_dict(encoded)
    assert type(result.callbacks["audit"]) is Audit
    assert type(result.callbacks["finish"]) is Finish
    assert result.callbacks["audit"].run_id == "test"
    print("SFT_CALLBACK_ROUNDTRIP_VERIFIED", flush=True)


def check_entrypoint(which):
    """Execute the real script dispatch; replace only GPU launch with a dry config build."""
    import olmo_core.internal.experiment as experiment

    here = Path(__file__).parent
    if which == "hero":
        from olmoe3_hero_sft8mi import RUN_ID

        script, name, expected_batch, expected_steps = (
            here / "olmoe3_hero_sft8mi.py", RUN_ID, 8_388_608, 210
        )
    else:
        from olmoe3_qkgain_plan import Run

        run = Run(which, "sft")
        script, name, expected_batch, expected_steps = (
            here / "olmoe3_qkgain_train.py", run.run_id, run.batch, run.end
        )
    os.environ.update(QKGAIN_RUN=name, QKGAIN_START="0", QKGAIN_STOP="2")
    os.environ.pop("QKGAIN_LOAD", None)
    checked = []

    def dry_main(*, config_builder):
        config = config_builder(experiment.CliContext(
            str(script), experiment.SubCmd.dry_run, name, "ai2/holmes", []
        ))
        assert config.data_loader.global_batch_size == expected_batch
        assert config.trainer.max_duration.value == expected_steps
        assert config.trainer.hard_stop.value == 2
        assert config.train_module.optim.weight_decay == 0
        assert not config.trainer.callbacks["checkpointer"].save_async
        # Repeat the serialization that failed in the original wrapper.
        restored = config.merge([])
        for key in ("qkgain_audit", "finish", "sft_validation"):
            callback = restored.trainer.callbacks[key]
            assert not isinstance(callback, dict) and callback.run_id == name
        for block in [config.model.block, *config.model.block_overrides.values()]:
            router = getattr(block, "routed_experts_router", None)
            assert router is None or router.emo is None
        checked.append(name)
        print("SFT_REAL_ENTRYPOINT_VERIFIED", name, expected_batch, expected_steps, flush=True)

    with patch.object(experiment, "main", dry_main), patch.object(
        sys, "argv", [str(script), "train", name, "ai2/holmes"]
    ):
        runpy.run_path(str(script), run_name="__main__")
    assert checked == [name]


if __name__ == "__main__":
    if sys.argv[1:] == ["--unit"]:
        check_specs()
        check_callbacks()
    elif sys.argv[1:] == ["--entrypoints"]:
        check_specs()
        for which in ("3to1-split", "3to1-shared", "7to1-split", "hero"):
            subprocess.run([sys.executable, __file__, "--entrypoint", which], check=True)
    elif len(sys.argv) == 3 and sys.argv[1] == "--entrypoint":
        check_entrypoint(sys.argv[2])
    else:
        raise ValueError(sys.argv[1:])
