"""Minimal adapters around the qualified training recipe; no new numerical kernels."""

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

from olmoe3_dolci_hero_plan import AUTO, DATA, PIN, ROOT, find_run, install, run

install()
import olmoe3_qkgain_train as adapter
from olmoe3_lr_sweep_watch import atomic_json

from olmo_core.distributed.utils import get_rank, get_world_size
from olmo_core.optim.scheduler import ConstantWithWarmup
from olmo_core.train.callbacks import Callback


@dataclass
class BranchPin(Callback):
    """Preserve the immutable branch initializer against ordinary trunk retention."""

    priority: ClassVar[int] = -15

    def post_checkpoint_saved(self, path):
        if get_rank() != 0 or Path(path).name != "step6000":
            return
        p = Path(path)
        assert p == run("hero").root / "step6000"
        PIN.parent.mkdir(parents=True, exist_ok=True)
        temporary = PIN.with_name("step6000.partial")
        if not PIN.exists():
            temporary.mkdir(exist_ok=True)
            for f in sorted(p.rglob("*")):
                assert not f.is_symlink()
                dest = temporary / f.relative_to(p)
                if f.is_dir():
                    dest.mkdir(exist_ok=True)
                elif f.is_file():
                    if dest.exists():
                        assert dest.stat().st_ino == f.stat().st_ino
                    else:
                        os.link(f, dest)
            temporary.rename(PIN)
        assert (PIN / "model_and_optim/.metadata").is_file()
        atomic_json(
            AUTO / "branch-source.json",
            dict(
                passed=True,
                source=str(PIN),
                step=6000,
                gpus=128,
                batch=16777216,
                metadata_sha256=hashlib.sha256((PIN / ".metadata.json").read_bytes()).hexdigest(),
                protection="owned immutable hardlink copy; never deleted automatically",
            ),
        )


def fingerprint(tensor):
    """Use the exact saved 128-element audit convention on a local shard slice."""
    value = tensor.detach().reshape(-1)
    assert value.numel()
    count = min(128, value.numel())
    offsets = adapter.torch.tensor(
        [i * (value.numel() - 1) // max(1, count - 1) for i in range(count)],
        device=value.device,
    )
    data = value[offsets].contiguous().view(adapter.torch.uint8).cpu().numpy().tobytes()
    return [value.numel(), str(value.dtype), hashlib.sha256(data).hexdigest()]


def verify_decay_reshard(trainer, path):
    """Check the approved EP1 128→64 fork against both original optimizer halves.

    Adapted from the qualified medium CBS restore audit. Checks every state key
    with exact sampled values, not full-tensor equality. Subsequent 64→64 resumes
    still use the original strict audit, including RNG and data-loader state.
    """
    assert Path(path) == PIN and get_world_size() == 64
    tm = trainer.train_module
    assert not tm.ep_enabled and not tm.pp_enabled
    rank = get_rank()
    references = {}

    def saved(old_rank):
        if old_rank not in references:
            row = json.loads((Path(path) / "resume_audit" / f"rank{old_rank}.json").read_text())
            assert (row["gpus"], row["rank"], row["step"]) == (128, old_rank, 6000)
            references[old_rank] = row
        return references[old_rank]

    actual = adapter.hero.state_sample(trainer)
    reference = saved(rank)
    for key in ("step", "tokens", "loss_history", "norm_history"):
        assert actual[key] == reference[key], ("Reshard changed", key)
    assert actual["tensors"].keys() == reference["tensors"].keys()
    tensors = dict(tm.optim.states)
    tensors.update(tm._persistent_model_buffer_state_dict())
    tensors.update((f"model_param/{name}", value) for name, value in tm.model.named_parameters())
    halves = 0
    for name, tensor in tensors.items():
        value = tensor.to_local() if hasattr(tensor, "to_local") else tensor
        value = value.detach().reshape(-1)
        old = reference["tensors"][name]
        if value.numel() == old[0]:
            assert actual["tensors"][name] == old, ("Unsharded restore mismatch", name)
            continue
        assert name in tm.optim.states and value.numel() == 2 * old[0], (
            "Unexpected EP1 reshard geometry",
            name,
            value.numel(),
            old[0],
        )
        for half in (0, 1):
            expected = saved(2 * rank + half)["tensors"][name]
            assert fingerprint(value.narrow(0, half * old[0], old[0])) == expected, (
                "Optimizer reshard mismatch",
                name,
                rank,
                half,
            )
            halves += 1
    assert halves > 0
    state = adapter.torch.load(
        Path(path) / "train" / f"rank{rank}.pt", map_location="cpu", weights_only=False
    )
    assert adapter.equal(state["data_loader"], trainer.data_loader.state_dict())
    assert trainer.data_loader.tokens_processed == actual["tokens"] == 6000 * 16777216
    return dict(
        source_gpus=128,
        gpus=64,
        sampled_state_exact=True,
        verified_old_shard_halves=halves,
        state_keys=len(tensors),
        rng_policy="Trainer default: reinitialize per-rank RNG when world size changes",
    )


@dataclass
class DecayResizeAudit(adapter.Audit):
    """Permit only the pinned initial 128→64 fork; keep later strict resume checks."""

    def post_checkpoint_loaded(self, path):
        if Path(path) != PIN:
            return super().post_checkpoint_loaded(path)
        r = find_run(self.run_id)
        assert r.kind == "decay" and self.step == r.start == 6000 and r.gpus == 64
        proof = verify_decay_reshard(self.trainer, path)
        atomic_json(
            r.root / "audit" / f"restore-{self.step}-rank{get_rank()}.json",
            dict(passed=True, fresh_stage=False, source=str(path), **proof),
        )
        if get_rank() == 0:
            print("DECAY_128_TO_64_RESTORE_VERIFIED", json.dumps(proof), flush=True)


def install_adapters(r):
    """Bind campaign-specific data/schedule before building import-qualified callbacks."""
    original_scheduler = adapter.scheduler
    adapter.scheduler = lambda x: (
        ConstantWithWarmup(warmup=2000) if x.kind == "hero" else original_scheduler(x)
    )
    adapter.hero.disk_action = lambda free: (
        "stop" if free < 10_000_000_000_000 else ("warn" if free < 12_000_000_000_000 else "ok")
    )
    original_trainer = adapter.trainer_config

    def trainer(common):
        c = original_trainer(common)
        if r.kind == "hero":
            c.callbacks["branch_pin"] = BranchPin()
        if r.kind == "decay":
            c.callbacks["qkgain_audit"] = DecayResizeAudit(run_id=r.run_id)
        if (
            r.kind in ("hero", "decay")
            and int(os.environ.get("QKGAIN_STOP", r.end)) <= r.start + 25
        ):
            c.metrics_collect_interval = 1
            c.no_evals = True
        return c

    adapter.trainer_config = trainer
    if r.kind.startswith("dolci"):
        adapter.SFT_DATA = DATA
        adapter.SFT_CACHE = DATA / "packing-cache"
        adapter.SFT_DATA_PLAN = AUTO / "dolci-data-plan.json"


def train():
    r = adapter.current()
    install_adapters(r)
    adapter.hero.qualified.apply_policy()
    adapter.main(config_builder=adapter.builder(r))
