"""Historical PT recipe with explicit reference routing and audited 128-to32 restore."""

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

import olmoe3_adaptive_decay_plan as p
from olmoe3_lr_sweep_watch import atomic_json

from olmo_core.distributed.utils import get_rank, get_world_size
from olmo_core.nn.moe.v2.router import MoERouterV2
from olmo_core.optim.scheduler import WSD
from olmo_core.train.callbacks import ConfigSaverCallback

# Bind the resolver before importing the historical adapter (which imports names).
p.install()
import olmoe3_qkgain_train as adapter  # noqa: E402  # isort: skip


def fingerprint(tensor):
    """Match the protected checkpoint's exact sampled-state convention."""
    value = tensor.detach().reshape(-1)
    assert value.numel()
    count = min(128, value.numel())
    offsets = adapter.torch.tensor(
        [i * (value.numel() - 1) // max(1, count - 1) for i in range(count)],
        device=value.device,
    )
    data = value[offsets].contiguous().view(adapter.torch.uint8).cpu().numpy().tobytes()
    return [value.numel(), str(value.dtype), hashlib.sha256(data).hexdigest()]


def verify_initial_reshard(trainer, path):
    """Verify all original optimizer pieces and unchanged model/data/skip-step state."""
    assert Path(path) == p.SOURCE and get_world_size() == p.GPUS
    tm = trainer.train_module
    assert not tm.ep_enabled and not tm.pp_enabled
    rank = get_rank()
    assert p.SOURCE_GPUS % p.GPUS == 0
    shard_factor = p.SOURCE_GPUS // p.GPUS
    references = {}

    def saved(old_rank):
        if old_rank not in references:
            row = json.loads((Path(path) / "resume_audit" / f"rank{old_rank}.json").read_text())
            assert (row["gpus"], row["rank"], row["step"]) == (p.SOURCE_GPUS, old_rank, p.START)
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
    pieces = 0
    for name, tensor in tensors.items():
        value = tensor.to_local() if hasattr(tensor, "to_local") else tensor
        value = value.detach().reshape(-1)
        old = reference["tensors"][name]
        if value.numel() == old[0]:
            assert actual["tensors"][name] == old, ("Unsharded restore mismatch", name)
            continue
        assert name in tm.optim.states and value.numel() == shard_factor * old[0], (
            "Unexpected EP1 reshard geometry",
            name,
            value.numel(),
            old[0],
        )
        for piece in range(shard_factor):
            expected = saved(shard_factor * rank + piece)["tensors"][name]
            assert fingerprint(value.narrow(0, piece * old[0], old[0])) == expected, (
                "Optimizer reshard mismatch",
                name,
                rank,
                piece,
            )
            pieces += 1
    assert pieces > 0
    state = adapter.torch.load(
        Path(path) / "train" / f"rank{rank}.pt", map_location="cpu", weights_only=False
    )
    assert adapter.equal(state["data_loader"], trainer.data_loader.state_dict())
    assert trainer.data_loader.tokens_processed == actual["tokens"] == p.START * p.BATCH
    return dict(
        source_gpus=p.SOURCE_GPUS,
        gpus=p.GPUS,
        sampled_state_exact=True,
        verified_old_shard_pieces=pieces,
        source_shards_per_current_rank=shard_factor,
        state_keys=len(tensors),
        rng_policy="Trainer reinitializes per-rank RNG at changed world size",
    )


@dataclass
class DecayAudit(adapter.Audit):
    """Guard initialization, routing transitions, checkpoint metadata and exact resumes."""

    _last_k: int = 0

    def post_checkpoint_loaded(self, path):
        if Path(path) != p.SOURCE:
            return super().post_checkpoint_loaded(path)
        assert self.step == p.START
        proof = verify_initial_reshard(self.trainer, path)
        r = p.find_run(self.run_id)
        atomic_json(
            r.root / "audit" / f"restore-{self.step}-rank{get_rank()}.json",
            dict(passed=True, source=str(path), fresh_stage=False, **proof),
        )

    def _set_k(self, update):
        r = p.find_run(self.run_id)
        k = p.expert_count(r.schedule, update)
        routers = [
            m
            for m in self.trainer.train_module.model.modules()
            if isinstance(m, MoERouterV2) and m.num_experts == 512
        ]
        assert len(routers) == 15
        for router in routers:
            assert router.reference_top_k == 16 and router.restore_weight_scale
            assert router.normalize_expert_weights == 1.0 and router.original_top_k is None
            assert not router.use_recompute_cache
            router.top_k = k
        if k != self._last_k:
            # ConfigSaver writes this dictionary into every completed checkpoint.
            # Keep the saved current K aligned with the actual dispatch at that step.
            for callback in self.trainer.callbacks.values():
                if isinstance(callback, ConfigSaverCallback) and callback.config is not None:

                    def update_config(value):
                        if isinstance(value, dict):
                            if value.get("reference_top_k") == 16:
                                value["top_k"] = k
                            for child in value.values():
                                update_config(child)
                        elif isinstance(value, list):
                            for child in value:
                                update_config(child)

                    update_config(callback.config["model"])
            if get_rank() == 0:
                atomic_json(
                    r.root / "audit" / f"routing-{update}.json",
                    dict(
                        step=update,
                        top_k=k,
                        reference_top_k=16,
                        multiplier=16,
                        routed_layers=15,
                        dense_first_layer=True,
                        shared_experts="unchanged",
                        dispatch="narrow",
                    ),
                )
                print("ADAPTIVE_K_TRANSITION", update, k, flush=True)
            self._last_k = k

    def pre_train(self):
        super().pre_train()
        self._set_k(min(p.END, self.step + 1))

    def pre_step(self, batch):
        self._set_k(self.step)
        super().pre_step(batch)


def install_adapters(r):
    """Change only experiment identities, resize checks and the K intervention."""
    adapter.scheduler = lambda _: WSD(warmup=2000, decay=p.END - p.START, decay_fraction=None)
    original_common = adapter.common_components
    original_model = adapter.model_config
    original_trainer = adapter.trainer_config

    def common(ctx, **kwargs):
        c = original_common(ctx, **kwargs)
        c.work_dir = str(p.ROOT / "data-work")
        return c

    def model(common):
        c = original_model(common)
        for block in [c.block, *c.block_overrides.values()]:
            router = getattr(block, "routed_experts_router", None)
            if router is not None:
                assert router.num_experts == 512 and router.top_k == 16 and router.emo is None
                router.reference_top_k = 16
                router.top_k = p.expert_count(
                    r.schedule, min(p.END, int(os.environ["QKGAIN_START"]) + 1)
                )
        c.validate()
        return c

    def trainer(common):
        c = original_trainer(common)
        c.callbacks["qkgain_audit"] = DecayAudit(run_id=r.run_id)
        c.callbacks["wandb"].project = "adaptive-compute"
        c.callbacks["wandb"].tags += [r.schedule, "reference-top16", f"{r.gpus}g"]
        if int(os.environ["QKGAIN_STOP"]) <= p.START + 4:
            c.metrics_collect_interval = 1
            c.no_evals = True
        return c

    adapter.common_components, adapter.model_config, adapter.trainer_config = (
        common,
        model,
        trainer,
    )


def train():
    """Enter the pinned trainer with this campaign's configuration adapter."""
    r = adapter.current()
    install_adapters(r)
    adapter.hero.qualified.apply_policy()
    adapter.main(config_builder=adapter.builder(r))


def validate(schedule):
    """Build and serialize the real training configuration without allocating a model."""
    from olmo_core.internal.experiment import CliContext, SubCmd

    r = p.run(schedule)
    os.environ.update(QKGAIN_RUN=r.run_id, QKGAIN_START=str(p.START), QKGAIN_STOP=str(p.START + 2))
    install_adapters(r)
    adapter.hero.qualified.apply_policy()
    c = adapter.builder(r)(CliContext(p.SCRIPT, SubCmd.dry_run, r.run_id, "ai2/holmes", [])).merge(
        []
    )
    assert (
        c.data_loader.global_batch_size == p.BATCH and c.train_module.rank_microbatch_size == 32768
    )
    assert c.trainer.load_optim_state and c.trainer.load_trainer_state
    assert not c.train_module.reset_optimizer_states_on_load
    assert c.trainer.load_path == str(p.SOURCE) and c.trainer.max_duration.value == p.END
    assert c.train_module.scheduler.get_lr(r.lr, p.START, p.END) == r.lr
    assert c.train_module.scheduler.get_lr(r.lr, p.END, p.END) == 0
    assert c.train_module.ep_config is None and c.train_module.pp_config is None
    assert c.trainer.callbacks["checkpointer"].fixed_steps == r.saves
    atomic_json(p.AUTO / "configs" / f"{schedule}.json", c.as_dict(json_safe=True))
    print("ADAPTIVE_DECAY_CONFIG_PASSED", schedule, flush=True)
