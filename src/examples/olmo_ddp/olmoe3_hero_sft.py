"""Exact small LC architecture with packed assistant-only SFT and matched LR sweeps."""

import hashlib
import json
import math
import os
import sys
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import ClassVar

import olmoe3_small_hero as hero
import torch
import torch.distributed as dist
from olmoe3_hero_sft_data import SFTPackedDatasetConfig, self_test
from olmoe3_hero_sft_plan import (
    AUTOMATION,
    BASELINE_CAMPAIGN,
    BATCH,
    CACHE,
    CAMPAIGN,
    CONTROL,
    DATA,
    DATA_PLAN,
    GPUS,
    LEGACY_SOURCE,
    MOUNT,
    SEED,
    SEQUENCE,
    SOURCE,
    data_plan,
    find_run,
    runs,
)
from olmoe3_lr_sweep_watch import atomic_json, log

from olmo_core.data import NumpyDataLoaderConfig, TokenizerConfig
from olmo_core.data.utils import get_labels
from olmo_core.distributed.utils import get_rank, get_world_size
from olmo_core.internal.experiment import (
    CliContext,
    DataComponents,
    SubCmd,
    build_config,
    main,
)
from olmo_core.optim.scheduler import LinearWithWarmup
from olmo_core.train import Duration
from olmo_core.train.callbacks import Callback
from olmo_core.train.common import LoadStrategy
from olmo_core.train.utils import EnvRngStates
from olmo_core.utils import move_to_device

hero.find_run = find_run


def dataset_config(tokenizer, split):
    """Keep whole conversations, supervised masks and recurrent/attention boundaries."""
    assert split in ("train", "validation")
    directory = DATA / ("train-single-source" if split == "train" else split)
    return SFTPackedDatasetConfig.glob(
        str(directory / "token_ids_part_*.npy"),
        tokenizer=tokenizer,
        label_mask_paths=[str(directory / "labels_mask_part_*.npy")],
        sequence_length=SEQUENCE,
        generate_doc_lengths=True,
        source_group_size=8,
        work_dir=str(CACHE / split),
        instance_filter_config=None,
    )


def common_components(context, **kwargs):
    common = hero.common_components(context, **kwargs)
    common.work_dir = str(ROOT_WORK / common.run_name)
    assert common.tokenizer.bos_token_id is None
    return common


ROOT_WORK = CACHE.parent / "work"


def data_components(common):
    return DataComponents(
        dataset=dataset_config(common.tokenizer, "train"),
        data_loader=NumpyDataLoaderConfig(
            global_batch_size=BATCH,
            work_dir=str(ROOT_WORK / common.run_name / "loader"),
            seed=SEED,
            num_workers=4,
            prefetch_factor=2,
            num_threads=2,
        ),
    )


def model_config(common):
    # Preserve the integration checkpoint's exact architecture, including shared
    # QK gains, 15 layers and 4:1 hybridization. Do not substitute the hero model.
    from olmo_core.nn.transformer.config import OLMoDDPModelConfig

    source = find_run(common.run_name).source
    config = OLMoDDPModelConfig.from_dict(json.loads((source / "config.json").read_text())["model"])
    config.recompute_each_block = True
    config.recompute_all_blocks_by_chunk = False
    for block in [config.block, *config.block_overrides.values()]:
        mixer = block.sequence_mixer
        if hasattr(mixer, "use_cute_kernel"):
            mixer.use_cute_kernel = False
    assert not config.two_batch_overlap
    return config


def train_module_config(common):
    r = find_run(common.run_name)
    config = hero.train_module_config(common)
    config.rank_microbatch_size = SEQUENCE
    config.optim.lr = r.lr
    config.optim.weight_decay = 0.0
    config.scheduler = LinearWithWarmup(warmup_fraction=0.03, alpha_f=0.0)
    config.z_loss_multiplier = None
    # The compiled packed path produced nonfinite startup CE on both LC sources;
    # the identical eager diagnostic was finite on all eight ranks. Keep SFT eager
    # until this compiler/variable-length interaction is separately qualified.
    config.compile_model = False
    config.reset_optimizer_states_on_load = (
        Path(os.environ.get("HERO_SFT_LOAD", str(r.source))) == r.source
    )
    return config


def same(a, b):
    """Compare saved RNG/loader containers without ambiguous tensor truth values."""
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return (
            isinstance(b, (list, tuple))
            and len(a) == len(b)
            and all(same(x, y) for x, y in zip(a, b))
        )
    if torch.is_tensor(a):
        return torch.equal(a.cpu(), b.cpu())
    if hasattr(a, "shape"):
        return bool((a == b).all())
    return a == b


@dataclass
class SFTAudit(Callback):
    """Check 64->8 weights-only transfer, later full-state resumes and finite updates."""

    priority: ClassVar[int] = 10
    run_id: str = ""

    def post_checkpoint_loaded(self, path):
        r = find_run(self.run_id)
        actual = hero.state_sample(self.trainer)
        if Path(path) == r.source:
            from olmoe3_integration_sft_audit import verify_source_weights

            # Independently compare to the original, not only the repacked copy.
            verify_source_weights(self.trainer, LEGACY_SOURCE)
            assert self.step == self.trainer.global_train_tokens_seen == 0
            assert self.trainer.data_loader.tokens_processed == 0
            optim = self.trainer.train_module.optim
            assert not optim._losses and not optim._grad_norms
            moments = 0
            for name, value in optim.states.items():
                if name.endswith((".exp_avg", ".exp_avg_sq", ".step")):
                    value = value.to_local() if hasattr(value, "to_local") else value
                    assert not torch.count_nonzero(value).item(), name
                    moments += 1
            assert moments > 0
            proof = {
                "weights_and_buffers_sampled_exact": True,
                "optimizer_reset": True,
                "data_reset": True,
            }
        else:
            assert Path(path).parent == r.root
            saved = json.loads((Path(path) / f"resume_audit/rank{get_rank()}.json").read_text())
            assert same(saved, actual), "SFT full-state sample changed on resume"
            trainer_saved = torch.load(
                Path(path) / "train" / f"rank{get_rank()}.pt",
                map_location="cpu",
                weights_only=False,
            )
            assert same(trainer_saved["rng"], EnvRngStates.current_state().as_dict())
            assert same(trainer_saved["data_loader"], self.trainer.data_loader.state_dict())
            proof = {"sampled_state_exact": True, "rng_exact": True, "data_state_exact": True}
        atomic_json(r.root / "audit" / f"restore-step{self.step}-rank{get_rank()}.json", proof)

    def pre_train(self):
        r = find_run(self.run_id)
        assert get_world_size() == GPUS and MOUNT.is_mount()
        assert self.trainer.data_loader.total_batches == data_plan()["steps_per_epoch"]
        registration = json.loads((CONTROL / "registrations" / f"{r.run_id}.json").read_text())
        assert registration["checkpoint_root"] == str(r.root) and registration["enabled"]
        assert registration["bucket_id"] == r.bucket and registration["remote_prefix"] == r.prefix
        tm = self.trainer.train_module
        original_save = tm.save_state_dict_direct

        def audited_save(directory, **kwargs):
            before = hero.state_sample(self.trainer)
            original_save(directory, **kwargs)
            after = hero.state_sample(self.trainer)
            assert same(before, after), "Synchronous save mutated state"
            atomic_json(Path(directory).parent / "resume_audit" / f"rank{get_rank()}.json", after)

        tm.save_state_dict_direct = audited_save
        self.first_batch = True
        if get_rank() == 0:
            atomic_json(
                r.root / "audit" / f"session-{os.environ.get('BEAKER_JOB_ID')}-{self.step}.json",
                dict(
                    **r.as_dict(),
                    source_commit=os.environ.get("GIT_REF"),
                    start=self.step,
                    data_plan=data_plan(),
                ),
            )

    def pre_step(self, batch):
        if self.first_batch:
            ids, mask = batch["input_ids"], batch["label_mask"]
            assert "doc_lens" in batch and mask.dtype == torch.bool
            assert mask.any() and (~mask).any() and not mask[ids == 100277].any()
            assert not mask[:, 0].any()
            input_sha256 = hashlib.sha256(ids.detach().cpu().numpy().tobytes()).hexdigest()
            r = find_run(self.run_id)
            if self.step == 1 and not r.smoke:
                smoke_run = next(x for x in runs(True) if x.arm == r.arm)
                baseline = json.loads(
                    (smoke_run.root / "audit" / f"batch-step1-rank{get_rank()}.json").read_text()
                )
                assert input_sha256 == baseline["input_sha256"], "Changed packed SFT data order"
                assert int(mask.sum()) == baseline["supervised_tokens"]
            atomic_json(
                find_run(self.run_id).root
                / "audit"
                / f"batch-step{self.step}-rank{get_rank()}.json",
                {
                    "supervised_tokens": int(mask.sum()),
                    "doc_lengths_present": True,
                    "input_sha256": input_sha256,
                },
            )
            self.first_batch = False

    def log_metrics(self, step, metrics):
        for key, value in metrics.items():
            if key in ("train/CE loss", "optim/total grad norm"):
                assert math.isfinite(float(value)), (step, key, value)
        if get_rank() == 0:
            with (find_run(self.run_id).root / "audit/metrics.jsonl").open("a") as handle:
                handle.write(json.dumps(dict(step=step, **metrics)) + "\n")


@dataclass
class SFTValidation(Callback):
    """Global assistant-token-weighted CE with packed document isolation, no dropped tail."""

    run_id: str = ""

    def pre_train(self):
        if os.environ.get("HERO_SFT_DIAGNOSTIC") == "1":
            install_diagnostic_hooks(self.trainer.train_module.model)
        self.dataset = dataset_config(
            (
                self.trainer.data_loader.dataset.tokenizer
                if hasattr(self.trainer.data_loader.dataset, "tokenizer")
                else TokenizerConfig.dolma2()
            ),
            "validation",
        ).build()
        self.dataset.prepare()
        self.evaluate()

    def post_step(self):
        if self.step % 200 == 0 or self.step == data_plan()["steps_per_epoch"]:
            self.evaluate()

    def post_train(self):
        self.evaluate()

    def evaluate(self):
        r = find_run(self.run_id)
        totals = torch.zeros(2, dtype=torch.float64, device=self.trainer.device)
        n = len(self.dataset)
        batches = 1 if r.smoke else math.ceil(n / GPUS)
        with torch.no_grad():
            for batch_index in range(batches):
                index = batch_index * GPUS + get_rank()
                batch = self.trainer.data_loader.collator([self.dataset[index % n]])
                batch = move_to_device(batch, self.trainer.device)
                labels = get_labels(batch, label_ignore_index=-100)
                output = self.trainer.train_module.eval_batch(batch, labels=labels)
                if os.environ.get("HERO_SFT_DIAGNOSTIC") == "1":
                    valid = labels != -100
                    log(
                        "SFT_DIAGNOSTIC_LOSS",
                        rank=get_rank(),
                        labels=int(valid.sum()),
                        nonfinite=int((~torch.isfinite(output.ce_loss[valid])).sum()),
                        total=int(valid.sum()),
                    )
                    dist.barrier()
                    raise SystemExit(0)
                if index < n:
                    valid = labels != -100
                    totals[0] += output.ce_loss[valid].double().sum()
                    totals[1] += valid.sum()
                del output
        dist.all_reduce(totals, group=self.trainer.dp_process_group)
        assert totals[1] > 0 and torch.isfinite(totals).all()
        ce = float((totals[0] / totals[1]).item())
        self.trainer.record_metric("eval/sft-validation/assistant CE loss", ce)
        if get_rank() == 0:
            with (r.root / "audit/validation.jsonl").open("a") as handle:
                handle.write(
                    json.dumps(
                        {
                            "step": self.step,
                            "ce_loss": ce,
                            "supervised_tokens": int(totals[1]),
                            "packed_batches": batches,
                            "smoke": r.smoke,
                        }
                    )
                    + "\n"
                )
        log("SFT_VALIDATION", run=r.run_id, step=self.step, ce_loss=ce)


def trainer_config(common):
    r = find_run(common.run_name)
    config = hero.trainer_config(common)
    for key in ("hero_audit", "hero_complete", "lm_evaluator"):
        config.callbacks.pop(key, None)
    source = Path(os.environ.get("HERO_SFT_LOAD", str(r.source)))
    fresh = source == r.source
    config.load_path = str(source)
    config.load_strategy = LoadStrategy.always
    config.load_optim_state = not fresh
    config.load_trainer_state = not fresh
    config.max_duration = Duration.epochs(r.epochs)
    stop = int(os.environ.get("HERO_SFT_STOP", str(4 if r.smoke else r.total_steps)))
    config.hard_stop = Duration.steps(stop)
    config.metrics_collect_interval = 1 if r.smoke else 10
    cp = config.callbacks["checkpointer"]
    cp.save_interval = None
    cp.fixed_steps = [2, 4] if r.smoke else r.checkpoint_steps
    config.callbacks["sft_audit"] = SFTAudit(run_id=r.run_id)
    config.callbacks["sft_validation"] = SFTValidation(run_id=r.run_id)
    wb = config.callbacks["wandb"]
    wb.group = CAMPAIGN + ("-smoke" if r.smoke else "")
    wb.tags = [
        r.arm,
        "pretrain-emo" if r.arm == "emo" else "pretrain-noemo",
        "midtrain-emo-disabled",
        "long-context-emo-disabled",
        "source-emo-disabled",
        "sft-emo-disabled",
        "sft",
        "gptoss120b-high",
        f"{r.epochs}-epochs",
        "8g",
        "512ki",
        "64k-packed",
        "fla",
        "assistant-masks",
        "block-recompute",
        "lr-" + r.lr_label,
    ]
    wb.notes = json.dumps(r.as_dict())
    if os.environ.get("HERO_SFT_DIAGNOSTIC") == "1":
        wb.enabled = False
    return config


def config_builder():
    return partial(
        build_config,
        global_batch_size=BATCH,
        max_sequence_length=SEQUENCE,
        num_nodes=1,
        common_config_builder=common_components,
        data_config_builder=data_components,
        model_config_builder=model_config,
        train_module_config_builder=train_module_config,
        trainer_config_builder=trainer_config,
        beaker_image=hero.qualified.base.BEAKER_IMAGE,
        beaker_workspace=hero.WORKSPACE,
        include_default_evals=False,
        num_execution_units=1,
    )


def prepare():
    """Prepare packing and verify both two-epoch configs and their restart smokes."""
    assert MOUNT.is_mount() and DATA.is_dir()
    from olmoe3_integration_sft_repack import repack

    repack(LEGACY_SOURCE, SOURCE, GPUS)
    self_test()
    manifest = json.loads((DATA / "manifest.json").read_text())
    assert manifest["status"] == "complete"
    assert json.loads((DATA / "train-single-source/READY.json").read_text())["passed"]
    tok = TokenizerConfig.dolma2()
    lengths = {}
    for split in ("train", "validation"):
        dataset = dataset_config(tok, split).build()
        dataset.prepare()
        lengths[split] = len(dataset)
        for idx in {0, len(dataset) - 1, len(dataset) // 2}:
            item = dataset[idx]
            assert item["input_ids"].shape == (SEQUENCE,)
            assert item["label_mask"].dtype == torch.bool and item["label_mask"].any()
            assert "doc_lens" in item
    steps = lengths["train"] // GPUS
    plan = {
        "passed": True,
        "source_commit": os.environ["GIT_REF"],
        "batch_tokens": BATCH,
        "sequence_length": SEQUENCE,
        "packed_instances": lengths,
        "steps_per_epoch": steps,
        "total_steps": steps * 2,
        "total_steps_by_epochs": {"2": steps * 2},
        "raw_training_tokens": manifest["splits"]["train"]["input_tokens"],
        "dropped_packed_instances_per_epoch": lengths["train"] % GPUS,
        "manifest_sha256": hashlib.sha256((DATA / "manifest.json").read_bytes()).hexdigest(),
    }
    atomic_json(DATA_PLAN, plan)
    baseline_automation = MOUNT / "uploader/automation" / BASELINE_CAMPAIGN
    assert manifest["dataset"] == "jacobmorrison/length-investigation-gptoss-120b-high"
    assert manifest["dataset_revision"] == "2fa53f4df6e4e41f9202c31cc8e26b2a04bce027"
    assert manifest["template"] == "olmo_thinker_no_think_sft_tokenization"
    for r in runs() + runs(True):
        config = config_builder()(CliContext(__file__, SubCmd.dry_run, r.run_id, "ai2/holmes", []))
        assert config.model.n_layers == 15 and config.model.d_model == 1024
        assert config.data_loader.global_batch_size == BATCH
        assert config.train_module.rank_microbatch_size == SEQUENCE
        assert config.dataset.generate_doc_lengths and config.dataset.label_mask_paths
        assert config.train_module.optim.lr == r.lr and config.train_module.optim.weight_decay == 0
        assert config.train_module.z_loss_multiplier is None
        assert not config.trainer.load_optim_state and not config.trainer.load_trainer_state
        assert config.model.recompute_each_block
        # Match on-disk JSON's string keys (e.g. integer block override indices).
        current = json.loads(json.dumps(config.as_dict(json_safe=True)))
        old_name = f"{BASELINE_CAMPAIGN}-{'smoke-' if r.smoke else ''}{r.arm}-lr{r.lr_label}"
        baseline = json.loads((baseline_automation / "configs" / f"{old_name}.json").read_text())
        assert current["model"]["n_layers"] == 15
        assert current["train_module"] == baseline["train_module"], "Changed training recipe"
        assert current["data_loader"]["seed"] == SEED
        assert current["trainer"]["max_duration"]["value"] == r.epochs
        atomic_json(AUTOMATION / "configs" / f"{r.run_id}.json", current)
    atomic_json(
        AUTOMATION / "config-success.json",
        {
            "passed": True,
            "source_commit": os.environ["GIT_REF"],
            "runs": [r.as_dict() for r in runs()],
        },
    )
    log("SFT_CONFIG_DATA_GATE_PASSED", **plan)


def attention_smoke():
    """Check packed FA4 against independent documents and a float64 SDPA reference."""
    import torch.nn.functional as F

    from olmo_core.nn.attention.flash_attn_api import dispatch_flash_attn_4

    torch.manual_seed(SEED)
    q = torch.randn(1, 13, 8, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(1, 13, 4, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn_like(k, requires_grad=True)
    cu = torch.tensor([0, 4, 13], device="cuda", dtype=torch.int32)
    out = dispatch_flash_attn_4(q, k, v, cu_seqlens=cu, max_seqlen=9, causal=True)
    out = out.reshape_as(q)
    pieces = [
        dispatch_flash_attn_4(q[:, a:b], k[:, a:b], v[:, a:b], causal=True)
        for a, b in ((0, 4), (4, 13))
    ]
    torch.testing.assert_close(out, torch.cat(pieces, dim=1), atol=2e-2, rtol=2e-2)
    qr, kr, vr = [x.detach().cpu().double().requires_grad_() for x in (q, k, v)]
    refs = [
        F.scaled_dot_product_attention(
            qr[:, a:b].transpose(1, 2),
            kr[:, a:b].transpose(1, 2),
            vr[:, a:b].transpose(1, 2),
            is_causal=True,
            enable_gqa=True,
        ).transpose(1, 2)
        for a, b in ((0, 4), (4, 13))
    ]
    ref = torch.cat(refs, dim=1)
    torch.testing.assert_close(out.detach().cpu().double(), ref, atol=2e-2, rtol=2e-2)
    out.float().square().sum().backward()
    ref.square().sum().backward()
    errors = {}
    for name, actual, expected in zip(("q", "k", "v"), (q, k, v), (qr, kr, vr)):
        grad = actual.grad.cpu().double()
        relative_rms = float(
            ((grad - expected.grad).square().mean() / expected.grad.square().mean()).sqrt()
        )
        assert torch.isfinite(grad).all() and relative_rms < 0.02, (name, relative_rms)
        errors[name] = relative_rms
    log(
        "SFT_PACKED_FA4_GATE_PASSED",
        gradient_relative_rms=errors,
        maximum_output_error=float((out.detach().cpu().double() - ref.detach()).abs().max()),
    )


def install_diagnostic_hooks(model):
    """Read-only eager forward diagnostics; never accepted as a training smoke."""
    import olmo_core.nn.attention.kda as kda_module

    def stats(name, tensor):
        if isinstance(tensor, (tuple, list)):
            tensor = tensor[0]
        if not torch.is_tensor(tensor):
            return
        finite = torch.isfinite(tensor)
        clean = torch.where(finite, tensor.float(), 0.0)
        log(
            "SFT_DIAGNOSTIC_ACTIVATION",
            rank=get_rank(),
            name=name,
            shape=list(tensor.shape),
            dtype=str(tensor.dtype),
            nonfinite=int((~finite).sum()),
            minimum=float(clean.min()),
            maximum=float(clean.max()),
        )

    original = kda_module.dispatch_chunk_kda
    calls = 0

    def kda_probe(**kwargs):
        nonlocal calls
        number = calls
        calls += 1
        for name in ("q", "k", "v", "g", "beta", "A_log", "dt_bias", "cu_seqlens"):
            stats(f"kda{number}/input/{name}", kwargs.get(name))
        result = original(**kwargs)
        stats(f"kda{number}/output", result)
        return result

    kda_module.dispatch_chunk_kda = kda_probe
    for name, module in model.named_modules():
        if (
            name == "lm_head"
            or name == "embeddings"
            or name.count(".") == 1
            and name.startswith("blocks.")
        ):
            module.register_forward_hook(lambda _module, _args, result, n=name: stats(n, result))


if __name__ == "__main__":
    hero.qualified.apply_policy()
    if sys.argv[1:] == ["--prepare"]:
        prepare()
    elif sys.argv[1:] == ["--attention-smoke"]:
        attention_smoke()
    else:
        main(config_builder=config_builder())
