"""Micro-anneal recipe — added for sftlab's olmo_core_anneal backend.
 One script serves any base (Qwen, OLMo, Llama, …) via --model_arch. Modeled on OLMo3-32B-midtraining.py:
anneal a fixed base checkpoint for a small token budget on a source_mixtures data mix, model-only
load (fresh optimizer) with an explicit decaying LR.

CLI (internal-experiment style, plus sftlab build-time flags this module pre-parses):
    micro_anneal.py <launch|train|dry_run> <run_name> <cluster> \
        --base_checkpoint=<olmo-core DCP dir>  --model_arch=qwen3_8B \
        --tokenizer=Qwen/Qwen3-8B-Base  --source_mixture_yaml=<mix>.yaml \
        --length_tokens=10000000000  --seq_len=4096  --peak_lr=1e-5 \
        --lr_schedule=linear_with_warmup  --warmup_steps=0 \
        --load_optim_state=false  --load_trainer_state=false \
        --save_folder=<weka dir>  --save_freq=0  --seed=42  --num_nodes=1 \
        [--launch.priority=high --launch.workspace=… --launch.beaker_image=… …]

The build-time flags (above the launch/dotted ones) must be known before the ExperimentConfig is
built (they pick the arch, mix, LR, load behaviour), so they cannot ride the post-hoc
`.merge(overrides)` path — this module strips them from argv and stashes them, then hands the rest
(<cmd> <run_name> <cluster> + any --launch.*/--trainer.* dotted overrides) to olmo-core's `main`.
They are re-appended to the remote `launch` command so the Beaker-side `train` hop sees them too.

"""

import re
import sys

# ── pre-parse sftlab build-time flags out of argv (before olmo_core.main sees it) ──────────────
_BUILD_KEYS = {
    "base_checkpoint", "model_arch", "tokenizer", "source_mixture_yaml", "source_mixture_b64",
    "length_tokens", "seq_len", "peak_lr", "lr_schedule", "warmup_steps", "load_optim_state",
    "load_trainer_state", "save_folder", "save_freq", "ephemeral_save_freq", "seed", "num_nodes",
    "global_batch_size", "router_emo",
}
_BUILD: dict[str, str] = {}
_build_argv: list[str] = []   # the stripped flags, re-appended to the remote launch cmd
_kept_argv: list[str] = []
for _a in sys.argv[1:]:
    _m = re.match(r"--([a-zA-Z_][\w]*)=(.*)$", _a)
    if _m and _m.group(1) in _BUILD_KEYS:
        _BUILD[_m.group(1)] = _m.group(2)
        _build_argv.append(_a)
    else:
        _kept_argv.append(_a)
sys.argv = [sys.argv[0], *_kept_argv]


def _bool(v: str) -> bool:
    return str(v).lower() in ("1", "true", "yes")


# Imports AFTER the argv strip (they're heavy; keeping them here also documents the olmo-core API).
import base64  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
from datetime import datetime  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Optional  # noqa: E402

import yaml  # noqa: E402

from olmo_core.config import Config  # noqa: E402
from olmo_core.data import (  # noqa: E402
    InstanceFilterConfig,
    NumpyDataLoaderConfig,
    NumpyFSLDatasetConfig,
    TokenizerConfig,
)
from olmo_core.data.source_mixture import SourceMixtureDatasetConfig, SourceMixtureList  # noqa: E402
from olmo_core.internal import cookbook  # noqa: E402
from olmo_core.internal.common import build_launch_config, get_root_dir, get_work_dir  # noqa: E402
from olmo_core.internal.experiment import CliContext, ExperimentConfig, SubCmd, main  # noqa: E402
from olmo_core.launch.beaker import BeakerEnvVar, BeakerWekaBucket, OLMoCoreBeakerImage  # noqa: E402
from olmo_core.launch.beaker_presets import get_preset  # noqa: E402
from olmo_core.nn.transformer import TransformerConfig  # noqa: E402
from olmo_core.nn.transformer.config import OLMoDDPModelConfig  # noqa: E402
from olmo_core.optim.scheduler import LinearWithWarmup, SchedulerUnits, WSD  # noqa: E402
from olmo_core.train import Duration  # noqa: E402

# Fallback only. The batch size belongs to the recipe being continued, not to this script — an
# anneal that changes it is no longer the same optimization as the run it resumes — so callers pass
# --global_batch_size (sftlab sends the profile's). ~4M tokens when they don't.
DEFAULT_GLOBAL_BATCH_SIZE = 4 * 1024 * 1024


log = logging.getLogger(__name__)


def _wandb_run_id(save_folder: str) -> str:
    """A W&B run id keyed to the checkpoint folder, so the W&B run tracks the training run.

    The trainer resumes from whatever it finds in `save_folder`, which makes that path the exact
    condition under which metrics should continue on one curve: same folder means the restart picks
    up mid-training, a new folder means training starts over from the base checkpoint and deserves
    its own run. Deriving the id from anything process-local instead — the default, a timestamp —
    splits a preempted job into one short run per attempt.
    """
    return hashlib.sha256(save_folder.encode()).hexdigest()[:16]


def _verify_peak_lr(base_checkpoint: str, peak_lr: float) -> None:
    """Check the declared peak LR against the one in the base checkpoint's optimizer state.

    Logged, never fatal, because two different operations both land in this script and they want
    different rates. OLMo3-7B-anneal.py takes no LR at all: it reads `optim.param_groups.<param>.lr`
    out of `model_and_optim` and decays from the rate the run was actually training at. A midtraining
    stage instead sets its own rate and re-warms to it, which is what OLMo3-7B-midtraining.py does
    and what the olmo3-7b-anneal profile reproduces at 2.07e-4 against a base whose live rate is
    ~3.9e-5. Neither is a mistake; annealing a trunk and emulating a midtrain stage are simply not
    the same op.

    So this reports the comparison rather than enforcing one reading of it. A gap means "check that
    you meant a midtrain-style re-warm", and a match means the anneal continues the trunk. What it
    catches is the third case, a profile that drifted from a base it was written against.
    """
    from torch.distributed.checkpoint.api import CheckpointException

    from olmo_core.distributed.checkpoint import load_state_dict
    from olmo_core.io import join_path

    for param in ("embeddings.weight", "embeddings.embedding.weight"):
        key = f"optim.param_groups.{param}.lr"
        state: dict[str, Optional[float]] = {key: None}
        try:
            load_state_dict(join_path(base_checkpoint, "model_and_optim"), state)
        # CheckpointException derives from BaseException, not Exception. A base whose optimizer
        # names its param groups differently (OLMoDDP) raises it for a missing key.
        except (Exception, CheckpointException) as e:
            log.warning(f"could not read the base optimizer state to verify peak_lr ({e})")
            return
        found = state[key]
        if found is None:
            continue
        found = float(found)
        # Relative tolerance, not equality: the profile carries a rounded decimal of a value the
        # optimizer holds in full precision.
        if abs(found - peak_lr) / max(found, 1e-12) > 1e-6:
            log.warning(
                f"peak_lr {peak_lr} differs from the base optimizer state's {found} ({param}), a "
                f"{peak_lr / found:.2f}x re-warm. Expected when this run emulates a midtraining "
                f"stage with its own rate; a mistake if it meant to decay the trunk from where the "
                f"base actually was, in which case pass --peak_lr={found}.")
        else:
            log.info(f"peak_lr {peak_lr} matches the base optimizer state ({param})")
        return
    log.warning("base optimizer state has no embeddings param group; peak_lr left unverified")


def _scheduler(warmup_steps: int, kind: str):
    """Decay-to-zero schedule. 'linear_with_warmup' -> LinearWithWarmup(alpha_f=0); 'wsd' -> WSD."""
    if kind == "wsd":
        return WSD(units=SchedulerUnits.steps, warmup=warmup_steps, warmup_fraction=None,
                   decay=None, decay_fraction=0.1)
    return LinearWithWarmup(units=SchedulerUnits.steps, warmup=warmup_steps, alpha_f=0.0)


def _disable_router_emo(model_config) -> None:
    """Route with the full expert set: drop EMO document expert pools from every routed block.

    Follows the olmoe3 ladder, whose downstream stages (midtraining, long context) run with EMO off
    while leaving global load balancing and every other router setting as the base trained them.
    """
    blocks = [model_config.block, *(model_config.block_overrides or {}).values()]
    routers = [r for blk in blocks if (r := getattr(blk, "routed_experts_router", None)) is not None]
    if not routers:
        raise ValueError("--router_emo=false, but the arch has no routed MoE blocks")
    for router in routers:
        router.emo = None


def build_experiment_config(cli_context: CliContext) -> ExperimentConfig:
    b = _BUILD
    seq_len = int(b["seq_len"])
    seed = int(b.get("seed", 42))

    global_batch_size = int(b.get("global_batch_size") or DEFAULT_GLOBAL_BATCH_SIZE)
    run_ts = f"{cli_context.run_name}-{datetime.now().astimezone().strftime('%Y%m%dT%H%M%S%z')}"
    root_dir = get_root_dir(cli_context.cluster)
    work_dir = get_work_dir(root_dir)

    # Model + tokenizer: base-agnostic. --model_arch selects the peer TransformerConfig classmethod
    # (qwen3_8B, olmo3_7B, …). The tokenizer (hence vocab_size) is either a NAMED olmo-core config
    # (OLMo/dolma bases — allenai/dolma2-tokenizer is tokenizer-only on HF, so from_hf 404s on its
    # missing model config.json) or an HF *model* repo id (Qwen3-8B-Base has a config.json).
    _NAMED_TOKENIZERS = {
        "allenai/dolma2-tokenizer": TokenizerConfig.dolma2,
        "allenai/dolma2-tokenizer-sigdig": TokenizerConfig.dolma2_sigdig,
    }
    _named = _NAMED_TOKENIZERS.get(b["tokenizer"])
    tokenizer_config = _named() if _named else TokenizerConfig.from_hf(b["tokenizer"])

    scheduler = _scheduler(int(b.get("warmup_steps", 0)), b.get("lr_schedule", "linear_with_warmup"))
    if b["model_arch"].endswith(".json"):
        # A base whose architecture no TransformerConfig classmethod describes: an MoE with per-layer
        # block_overrides, a linear-attention sequence mixer, a router. Rebuilding it from a preset
        # would mean maintaining a second description of the same model and being silently wrong
        # when they disagree, so the `model` and `train_module` blocks are lifted verbatim from the
        # base checkpoint's own config.json into a file under configs/ and rebuilt from there.
        #
        # THE TRAIN MODULE COMES WITH IT, not just the model. An MoE base is trained by its own
        # train module and optimizer (OLMoDDPTrainModuleConfig + OLMoDDPOptimizerConfig, with param
        # groups for the routed experts); cookbook.configure_train_module builds a plain
        # SkipStepAdamW, whose state a `--load_optim_state=true` resume cannot match. Only the LR
        # and the schedule are overridden here, plus EMO routing when --router_emo=false.
        #
        # A FILE IN THE REPO rather than a read of the checkpoint: the builder runs on both hops and
        # the launch host has no /weka mount, so reading the checkpoint here would fail before
        # anything is submitted. gantry clones the repo into the job, so both hops see this file.
        arch_path = Path(b["model_arch"])
        if not arch_path.is_file():   # else relative to this recipe, e.g. configs/<base>.json
            arch_path = Path(__file__).parent / arch_path
        arch = json.loads(arch_path.read_text())
        model_config = TransformerConfig.from_dict(arch["model"])
        if not _bool(b.get("router_emo", "true")):
            _disable_router_emo(model_config)
        train_module_config = Config.from_dict(arch["train_module"])
        train_module_config.optim.lr = float(b["peak_lr"])
        train_module_config.scheduler = scheduler
        # The mix is tokenized with one vocab; a base built for another would train on garbage ids.
        if model_config.vocab_size != tokenizer_config.padded_vocab_size():
            raise ValueError(
                f"{b['model_arch']} declares vocab_size {model_config.vocab_size}, but --tokenizer="
                f"{b['tokenizer']} pads to {tokenizer_config.padded_vocab_size()}. The base and the "
                f"tokenized mix must agree.")
    else:
        model_config = getattr(TransformerConfig, b["model_arch"])(
            vocab_size=tokenizer_config.padded_vocab_size(),
        )
        train_module_config = cookbook.configure_train_module(
            max_sequence_length=seq_len,
            rank_microbatch_size=seq_len,
            learning_rate=float(b["peak_lr"]),
            scheduler=scheduler,
        )

    # The varied ingredient: a source_mixtures spec (sources + target_ratio summing to 1.0).
    # Prefer the INLINED base64 YAML: sftlab passes the mix CONTENTS in the command, so neither the
    # launch host nor the Beaker node has to open a shared file (the launcher re-runs this builder
    # on BOTH hops). This mirrors how the SFT launcher carries unopened path strings, and keeps the
    # mix authored locally in the sftlab repo. Fall back to a file path for manual/legacy runs.
    if b.get("source_mixture_b64"):
        mix_dict = yaml.safe_load(base64.b64decode(b["source_mixture_b64"]).decode())
        source_list = SourceMixtureList.from_dict(mix_dict)
    else:
        source_list = SourceMixtureList.from_yaml(b["source_mixture_yaml"])
    source_list.validate()
    dataset_config = NumpyFSLDatasetConfig.from_src_mix(
        src_mix=SourceMixtureDatasetConfig(
            source_list=source_list,
            requested_tokens=int(float(b["length_tokens"])),
            global_batch_size=global_batch_size,
            processes=16,
            seed=seed,
        ),
        tokenizer=tokenizer_config,
        work_dir=work_dir,
        sequence_length=seq_len,
        instance_filter_config=InstanceFilterConfig(
            repetition_max_period=13, repetition_min_period=1, repetition_max_count=32
        ),
    )
    data_loader_config = NumpyDataLoaderConfig(global_batch_size=global_batch_size, seed=seed, num_workers=4)

    save_freq = int(b.get("save_freq", 0))
    # sftlab is a pure Beaker-job INITIATOR: `launch`/`dry_run` run on a host with NO /weka mount,
    # so configure_trainer's dir_is_empty(base_checkpoint) preflight would false-positive — a
    # missing mount is indistinguishable from an empty dir (io.dir_is_empty returns True when the
    # dir doesn't exist) — and abort the launch before any job is submitted. Skip that preflight on
    # the initiator hop only; the remote `train` hop rebuilds this identical config WITH /weka
    # mounted and validates for real (and the trainer's own load_checkpoint fails loudly if the
    # base is genuinely absent). Keeps a launch touching only the Beaker API, never weka.
    _initiator = cli_context.cmd in (SubCmd.launch, SubCmd.dry_run, SubCmd.launch_prep)
    if not _initiator:
        # Only the remote `train` hop has /weka, so this is where the declared LR meets the
        # checkpoint. Before the trainer is built, so a mismatch costs the job's startup and not
        # its first step.
        _verify_peak_lr(b["base_checkpoint"], float(b["peak_lr"]))
    _saved_dir_is_empty = cookbook.dir_is_empty
    if _initiator:
        cookbook.dir_is_empty = lambda _p: False
    try:
        trainer_config = cookbook.configure_trainer(
            # The FIXED base checkpoint. A HF->core converted base has model weights only, so
            # load_optim_state=false (fresh optimizer) and load_trainer_state=false (fresh data pass).
            load_path=b["base_checkpoint"],
            load_trainer_state=_bool(b.get("load_trainer_state", "false")),
            load_optim_state=_bool(b.get("load_optim_state", "false")),
            max_duration=Duration.tokens(int(float(b["length_tokens"]))),
            checkpoint_dir=b["save_folder"],
            work_dir=work_dir,
        )
    finally:
        cookbook.dir_is_empty = _saved_dir_is_empty
    # Ephemeral checkpoints for PREEMPTION-RESUME (ai2/jupiter preempts high-prio jobs): the trainer
    # resumes from the latest checkpoint in save_folder before falling back to load_path (the base),
    # so a frequent ephemeral checkpoint (only the most recent is kept -> ~one DCP on weka, no
    # permanent HF-curve intermediates) means a preempted job restarts near where it left off instead
    # of from the base. 0 disables. Distinct from save_freq (permanent, HF-curve) which stays 0.
    _ephemeral = int(float(b.get("ephemeral_save_freq", 200)))
    trainer_config = trainer_config.with_callbacks(
        cookbook.configure_default_callbacks(
            run_name=run_ts, wandb_group_name=cli_context.run_name,
            wandb_run_id=_wandb_run_id(b["save_folder"]),
            **({"checkpoint_save_interval": save_freq} if save_freq > 0 else {}),
            **({"ephemeral_checkpoint_save_interval": _ephemeral} if _ephemeral > 0 else {}),
        )
    )

    launch_config = build_launch_config(
        name=cli_context.run_name,
        # Re-append the stripped build-time flags so the remote `train` hop re-parses them.
        cmd=[*cli_context.remote_cmd, *_build_argv],
        cluster=cli_context.cluster,
        root_dir=root_dir,
        workspace="ai2/oe-science",
        num_nodes=int(b.get("num_nodes", 1)),
        nccl_debug=False,
        beaker_image=OLMoCoreBeakerImage.stable,  # override via --launch.beaker_image=
    )
    # OLMoDDP models train on the olmo-ddp preset's image, env and symm-mem extension prebuild. KDA
    # layers with experimental kernels also need kernel-fun and FLA, which the recipe image may lack.
    # The preset image ships wandb 0.30, which drops Run.get_url and finish(quiet=...) that the
    # beaker and wandb callbacks call.
    if isinstance(model_config, OLMoDDPModelConfig):
        _preset = get_preset("olmo-ddp")
        launch_config.beaker_image = _preset.beaker_image
        launch_config.env_vars.extend(BeakerEnvVar(name=k, value=v) for k, v in _preset.env_vars)
        launch_config.post_setup = " && ".join([
            "pip install 'kernel-fun==0.2.0' 'flash-linear-attention==0.5.2' 'wandb<0.30'",
            _preset.post_setup,
        ])

    # Mount every weka bucket this run touches. build_launch_config adds oe-training-default when
    # the root dir is on weka and nothing else, so a base in another bucket (the Olmo 3.5
    # checkpoints live in olmo-3p5-checkpoints) is simply absent in the job and the trainer fails on
    # a path that exists. Derived from the paths rather than configured, so a profile that moves the
    # base or the save folder needs no second edit here.
    _mounted = {bucket.bucket for bucket in launch_config.weka_buckets}
    for _path in (b["base_checkpoint"], b["save_folder"]):
        _parts = str(_path).split("/")
        if len(_parts) > 2 and _parts[1] == "weka" and _parts[2] not in _mounted:
            launch_config.weka_buckets.append(BeakerWekaBucket(_parts[2], f"/weka/{_parts[2]}"))
            _mounted.add(_parts[2])

    # build_launch_config attaches optional secrets (COMET_API_KEY, R2_ENDPOINT_URL,
    # WEKA_ENDPOINT_URL, SLACK_WEBHOOK_URL) that most workspaces do not define.
    # BeakerLaunchConfig._get_env_secrets skips its existence check whenever the launcher itself
    # runs inside a Beaker batch job, which is exactly how sftlab drives this, so an undefined
    # optional secret reaches the experiment spec and Beaker rejects the whole submit with
    # `[code=404] no secret found with name ...`. Resolve them here instead: required secrets pass
    # through untouched, optional ones only if the workspace actually has them.
    launch_config.env_secrets = [
        s for s in launch_config.env_secrets
        if s.required or launch_config._secret_exists(s)
    ]

    config = ExperimentConfig(
        run_name=cli_context.run_name,
        launch=launch_config,
        model=model_config,
        train_module=train_module_config,
        trainer=trainer_config,
        dataset=dataset_config,
        data_loader=data_loader_config,
        init_seed=seed,
    )
    return config.merge(cli_context.overrides)  # --launch.*/--trainer.*/… dotted overrides


if __name__ == "__main__":
    main(config_builder=build_experiment_config)
