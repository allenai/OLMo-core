"""Micro-anneal recipe (ARCH-PARAMETERIZED) — added for sftlab's olmo_core_anneal backend.

Launched by sftlab (`python <this> launch <run_name> <cluster> …`); it must live IN this repo (on
the davidg/anneal branch) because the launch clones the pushed commit into the Beaker job. One
script serves any base (Qwen, OLMo, Llama, …) via --model_arch. Modeled on OLMo3-32B-midtraining.py:
anneal a FIXED base checkpoint for a small token budget on a source_mixtures data mix, model-only
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

The build-time flags (above the launch/dotted ones) must be known BEFORE the ExperimentConfig is
built (they pick the arch, mix, LR, load behaviour), so they cannot ride the post-hoc
`.merge(overrides)` path — this module strips them from argv and stashes them, then hands the rest
(<cmd> <run_name> <cluster> + any --launch.*/--trainer.* dotted overrides) to olmo-core's `main`.
They are re-appended to the remote `launch` command so the Beaker-side `train` hop sees them too.

NOTE: first run should be validated with the `dry_run` subcommand (renders the resolved config,
no Beaker submit) — this is internal-API code and may need an iteration to match olmo-core exactly.
"""

import re
import sys

# ── pre-parse sftlab build-time flags out of argv (before olmo_core.main sees it) ──────────────
_BUILD_KEYS = {
    "base_checkpoint", "model_arch", "tokenizer", "source_mixture_yaml", "length_tokens",
    "seq_len", "peak_lr", "lr_schedule", "warmup_steps", "load_optim_state", "load_trainer_state",
    "save_folder", "save_freq", "seed", "num_nodes",
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
from datetime import datetime  # noqa: E402

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
from olmo_core.launch.beaker import OLMoCoreBeakerImage  # noqa: E402
from olmo_core.nn.transformer import TransformerConfig  # noqa: E402
from olmo_core.optim.scheduler import LinearWithWarmup, SchedulerUnits, WSD  # noqa: E402
from olmo_core.train import Duration  # noqa: E402

GLOBAL_BATCH_SIZE = 4 * 1024 * 1024  # ~4M tokens; override via --global_batch_size / --data_loader.*


def _scheduler(warmup_steps: int, kind: str):
    """Decay-to-zero schedule. 'linear_with_warmup' -> LinearWithWarmup(alpha_f=0); 'wsd' -> WSD."""
    if kind == "wsd":
        return WSD(units=SchedulerUnits.steps, warmup=warmup_steps, warmup_fraction=None,
                   decay=None, decay_fraction=0.1)
    return LinearWithWarmup(units=SchedulerUnits.steps, warmup=warmup_steps, alpha_f=0.0)


def build_experiment_config(cli_context: CliContext) -> ExperimentConfig:
    b = _BUILD
    seq_len = int(b["seq_len"])
    seed = int(b.get("seed", 42))
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
    model_config = getattr(TransformerConfig, b["model_arch"])(
        vocab_size=tokenizer_config.padded_vocab_size(),
    )

    train_module_config = cookbook.configure_train_module(
        max_sequence_length=seq_len,
        rank_microbatch_size=seq_len,
        learning_rate=float(b["peak_lr"]),
        scheduler=_scheduler(int(b.get("warmup_steps", 0)), b.get("lr_schedule", "linear_with_warmup")),
    )

    # The varied ingredient: a source_mixtures YAML (sources + target_ratio summing to 1.0).
    source_list = SourceMixtureList.from_yaml(b["source_mixture_yaml"])
    source_list.validate()
    dataset_config = NumpyFSLDatasetConfig.from_src_mix(
        src_mix=SourceMixtureDatasetConfig(
            source_list=source_list,
            requested_tokens=int(float(b["length_tokens"])),
            global_batch_size=GLOBAL_BATCH_SIZE,
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
    data_loader_config = NumpyDataLoaderConfig(global_batch_size=GLOBAL_BATCH_SIZE, seed=seed, num_workers=4)

    save_freq = int(b.get("save_freq", 0))
    # sftlab is a pure Beaker-job INITIATOR: `launch`/`dry_run` run on a host with NO /weka mount,
    # so configure_trainer's dir_is_empty(base_checkpoint) preflight would false-positive — a
    # missing mount is indistinguishable from an empty dir (io.dir_is_empty returns True when the
    # dir doesn't exist) — and abort the launch before any job is submitted. Skip that preflight on
    # the initiator hop only; the remote `train` hop rebuilds this identical config WITH /weka
    # mounted and validates for real (and the trainer's own load_checkpoint fails loudly if the
    # base is genuinely absent). Keeps a launch touching only the Beaker API, never weka.
    _initiator = cli_context.cmd in (SubCmd.launch, SubCmd.dry_run, SubCmd.launch_prep)
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
    trainer_config = trainer_config.with_callbacks(
        cookbook.configure_default_callbacks(
            run_name=run_ts, wandb_group_name=cli_context.run_name,
            **({"checkpoint_save_interval": save_freq} if save_freq > 0 else {}),
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
