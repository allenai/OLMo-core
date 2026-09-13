"""Two-stage Qwen3.5-4B CPT, sharing the existing eight-node 256K baseline."""

import importlib.util
from dataclasses import replace
from pathlib import Path

from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.internal.experiment import CliContext, ExperimentConfig
from olmo_core.nn.attention import AttentionType
from olmo_core.train import Duration

# Despite its filename, this baseline uses 262144 model tokens per sequence.
_BASELINE = Path(__file__).with_name("Qwen3.5-4B-fast-compressive-landmark-longmino512k.py")
_spec = importlib.util.spec_from_file_location("_qwen35_staged_cpt_baseline", _BASELINE)
assert _spec is not None and _spec.loader is not None
_baseline = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_baseline)


def build_staged_experiment(cli_context: CliContext, *, compressive: bool) -> ExperimentConfig:
    """Apply stage defaults before CLI overrides; stage two requires an explicit checkpoint."""
    # Keep the real CLI context (including remote_cmd) for launch construction, but apply its
    # overrides only after stage defaults. The baseline builder otherwise merges them too early.
    config = _baseline.build_experiment_config(replace(cli_context, overrides=[]))
    if config.launch is not None:
        config.launch.cmd = cli_context.remote_cmd
        config.launch.num_nodes = 8

    mixer = config.model.block["attn"].sequence_mixer
    mixer.name = (
        AttentionType.fast_compressive_landmark if compressive else AttentionType.sparse_landmark
    )
    mixer.layer_types = None
    mixer.mem_freq = 63
    mixer.num_landmarks = 1
    # None selects the default (disabled) and is accepted by both attention types.
    mixer.gate_temperature = None

    config.trainer.max_duration = Duration.tokens(10_000_000_000)
    config.trainer.hard_stop = None if compressive else Duration.tokens(5_000_000_000)
    config.trainer.load_optim_state = compressive
    config.trainer.load_trainer_state = compressive
    if compressive:
        config.trainer.load_path = None

    config = config.merge(cli_context.overrides)
    if compressive:
        if not config.trainer.load_path:
            raise OLMoConfigurationError(
                "Stage two requires --trainer.load_path=<stage-one-save-folder>/step1193 "
                "(use the actual final checkpoint if the batch size or stopping point changed)."
            )
        if not config.trainer.load_optim_state or not config.trainer.load_trainer_state:
            raise OLMoConfigurationError("Stage two must restore optimizer and trainer state.")
    return config
