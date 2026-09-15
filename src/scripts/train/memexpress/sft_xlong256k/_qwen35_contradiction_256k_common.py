"""Compressive vs dense, contradiction-only SFT from their final 256K CPT checkpoints.

Readout: held-out in-domain contradiction F1 by context rung, 2K through 256K,
using query-position=both, matching the training renderer.

Data-controlled: three loader epochs of the same contradiction documents, seed 34521,
BFD packing, LR 4e-5 and 3% warmup. Compressive spends extra compute on landmark slots
and block padding. The CPT runs matched model tokens (10,003,415,040), not content:
compressive saw 63/64 as much CPT content. This biases its starting point against it.
Loader epochs discard fewer than four shuffled packed windows at each epoch boundary;
prep records the exact remainder for each arm. No task sampling or Dolci is used.
"""

from dataclasses import replace

from _qwen35_xlong5_dolci25_256k_common import build_qwen35_xlong5_experiment

from olmo_core.data.composable import (
    LandmarkPackingInstanceSourceConfig,
    LandmarkPackingStrategy,
    LongDocStrategy,
    NumpyDocumentSourceConfig,
    PackingInstanceSourceConfig,
)
from olmo_core.internal.experiment import CliContext, ExperimentConfig, SubCmd
from olmo_core.nn.attention import AttentionBackendName, AttentionType
from olmo_core.train import Duration
from olmo_core.train.callbacks import CheckpointerCallback

DENSE_SEQUENCE_LENGTH = 262144
DP_DEGREE = 4  # 2 nodes * 8 GPUs / Ulysses CP=4
DATA_ROOT = (
    "/weka/oe-training-default/ai2-llm/checkpoints/prasanns/"
    "xlong5_2k256k_qwen35/shards_full/contradiction_train"
)
_ARMS = {
    "dense": dict(
        cpt_run="fq3brt27",
        cpt_name="q35-4b-dense-256k-fix",
        mem_freq=None,
        num_landmarks=0,
        mem_id=None,
        packed_windows=None,
    ),
    "compressive": dict(
        cpt_run="2brjoa8r",
        cpt_name="q35-4b-fastcomplm-256k-fix",
        mem_freq=63,
        num_landmarks=1,
        mem_id=248200,
        packed_windows=None,
    ),
}


def build_contradiction_experiment(cli_context: CliContext, *, arm: str) -> ExperimentConfig:
    """Build one arm; keep the established two-node 256K optimization settings."""
    spec = _ARMS[arm]
    config = build_qwen35_xlong5_experiment(replace(cli_context, overrides=[]), arm="qboth")
    tokenizer = config.data_loader.tokenizer  # Qwen3.5, bos=None for EOS-separated documents
    source = NumpyDocumentSourceConfig(
        source_paths=[f"{DATA_ROOT}/token_ids_part_*.npy"],
        label_mask_paths=[f"{DATA_ROOT}/labels_mask_*.npy"],
        tokenizer=tokenizer,
        expand_glob=True,
    )
    sequence_length = DENSE_SEQUENCE_LENGTH
    if spec["mem_freq"] is not None:
        mem_freq, num_landmarks = spec["mem_freq"], spec["num_landmarks"]
        block_size = mem_freq + num_landmarks
        n_blocks = -(-DENSE_SEQUENCE_LENGTH // mem_freq)
        sequence_length = n_blocks * block_size  # 266368, content capacity 262206
        assert sequence_length // block_size * mem_freq >= DENSE_SEQUENCE_LENGTH
        mixer = config.model.block["attn"].sequence_mixer
        mixer.name = AttentionType.fast_compressive_landmark
        mixer.backend = AttentionBackendName.flash_2
        mixer.mem_freq = mem_freq
        mixer.num_landmarks = num_landmarks
        mixer.gate_temperature = None  # disabled in the CPT checkpoint
        config.dataset = [
            LandmarkPackingInstanceSourceConfig(
                source=source,
                sequence_length=sequence_length,
                mem_freq=mem_freq,
                num_landmarks=num_landmarks,
                mem_id=spec["mem_id"],
                pad_id=tokenizer.pad_token_id,
                packing_strategy=LandmarkPackingStrategy.best_fit_decreasing,
                label="contradiction",
            )
        ]
        config.data_loader.generate_doc_lengths = False  # packer supplies block-aligned lengths
    else:
        config.dataset = [
            PackingInstanceSourceConfig(
                sources=[source],
                sequence_length=sequence_length,
                tokenizer=tokenizer,
                long_doc_strategy=LongDocStrategy.exclude,
                source_group_size=1000000,  # pack all shards together, as on the landmark arm
                label="contradiction",
            )
        ]
    assert sequence_length % config.train_module.cp_config.degree == 0
    config.train_module.rank_microbatch_size = sequence_length
    config.train_module.max_sequence_length = sequence_length
    config.data_loader.global_batch_size = DP_DEGREE * sequence_length
    config.trainer.load_path = (
        f"/weka/oe-training-default/ai2-llm/checkpoints/{spec['cpt_name']}/"
        "step2385/model_and_optim"
    )
    config.trainer.load_optim_state = False
    config.trainer.load_trainer_state = False
    config.trainer.save_overwrite = False
    config.trainer.max_duration = Duration.epochs(3)
    config.trainer.callbacks["checkpointer"] = CheckpointerCallback(
        save_interval=250,
        ephemeral_save_interval=100,
        max_checkpoints=3,
        save_async=True,
    )  # post_train saves a permanent final checkpoint as well
    if cli_context.cmd in (SubCmd.train, SubCmd.train_single, SubCmd.launch):
        if spec["packed_windows"] is None:
            raise ValueError("Run CPU prep and record packed_windows before launching training")
    return config.merge(cli_context.overrides)
