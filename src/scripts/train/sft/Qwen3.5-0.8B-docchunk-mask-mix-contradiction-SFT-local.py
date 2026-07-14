"""LOCAL (torchrun, no Beaker/weka) docchunk + **mask-mixing** SFT vehicle for the hybrid
Qwen3.5-0.8B (Gated DeltaNet linear-attention + full-attention, 3:1). The Qwen3.5-0.8B analogue of
``Qwen3-0.6B-docchunk-mask-mix-contradiction-SFT-local.py``.

Key differences from the Qwen3-0.6B script (see ``qwen3_5_like`` in
``olmo_core.nn.transformer.config``):

  * model: ``qwen3_5_0_8B(document_chunked=True, cross_doc_mode=...)`` -- the document-chunked mask
    is applied ONLY to the 6 full-attention blocks (layers 3, 7, 11, 15, 19, 23). The 18 Gated
    DeltaNet blocks are linear attention (inherently unmasked) and simply ignore the runtime
    ``chunk_ids`` threaded to every block.
  * tokenizer: Qwen3.5 has its OWN, larger tokenizer (vocab 248320, eos ``<|endoftext|>``=248044,
    box markers ``<|box_start|>``=248049 / ``<|box_end|>``=248050) -- NOT the Qwen3 tokenizer
    (vocab 151936, eos 151643, box 151648/9). The docdense shard MUST be tokenized with the Qwen3.5
    tokenizer (see ``contradiction_n20_docdense_nocot_qwen35``).
  * base: a Qwen3.5-0.8B-Base olmo distcp converted from HF via
    ``olmo_core.nn.hf.convert.convert_qwen3_5_state_from_hf`` (model-only).

Mask-mixing (``--mix-mode {none,static,curriculum}``) and the cross-doc-mode wiring
(``--cross-doc-mode {chunked,hierarchical_dilated,random_doc}``) are identical to the 0.6B script.

Run with ``--no-compile`` (the python-seeded mix coin + counter are not torch.compile-capturable)::

    PYTHONPATH=<repo>/src torchrun --nproc_per_node=1 \\
      src/scripts/train/sft/Qwen3.5-0.8B-docchunk-mask-mix-contradiction-SFT-local.py \\
      --data-dir /scratch/users/prasann/longctx_sft_qwen/contradiction_n20_docdense_nocot_qwen35 \\
      --mix-mode curriculum --mix-start-p 0.8 --mix-end-p 0.0 --no-compile --run-name smoke
"""

import argparse
import json

from olmo_core.config import DType
from olmo_core.data import TokenizerConfig
from olmo_core.data.composable import (
    ComposableDataLoaderConfig,
    PadToLengthInstanceSourceConfig,
)
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.distributed.utils import get_rank, get_world_size
from olmo_core.float8 import Float8Config
from olmo_core.nn.attention import AttentionBackendName
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.optim import LinearWithWarmup, OptimGroupOverride, SkipStepAdamWConfig
from olmo_core.train import (
    Duration,
    LoadStrategy,
    TrainerConfig,
    prepare_training_environment,
    teardown_training_environment,
)
from olmo_core.train.callbacks import (
    ConfigSaverCallback,
    GPUMemoryMonitorCallback,
)
from olmo_core.train.train_module import (
    TransformerDataParallelConfig,
    TransformerDataParallelWrappingStrategy,
    TransformerTrainModuleConfig,
)
from olmo_core.utils import seed_all

# ---- reserved ids (Qwen3.5 tokenizer; the docdense shard is tokenized with these) ----
DOC_START_ID = 248049  # <|box_start|>
DOC_END_ID = 248050  # <|box_end|>
EOS_TOKEN_ID = 248044  # <|endoftext|> (pad == eos)
VOCAB_SIZE = 248320  # Qwen3.5-0.8B-Base embedding size (matches the converted base ckpt)

# ---- LOCAL paths (shared NFS / readable from any Berkeley GPU node) ----
# Base: Qwen3.5-0.8B-Base converted to olmo distcp (model-only). Point at the model_and_optim SUBDIR
# (parent has no .metadata marker). Shared on /scratch so every attention-pattern variant inits from
# the SAME weights for a fair comparison of the attention MASK only.
BASE_CHECKPOINT = "/scratch/users/prasann/cpt_mix_ckpts/q35-08b-base-modelonly/model_and_optim"
SAVE_ROOT = "/scratch/users/prasann/olmo_ckpts"
WORK_DIR = "/scratch/users/prasann/longctx_sft_qwen/dataset-cache-docchunk-mix-q35"

LR = 5e-5
NUM_EPOCHS = 3


def build_and_fit(opts: argparse.Namespace) -> None:
    run_name = opts.run_name
    save_folder = opts.save_folder or f"{SAVE_ROOT}/{run_name}"
    base_checkpoint = opts.base_checkpoint or BASE_CHECKPOINT

    meta = json.load(open(f"{opts.data_dir}/metadata.json"))
    n_examples = int(meta["num_instances"])
    seq_len = opts.seq_len
    if meta.get("max_example_len", 0) > seq_len:
        raise SystemExit(
            f"--seq-len={seq_len} < max example length {meta['max_example_len']} "
            f"(PadToLength would SKIP the long examples). Raise --seq-len."
        )
    # micro-batch = 1 instance (rank_microbatch = seq_len); global batch = grad_accum instances.
    # The curriculum's p_standard is driven by a PER-RANK forward counter, and data is sharded across
    # the DP ranks (1 instance/GPU/forward), so rank 0 does only n_examples*epochs/world_size forwards.
    # Divide by world_size so p_standard actually anneals to mix_end_p by end-of-training (else it stalls
    # at mix_start_p*(1-1/world_size) -- worse the more GPUs; see the mask-mix-ngpu-anneal bug).
    global_batch_size = opts.grad_accum * seq_len
    world_size = max(1, get_world_size())
    mix_total_forwards = max(1, (n_examples * opts.epochs) // world_size)

    # ---- mask-mix config keys threaded into document_chunk_attention ----
    mix_keys: dict = {}
    if opts.mix_mode == "static":
        mix_keys = dict(
            standard_mix_prob=opts.standard_mix_prob,
            mix_seed=opts.mix_seed,
            mix_log_interval=opts.mix_log_interval,
        )
    elif opts.mix_mode == "curriculum":
        mix_keys = dict(
            mix_start_p=opts.mix_start_p,
            mix_end_p=opts.mix_end_p,
            mix_total_forwards=mix_total_forwards,
            mix_seed=opts.mix_seed,
            mix_log_interval=opts.mix_log_interval,
        )
    print(
        f"[q35-docchunk-mix] mode={opts.mix_mode} n_examples={n_examples} epochs={opts.epochs} "
        f"seq_len={seq_len} mix_total_forwards={mix_total_forwards} mix_keys={mix_keys}",
        flush=True,
    )

    # Qwen3.5-0.8B-Base tokenizer (its OWN vocab; NOT TokenizerConfig.qwen3()).
    tokenizer_config = TokenizerConfig(
        vocab_size=VOCAB_SIZE,
        eos_token_id=EOS_TOKEN_ID,
        pad_token_id=EOS_TOKEN_ID,
        bos_token_id=None,
        identifier="Qwen/Qwen3.5-0.8B-Base",
    )
    doc_tokenizer_config = tokenizer_config

    # ---- cross-document visibility policy (attention-pattern sweep) -- applied to the 6 full-attn
    # blocks only (the GDN blocks are linear attention and stay unmasked).
    #   * "chunked"              -> plain document-chunked (each context doc attends itself only; the
    #                               free instruction/query/answer tokens attend the full context).
    #   * "hierarchical_dilated" -> a strict superset of "chunked": each context query additionally
    #                               attends dilation_n docs (incl. self) at a per-layer stride of
    #                               dilation_m**layer (whole-document granularity). NOT the positional
    #                               dilated_sliding_window.
    #   * "random_doc"           -> each context doc attends itself + a seeded-random doc_keep_prob
    #                               subset of the STRICTLY-EARLIER docs (BigBird-style).
    qwen_kwargs: dict = dict(
        vocab_size=VOCAB_SIZE,
        document_chunked=True,
        cross_doc_mode=opts.cross_doc_mode,
        # Pin flash_2 so the saved config.json supports KV-cached generation at eval (matches the
        # 0.6B script; the qwen3_5 factory defaults backend=None otherwise).
        attn_backend=AttentionBackendName.flash_2,
    )
    if opts.cross_doc_mode == "hierarchical_dilated":
        qwen_kwargs["dilation_n"] = opts.dilation_n
        qwen_kwargs["dilation_m"] = opts.dilation_m
        if opts.dilation_cycle is not None:
            qwen_kwargs["dilation_cycle"] = opts.dilation_cycle
    elif opts.cross_doc_mode == "random_doc":
        qwen_kwargs["doc_keep_prob"] = opts.doc_keep_prob
        qwen_kwargs["random_doc_seed"] = opts.random_doc_seed
    # Hybrid full/chunked: these layer indices run plain full (causal) attention while the rest use
    # the chunked mask. (Indices are absolute; only layers 3/7/11/15/19/23 are attention layers.)
    full_attn_layers = None
    if opts.full_attention_layers:
        full_attn_layers = [
            int(x) for x in opts.full_attention_layers.split(",") if x.strip() != ""
        ]
        qwen_kwargs["full_attention_layers"] = full_attn_layers
    print(
        f"[q35-docchunk-mix] cross_doc_mode={opts.cross_doc_mode} "
        f"full_attention_layers={full_attn_layers} "
        f"dilation_n={opts.dilation_n if opts.cross_doc_mode=='hierarchical_dilated' else None} "
        f"dilation_m={opts.dilation_m if opts.cross_doc_mode=='hierarchical_dilated' else None} "
        f"dilation_cycle={opts.dilation_cycle if opts.cross_doc_mode=='hierarchical_dilated' else None} "
        f"doc_keep_prob={opts.doc_keep_prob if opts.cross_doc_mode=='random_doc' else None}",
        flush=True,
    )
    model_config = TransformerConfig.qwen3_5_0_8B(**qwen_kwargs)
    model_config.document_chunk_attention = {
        "doc_start_id": DOC_START_ID,
        "doc_end_id": DOC_END_ID,
        "eos_id": EOS_TOKEN_ID,
        "mode": "chunked",
        **mix_keys,
    }

    train_module_config = TransformerTrainModuleConfig(
        rank_microbatch_size=seq_len,
        max_sequence_length=seq_len,
        optim=SkipStepAdamWConfig(
            lr=opts.lr,
            weight_decay=0.0,
            betas=(0.9, 0.95),
            group_overrides=[
                OptimGroupOverride(params=["embeddings.weight"], opts=dict(weight_decay=0.0))
            ],
        ),
        scheduler=LinearWithWarmup(warmup_fraction=0.03, alpha_f=0.0),
        compile_model=opts.compile,
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.fsdp,
            param_dtype=DType.bfloat16,
            reduce_dtype=DType.float32,
            wrapping_strategy=TransformerDataParallelWrappingStrategy.full,
            shard_degree=1,
        ),
        float8_config=Float8Config(enabled=False),
        z_loss_multiplier=None,
        max_grad_norm=1.0,
    )

    instance_source_config = PadToLengthInstanceSourceConfig.from_npy(
        f"{opts.data_dir}/token_ids_part_*.npy",
        tokenizer=doc_tokenizer_config,
        sequence_length=seq_len,
        label_mask_paths=[f"{opts.data_dir}/labels_mask_*.npy"],
        expand_glob=True,
    )

    data_loader_config = ComposableDataLoaderConfig(
        tokenizer=tokenizer_config,
        work_dir=opts.work_dir or WORK_DIR,
        global_batch_size=global_batch_size,
        seed=34521,
        num_workers=2,
    )

    trainer_config = (
        TrainerConfig(
            save_folder=save_folder,
            save_overwrite=True,
            load_path=base_checkpoint,
            load_strategy=LoadStrategy.always,
            load_trainer_state=False,
            load_optim_state=False,
            metrics_collect_interval=1,
            cancel_check_interval=1,
            max_duration=Duration.epochs(opts.epochs),
            hard_stop=Duration.steps(opts.max_steps) if opts.max_steps else None,
            no_checkpoints=False,
        )
        .with_callback("gpu_monitor", GPUMemoryMonitorCallback())
        .with_callback("config_saver", ConfigSaverCallback())
    )

    seed_all(12536)
    model = model_config.build(init_device="meta")
    train_module = train_module_config.build(model)
    source = instance_source_config.build(data_loader_config.work_dir)
    data_loader = data_loader_config.build(source, dp_process_group=train_module.dp_process_group)
    trainer = trainer_config.build(train_module, data_loader)
    trainer.fit()

    # Save a model-only checkpoint (config.json + model_and_optim/) so the docchunk eval loads it.
    if opts.save_checkpoint:
        from olmo_core.distributed.checkpoint import save_model_and_optim_state

        model_dir = f"{save_folder}/model_and_optim"
        save_model_and_optim_state(model_dir, train_module.model, save_overwrite=True)
        if get_rank() == 0:
            experiment = {
                "model": model_config.as_config_dict(),
                "dataset": {"tokenizer": tokenizer_config.as_config_dict()},
            }
            with open(f"{save_folder}/config.json", "w") as f:
                json.dump(experiment, f)
            print(f"[q35-docchunk-mix] saved model-only checkpoint -> {save_folder}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", required=True, help="qwen3.5-tokenized docdense shard dir")
    ap.add_argument("--run-name", default="q35-08b-docchunk-mix-smoke")
    ap.add_argument("--save-folder", default=None)
    ap.add_argument(
        "--base-checkpoint",
        default=None,
        help=f"model_and_optim distcp subdir (default {BASE_CHECKPOINT})",
    )
    ap.add_argument(
        "--work-dir",
        default=None,
        help=f"data-loader cache dir (default {WORK_DIR}); use node-local /data for speed",
    )
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument(
        "--grad-accum",
        type=int,
        default=8,
        help="gradient accumulation = instances per optimizer step (mbs=1)",
    )
    ap.add_argument(
        "--save-checkpoint",
        action="store_true",
        help="after fit, save model-only checkpoint (config.json + model_and_optim) for eval",
    )
    ap.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    ap.add_argument("--max-steps", type=int, default=0, help="stop after N steps (0 = full)")
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument(
        "--cross-doc-mode",
        choices=["chunked", "hierarchical_dilated", "random_doc"],
        default="chunked",
        help="document-chunked cross-document visibility policy (see build_and_fit)",
    )
    ap.add_argument(
        "--dilation-n",
        type=int,
        default=3,
        help="hierarchical_dilated: #docs a context query attends per layer (incl. self)",
    )
    ap.add_argument(
        "--dilation-m",
        type=int,
        default=2,
        help="hierarchical_dilated: dilation base; per-layer stride = dilation_m**layer",
    )
    ap.add_argument(
        "--dilation-cycle",
        type=int,
        default=None,
        help="hierarchical_dilated: rotation period L (rotating 'Hierarchical K').",
    )
    ap.add_argument(
        "--doc-keep-prob",
        type=float,
        default=0.5,
        help="random_doc: fraction of STRICTLY-EARLIER docs each context doc attends (0..1)",
    )
    ap.add_argument("--random-doc-seed", type=int, default=42, help="random_doc: RNG seed")
    ap.add_argument(
        "--full-attention-layers",
        type=str,
        default="",
        help="comma-separated absolute layer indices that use PLAIN FULL (causal) "
        "attention instead of the chunked mask. Empty = all attn layers chunked.",
    )
    ap.add_argument("--mix-mode", choices=["none", "static", "curriculum"], default="none")
    ap.add_argument("--standard-mix-prob", type=float, default=0.10)
    ap.add_argument("--mix-start-p", type=float, default=0.80)
    ap.add_argument("--mix-end-p", type=float, default=0.0)
    ap.add_argument("--mix-seed", type=int, default=42)
    ap.add_argument("--mix-log-interval", type=int, default=5)
    ap.add_argument("--no-compile", dest="compile", action="store_false")
    opts = ap.parse_args()

    prepare_training_environment()
    try:
        build_and_fit(opts)
    finally:
        teardown_training_environment()


if __name__ == "__main__":
    main()
