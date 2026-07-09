"""LOCAL (torchrun, no Beaker/weka) docchunk + **mask-mixing** smoke vehicle.

Adapts ``Qwen3-0.6B-fast-landmark-contradiction-n20-SFT-local.py`` to the *document-chunked* path to
exercise the native mask-mixing schedule (see
:meth:`olmo_core.nn.transformer.Transformer.enable_document_chunk_attention`):

  * model: ``qwen3_0_6B(document_chunked=True, cross_doc_mode="chunked")`` (dense chunked mask);
  * data: a docdense contradiction shard (``<|box_start|>`` / ``<|box_end|>`` markers present),
    one example per instance right-padded to ``--seq-len`` (``PadToLengthInstanceSource``);
  * mixing: ``--mix-mode {none,static,curriculum}`` wires the five mix keys into the
    ``document_chunk_attention`` config. ``mix_total_forwards`` is auto-computed as
    ``n_examples * epochs`` (global batch == seq len -> 1 instance/step -> forwards == steps).

Run with ``--no-compile`` (the python-seeded mix coin + counter are not torch.compile-capturable)::

    PYTHONPATH=<repo>/src torchrun --nproc_per_node=1 \\
      src/scripts/train/sft/Qwen3-0.6B-docchunk-mask-mix-contradiction-SFT-local.py \\
      --data-dir /scratch/users/prasann/longctx_sft_qwen/contradiction_n20_docdense \\
      --mix-mode curriculum --mix-start-p 0.8 --mix-end-p 0.0 --mix-log-interval 5 \\
      --no-compile --run-name smoke-curriculum
"""

import argparse
import json
from dataclasses import replace

from olmo_core.config import DType
from olmo_core.data import TokenizerConfig
from olmo_core.data.composable import (
    ComposableDataLoaderConfig,
    PadToLengthInstanceSourceConfig,
)
from olmo_core.distributed.parallel import DataParallelType
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
from olmo_core.distributed.utils import get_rank
from olmo_core.utils import seed_all

# ---- reserved ids (match the docdense converter / document_chunk_landmark defaults) ----
DOC_START_ID = 151648
DOC_END_ID = 151649
EOS_TOKEN_ID = 151643  # Qwen3-Base <|endoftext|> (pad == eos; trailing pad -> PAD by the "after first
#                        eos" rule, so no dedicated pad_id is needed for the dense docchunk path).

# ---- LOCAL paths (shared NFS / readable from any Berkeley GPU node) ----
# Base: an existing qwen3_0_6B dense olmo-core distcp (d_model 1024 / n_layers 28 / vocab 151936 ==
# TransformerConfig.qwen3_0_6B). Point at the model_and_optim SUBDIR (parent has no .metadata marker).
# Shared dense CPT base (on shared /scratch, so all attention-pattern variants -- full / chunked /
# hierarchical -- init from the SAME weights for a fair comparison). Point at the model_and_optim subdir.
BASE_CHECKPOINT = "/scratch/users/prasann/cpt_mix_ckpts/q06b-dense-cpt-modelonly/model_and_optim"
SAVE_ROOT = "/scratch/users/prasann/olmo_ckpts"
WORK_DIR = "/scratch/users/prasann/longctx_sft_qwen/dataset-cache-docchunk-mix"

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
    # #training forwards == n_examples * epochs (independent of grad_accum), which is mix_total_forwards.
    global_batch_size = opts.grad_accum * seq_len
    mix_total_forwards = n_examples * opts.epochs

    # ---- mask-mix config keys threaded into document_chunk_attention ----
    mix_keys: dict = {}
    if opts.mix_mode == "static":
        mix_keys = dict(standard_mix_prob=opts.standard_mix_prob, mix_seed=opts.mix_seed,
                        mix_log_interval=opts.mix_log_interval)
    elif opts.mix_mode == "curriculum":
        mix_keys = dict(mix_start_p=opts.mix_start_p, mix_end_p=opts.mix_end_p,
                        mix_total_forwards=mix_total_forwards, mix_seed=opts.mix_seed,
                        mix_log_interval=opts.mix_log_interval)
    print(f"[docchunk-mix] mode={opts.mix_mode} n_examples={n_examples} epochs={opts.epochs} "
          f"seq_len={seq_len} mix_total_forwards={mix_total_forwards} mix_keys={mix_keys}",
          flush=True)

    tokenizer_config = TokenizerConfig.qwen3()
    doc_tokenizer_config = replace(tokenizer_config, bos_token_id=None)

    # ---- cross-document visibility policy (attention-pattern sweep) ----
    #   * "chunked"            -> plain document-chunked (each context doc attends itself only; the
    #                             free instruction/query/answer tokens attend the full context).
    #   * "hierarchical_dilated" -> a strict superset of "chunked": each context query additionally
    #                             attends dilation_n docs (incl. self) at a per-layer stride of
    #                             dilation_m**layer (chunk / whole-document granularity). The FREE
    #                             query + answer tokens still attend the full context. This is the
    #                             intended "hierarchical" variant (NOT the positional
    #                             dilated_sliding_window, which is token-granularity + no free/beginning).
    qwen_kwargs: dict = dict(
        vocab_size=tokenizer_config.padded_vocab_size(),
        document_chunked=True,
        cross_doc_mode=opts.cross_doc_mode,
        # qwen3_0_6B does NOT inject an attn_backend (unlike olmo2/qwen3_5 factories), so
        # llama_like defaults backend=None -> TorchAttentionBackend, which can't KV-cache during
        # eval decode. Pin flash_2 so the saved config.json supports KV-cached generation.
        attn_backend=AttentionBackendName.flash_2,
    )
    if opts.cross_doc_mode == "hierarchical_dilated":
        # dilation_n / dilation_m are ONLY valid with hierarchical_dilated (config.py asserts this).
        qwen_kwargs["dilation_n"] = opts.dilation_n
        qwen_kwargs["dilation_m"] = opts.dilation_m
    elif opts.cross_doc_mode == "random_doc":
        # random_doc: each context doc attends itself + a seeded-random ``doc_keep_prob`` subset of the
        # STRICTLY EARLIER docs (BigBird-style; RANDOM connectivity, not strided). FREE query/answer
        # tokens still attend the full context. keep_prob=1.0 == full cross-doc (approaches full attn).
        qwen_kwargs["doc_keep_prob"] = opts.doc_keep_prob
        qwen_kwargs["random_doc_seed"] = opts.random_doc_seed
    print(f"[docchunk-mix] cross_doc_mode={opts.cross_doc_mode} "
          f"dilation_n={opts.dilation_n if opts.cross_doc_mode=='hierarchical_dilated' else None} "
          f"dilation_m={opts.dilation_m if opts.cross_doc_mode=='hierarchical_dilated' else None} "
          f"doc_keep_prob={opts.doc_keep_prob if opts.cross_doc_mode=='random_doc' else None}",
          flush=True)
    model_config = TransformerConfig.qwen3_0_6B(**qwen_kwargs)
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
            # no_checkpoints=True would ALSO skip the base-checkpoint LOAD block (trainer.fit gates
            # loading on `not no_checkpoints`) -> silent train-from-scratch (CE ~ln(vocab) ~= 12).
            # Keep it False so load_path (the dense CPT base) is loaded; there is no CheckpointerCallback
            # here so nothing is auto-saved during training -- the explicit --save-checkpoint below
            # writes the single model-only checkpoint for eval.
            no_checkpoints=False,
        )
        .with_callback("gpu_monitor", GPUMemoryMonitorCallback())
        .with_callback("config_saver", ConfigSaverCallback())
    )

    seed_all(12536)
    model = model_config.build(init_device="meta")
    train_module = train_module_config.build(model)
    source = instance_source_config.build(data_loader_config.work_dir)
    data_loader = data_loader_config.build(
        source, dp_process_group=train_module.dp_process_group
    )
    trainer = trainer_config.build(train_module, data_loader)
    trainer.fit()

    # Save a model-only checkpoint in the eval loader's expected layout
    # (config.json + model_and_optim/, matching scripts/train/convert_hf_to_olmo.py) so
    # eval_lc_native_docchunk_contra.py --model-path <save_folder> loads it directly.
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
            print(f"[docchunk-mix] saved model-only checkpoint -> {save_folder}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", required=True, help="docdense shard dir (token_ids_part_*.npy + metadata.json)")
    ap.add_argument("--run-name", default="q06b-docchunk-mix-smoke")
    ap.add_argument("--save-folder", default=None)
    ap.add_argument("--base-checkpoint", default=None,
                    help=f"model_and_optim distcp subdir (default {BASE_CHECKPOINT})")
    ap.add_argument("--work-dir", default=None,
                    help=f"data-loader cache dir (default {WORK_DIR}); use node-local /data for speed")
    ap.add_argument("--seq-len", type=int, default=8192)
    ap.add_argument("--grad-accum", type=int, default=8,
                    help="gradient accumulation = instances per optimizer step (mbs=1)")
    ap.add_argument("--save-checkpoint", action="store_true",
                    help="after fit, save model-only checkpoint (config.json + model_and_optim) for eval")
    ap.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    ap.add_argument("--max-steps", type=int, default=0, help="stop after N steps (0 = full)")
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--cross-doc-mode", choices=["chunked", "hierarchical_dilated", "random_doc"],
                    default="chunked",
                    help="document-chunked cross-document visibility policy (see build_and_fit)")
    ap.add_argument("--dilation-n", type=int, default=3,
                    help="hierarchical_dilated: #docs a context query attends per layer (incl. self)")
    ap.add_argument("--dilation-m", type=int, default=2,
                    help="hierarchical_dilated: dilation base; per-layer stride = dilation_m**layer")
    ap.add_argument("--doc-keep-prob", type=float, default=0.5,
                    help="random_doc: fraction of STRICTLY-EARLIER docs each context doc attends (0..1)")
    ap.add_argument("--random-doc-seed", type=int, default=42, help="random_doc: RNG seed")
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
