"""Single-file architecture + pretraining specification for compute partners.

Copy ONLY this file; Python 3.10+ and its standard library are sufficient::

    python standalone_configs.py                  # JSON for all four sizes
    python standalone_configs.py --model-size tiny
    python standalone_configs.py --model-size small --no-emo

This is a portable specification, NOT a training launcher or TPU implementation.
Tiny is our current 0.794B-active hero (previously called production Small).
Partner Small is the previously selected 3.781B-active production Large, NOT
3.2B. The user reconfirmed these geometries on 2026-09-14. Medium/Large remain
approximately 4x architectural proposals without training qualification.

Training values were checked against the live Tiny hero checkpoint config at
step 197500 on 2026-09-14. Shared optimizer/init/schedule-shape defaults are copied
to larger sizes as UNVALIDATED starting points. None in unresolved training fields
means TODO, never zero or an implicit default; disabled optional architecture
features use None for not applicable. Actual Tiny batch/LR values are references,
not assigned to the larger rungs. NVIDIA execution settings are NOT TPU requirements.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass

CORE_COMMIT = "92976e128937ea1660257de80692079c7e2162c0"
CORE_URL = f"https://github.com/allenai/OLMo-core/tree/{CORE_COMMIT}"
TINY_BATCH_TOKENS = 16_777_216
TINY_PEAK_LR = 1.1e-3
TINY_TARGET_TOKENS = 14_000_000_000_000
WARMUP_STEPS = 2_000


@dataclass(frozen=True)
class Geometry:
    """Dimensions and embedding-inclusive parameter counts for one model."""

    d_model: int
    n_layers: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    expert_hidden_size: int
    num_routed_experts: int
    latent_dim: int
    expected_total_params: int
    expected_active_params: int
    attention_period: int = 8

    @property
    def full_attention_layers(self) -> tuple[int, ...]:
        """Zero-based gated full-attention block indices."""
        return tuple(range(self.attention_period - 1, self.n_layers, self.attention_period))


FAMILY = {
    "tiny": Geometry(1024, 16, 8, 4, 128, 1024, 512, 512, 12_496_341_632, 794_233_472),
    "small": Geometry(1536, 40, 16, 8, 128, 1536, 512, 768, 72_237_847_936, 3_780_515_200),
    "medium": Geometry(2560, 64, 24, 12, 128, 2560, 512, 1280, 322_601_566_720, 15_421_227_520),
    "large": Geometry(4608, 80, 48, 24, 128, 4608, 512, 2304, 1_310_163_554_560, 62_133_719_296),
}

# Installation toy for the sibling benchmark, NOT a fifth partner rung.
GEOMETRIES = {
    "30m": Geometry(128, 5, 1, 1, 128, 192, 32, 64, 32_323_589, 29_964_293, 5),
    **FAMILY,
}


def architecture(name: str, emo: bool = True) -> dict:
    """Return an implementation-independent description of every model component."""
    g = FAMILY[name]
    return {
        "name": name,
        "status": (
            "trained hero architecture"
            if name == "tiny"
            else "previous Large candidate" if name == "small" else "untrained size proposal"
        ),
        "geometry": asdict(g),
        "parameter_count_convention": (
            "Full embedding/untied LM-head tables, all shared/non-expert parameters, "
            "and 16 selected routed experts per MoE block. Not FLOPs."
        ),
        "vocab_size_model": 100_352,
        "tie_word_embeddings": False,
        "embedding_scale": math.sqrt(g.d_model),
        "embedding_rmsnorm": True,
        "lm_head_rmsnorm": True,
        "block_norms": "input and output RMSNorm for attention and FFN (four per block)",
        "rmsnorm_epsilon": 1e-6,
        "rmsnorm_variance_and_gain_product_dtype": "float32 before casting back",
        "dropout_probability": 0.0,
        "attention": {
            "full_attention_layers_zero_based": list(g.full_attention_layers),
            "kda_layers": g.n_layers - len(g.full_attention_layers),
            "kda_to_full_attention_ratio": "7:1",
            "position_encoding": "none (NoPE)",
            "full_attention": {
                "causal": True,
                "query_heads": g.n_heads,
                "kv_heads": g.n_kv_heads,
                "head_dim": 128,
                "gqa_query_to_kv_ratio": 2,
                "qk_norm": "per-head RMSNorm with independent learned [heads,128] gains",
                "output_gate": "elementwise sigmoid of a separate query-width projection",
                "output_gate_full_precision": True,
                "scalable_softmax": True,
                "scalable_softmax_formula": (
                    "After QK norm, multiply Q by ln(visible causal-span tokens) times a learned "
                    "scalar per query head; also apply ordinary 1/sqrt(head_dim). The hero uses "
                    "sequence-relative positions; explicit document masking resets the span."
                ),
                "linear_bias": False,
            },
            "kda": {
                "query_key_heads": g.n_heads,
                "value_heads": g.n_heads,
                "key_head_dim": 128,
                "value_head_dim": 256,
                "expand_v": 2.0,
                "allow_negative_eigenvalues": True,
                "beta": "2 * sigmoid(beta_projection)",
                "query_key_l2_normalization": True,
                "query_scale_after_l2_normalization": "1/sqrt(key_head_dim)",
                "causal_conv_kernel_size": 4,
                "causal_conv_bias": False,
                "causal_conv_activation": "silu",
                "decay_gate_low_rank_dim": 256,
                "output_gate_low_rank_dim": 256,
                "output_gate_second_projection_bias": True,
                "output_gate": "sigmoid",
                "output_rmsnorm_epsilon": 1e-5,
                "all_other_projection_biases": False,
            },
        },
        "feed_forward": {
            "activation": "SwiGLU (linear up multiplied by silu(linear gate))",
            "bias": False,
            "first_block": {"type": "dense", "hidden_size": 8 * g.d_model},
            "remaining_blocks": {
                "type": "latent MoE plus uncompressed shared expert",
                "routed_experts": 512,
                "active_routed_experts_per_token": 16,
                "shared_experts": 1,
                "routed_and_shared_hidden_size": g.expert_hidden_size,
                "latent_dim": g.latent_dim,
                "latent_ratio": "d_model/2 exactly",
                "latent_down_and_up_projections": True,
                "latent_up_input_norm": False,
                "router_input_dim": g.d_model,
                "router": "bias-free linear, fp32 softmax, top-k",
                "selected_weight_normalization": "L1 normalize to sum 1, then multiply by top_k=16",
                "global_load_balancing": True,
                "load_balancing_loss_weight": 0.01,
                "router_z_loss_weight": 1e-5,
                "emo_enabled": emo,
                "emo_document_pool_min": 16 if emo else None,
                "emo_document_pool_max": 512 if emo else None,
                "emo_eval_document_pool": 512 if emo else None,
                "emo_note": "Both EMO-on and EMO-off Tiny heroes exist; no winner selected yet.",
            },
        },
    }


def training(name: str) -> dict:
    """Tiny's actual recipe, or explicitly unvalidated inheritance for larger rungs."""
    if name not in FAMILY:
        raise ValueError(name)
    tiny = name == "tiny"
    steps = (TINY_TARGET_TOKENS + TINY_BATCH_TOKENS - 1) // TINY_BATCH_TOKENS
    result = {
        "stage": "pretraining, not midtraining/long-context/SFT",
        "status": "observed Tiny hero recipe" if tiny else "Tiny-derived unvalidated template",
        "reference_tiny_values_not_assigned_to_larger_models": {
            "global_batch_size_tokens": TINY_BATCH_TOKENS,
            "peak_lr": TINY_PEAK_LR,
            "target_tokens": TINY_TARGET_TOKENS,
            "num_gpus": 64,
            "microbatch_sequences_per_gpu": 4,
        },
        "data": {
            "mixture": "Dolma3p5-14t",
            "tokenizer": "allenai/dolma2-tokenizer",
            "tokenizer_vocab_size": 100_278,
            "padded_model_vocab_size": 100_352,
            "eos_token_id": 100_257,
            "pad_token_id": 100_277,
            "sequence_length_tokens": 8192,
            "data_order_seed": 928_543_231,
            "dataset_type": "NumpyFSLDataset: fixed-length contiguous token blocks",
            "generate_doc_lengths": False,
            "intra_document_attention_masking": False,
            "document_boundary_semantics": (
                "EOS-derived segments constrain EMO expert pools only; attention and KDA "
                "span each full 8192-token sequence, without resets at internal EOS tokens."
            ),
            "instance_filter": {
                "repetition_min_period": 1,
                "repetition_max_period": 13,
                "repetition_max_count": 32,
                "rejected_instance": "mask all labels out of the training loss",
            },
            "transport": "pretokenized local copy; partner storage location TODO",
        },
        "batch": {
            "global_batch_size_tokens": TINY_BATCH_TOKENS if tiny else None,  # TODO: larger CBS
            "global_batch_size_sequences": 2048 if tiny else None,
            "microbatch_sequences_per_device": 4 if tiny else None,  # TODO: memory qualification
            "microbatch_tokens_per_device": 32768 if tiny else None,
            "gradient_accumulation_steps": 8 if tiny else None,
        },
        "optimizer": {
            "algorithm": "AdamW, decoupled weight decay, bias-corrected moments",
            "core_implementation": "OLMoDDPOptimizer (distributed, compiled, skip-step AdamW)",
            "peak_lr": TINY_PEAK_LR if tiny else None,  # TODO: larger-model LR transfer/sweep
            "betas": [0.9, 0.95],
            "epsilon": 1e-8,
            "weight_decay": 0.1,
            "weight_decay_exemptions": ["input token embedding table only"],
            "core_embedding_override": {"params": ["embeddings.weight"], "weight_decay": 0.0},
            "routed_expert_override": "separate parameter group; same LR and WD",
            "norm_and_untied_lm_head_weight_decay": 0.1,
            "global_gradient_clip_norm": 1.0,
            "skip_step": {
                "enabled": True,
                "rolling_interval_steps": 128,
                "sigma_factor": 6,
                "signals": ["loss", "pre-clipping global gradient norm"],
                "nonfinite_loss_or_gradient": "abort, not skip",
                "resume": "restore moments, step counters, and rolling histories",
            },
        },
        "lr_schedule": {
            "family": "WSD with independently forked decay jobs",
            "trunk_implementation": "ConstantWithWarmup",
            "units": "trainer global steps (including skipped optimizer updates)",
            "warmup_steps": WARMUP_STEPS,
            "warmup_start_lr": 0.0,
            "warmup_shape": "linear",
            "stable_phase": "constant peak LR, including at the trunk stopping horizon",
            "automatic_decay_in_trunk": False,
            "full_horizon_decay_fraction": None,  # TODO: choose final hero decay branch(es)
            "full_horizon_decay_start_step": None,
            "observed_tiny_2t_decay_example": {
                "fork_step": 108_000,
                "end_step": 120_000,
                "decay_steps": 12_000,
                "decay_shape": "linear peak LR to zero, no new warmup",
                "fraction_of_endpoint_budget": 0.1,
                "restore": "full model, optimizer, RNG and data state from trunk",
                "applies_to": "historical Tiny 2T branch only; not a 14T decision",
            },
        },
        "horizon": {
            "target_tokens": TINY_TARGET_TOKENS if tiny else None,
            "total_steps_ceiling": steps if tiny else None,
            "actual_tokens_at_ceiling": steps * TINY_BATCH_TOKENS if tiny else None,
            "early_stop": "Tiny initially stopped near 3T and resumed; no LR restart or decay",
        },
        "initialization": {
            "method": "OLMo-core InitMethod.normal (truncated normal, NOT untruncated)",
            "normal_mean": 0.0,
            "normal_std_before_truncation": 0.02,
            "truncation_bounds": [-0.06, 0.06],
            "matrix_scope": "embedding, LM head, attention, experts, router and latent projections",
            "depth_dependent_output_rescaling": False,
            "bias_values": 0.0,
            "rmsnorm_gains": 1.0,
            "per_head_qk_gains": 1.0,
            "scalable_softmax_head_gains": 1.0,
            "kda_A_log": "log(U[1,16]) per KDA head",
            "kda_dt_bias": 0.0,
            "model_weight_seed": 0,
            "process_seed": 12536,
            "warning": "Different backends/sharding/RNGs need not produce bitwise-identical weights.",
        },
        "precision": {
            "training_weights_and_main_matmuls": "bfloat16",
            "master_weights_and_adam_moments": "float32",
            "gradient_accumulation_and_reduction": "float32",
            "router_logits_and_combine_weights": "float32",
            "fp8_or_mxfp8": False,
            "activation_checkpointing": False if tiny else None,  # TODO: larger memory policy
        },
        "loss": {
            "objective": "causal next-token cross entropy",
            "label_ignore_index": -100,
            "sequence_end_label": "ignored (no next token available)",
            "lm_z_loss_weight": 1e-5,
        },
        "checkpointing": {
            "contents": ["model", "optimizer/master weights", "RNG", "data/trainer state"],
            "save_async": False,
            "resume_policy": "restore full state; do not reset optimizer, data or schedule",
            "interval_schedule": (
                [
                    {"through_step_inclusive": 18_000, "save_every_steps": 100},
                    {"through_step_inclusive": 60_000, "save_every_steps": 250},
                    {"through_step_inclusive": steps, "save_every_steps": 500},
                ]
                if tiny
                else None  # TODO: cadence from checkpoint size/storage throughput
            ),
            "trainer_deletes_checkpoints": False,
            "upload_and_retention": "separate service; partner implementation/storage policy TODO",
        },
        "validation": {
            "dataset": "v3-small-ppl-validation",
            "every_steps": 1000,
            "duration": "one epoch",
            "on_finish": True,
        },
        "nvidia_reference_execution_not_tpu_requirements": {
            "validated_for_this_rung": tiny,
            "device_type": "B300" if tiny else None,
            "num_devices": 64 if tiny else None,
            "nodes": 8 if tiny else None,
            "data_parallel_degree": 64 if tiny else None,
            "expert_parallel_degree": 1 if tiny else None,
            "pipeline_parallel_degree": 1 if tiny else None,
            "tensor_parallel_degree": 1 if tiny else None,
            "context_parallel_degree": 1 if tiny else None,
            "optimizer_state_sharding": "reduce-scatter distributed optimizer" if tiny else None,
            "dense_parameters": "replicated DDP" if tiny else None,
            "compile_model_and_optimizer": True,
            "kernel_reference": "optimized CuTe KDA/conv, FA4, fused MoE-v2 and EMO paths",
            "implementation_reference": (
                f"https://github.com/allenai/OLMo-core/blob/{CORE_COMMIT}"
                "/src/examples/olmo_ddp/olmoe3_small_hero.py"
            ),
        },
        "todos": [
            "Implement/qualify the architecture, precision, skip-step and collective behavior on TPU.",
            "Choose partner storage, data access, checkpoint upload/retention and TPU topology.",
            "Choose final 14T decay branch duration/start; the live trunk does not decay itself.",
            "EMO-on/off are supported; decide which training arm(s) the partner should run.",
        ],
    }
    if not tiny:
        result["todos"] += [
            "Choose global batch via CBS experiments; do not silently inherit 16Mi from Tiny.",
            "Choose peak LR after batch/size transfer and tuning; Tiny 1.1e-3 is a reference only.",
            "Choose token budget, device count, microbatch, accumulation, parallelism and recomputation.",
            "Qualify inherited optimizer, WD, init and 2k warmup at this model size.",
            "Choose larger-model checkpoint cadence, evaluation cadence and storage sizing.",
        ]
    return result


def validate() -> None:
    """Check dimensions, exact counts, Tiny batch arithmetic and explicit unknowns."""
    for name, g in FAMILY.items():
        assert g.latent_dim * 2 == g.d_model == g.expert_hidden_size
        assert all(d % 256 == 0 for d in (g.d_model, g.latent_dim, g.expert_hidden_size))
        assert g.n_layers % 8 == 0 and g.head_dim == 128 and g.n_heads == 2 * g.n_kv_heads
        d, n, h = g.d_model, g.n_layers, g.n_heads
        fa, kda = n // 8, n - n // 8
        kda_params = 769 * d * h + 512 * d + 100737 * h + 256
        fa_params = 4 * d * h * 128 + (h + h // 2) * 128 + h
        active = (
            2 * 100352 * d
            + 2 * d
            + n * 4 * d
            + kda * kda_params
            + fa * fa_params
            + 24 * d * d
            + (n - 1) * (28 * d * d + 512 * d)
        )
        total = active + (n - 1) * 744 * d * d  # 496 inactive experts/block, 3*d*(d/2) each
        assert (active, total) == (g.expected_active_params, g.expected_total_params), name
        t = training(name)
        if name == "tiny":
            assert t["batch"]["global_batch_size_tokens"] == 64 * 4 * 8192 * 8
            assert t["horizon"]["total_steps_ceiling"] == 834466
        else:
            assert t["batch"]["global_batch_size_tokens"] is None
            assert t["optimizer"]["peak_lr"] is None
            assert t["nvidia_reference_execution_not_tpu_requirements"]["num_devices"] is None


def main() -> None:
    """Print a self-contained JSON handoff; never launch or allocate anything."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-size", choices=("all", *FAMILY), default="all")
    parser.add_argument("--emo", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    validate()
    names = FAMILY if args.model_size == "all" else [args.model_size]
    print(
        json.dumps(
            {
                "schema_version": 1,
                "verified_date": "2026-09-14",
                "source_core_commit": CORE_COMMIT,
                "naming": "Tiny=current 0.794B hero; Small=previous 3.781B Large (not 3.2B).",
                "models": {
                    name: {"architecture": architecture(name, args.emo), "training": training(name)}
                    for name in names
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
