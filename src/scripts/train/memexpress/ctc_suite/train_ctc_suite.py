"""Task-agnostic LOCAL (torchrun, no Beaker/weka) SFT trainer for the CTC suite experiment
(``records/ctc-suite-scaling-plan.md`` §4/§11): one joint run per (task, arm) on the hybrid
Qwen3.5 models (Gated DeltaNet + full attention, 3:1), full vs document-chunked arms.

Generalized from ``attn_explore/Qwen3.5-0.8B-docchunk-mask-mix-contradiction-SFT-local.py`` +
the curriculum/hard-fail logic of its Qwen3-0.6B sibling. What it keeps:

  * distcp base loading (``load_path`` -> the shared per-scale converted base; model-only), with a
    ``.metadata`` guard so a mis-staged base cannot silently train from scratch;
  * the mask-mix curriculum derivation with BOTH divisor fixes (world_size AND
    micro-batch-instances) and the hard-fail check that the anneal actually lands on
    ``mix_end_p`` (see the mask-mix-ngpu-anneal bug);
  * ``metadata.json`` as a hard requirement of the shard dir (``num_instances`` drives the
    curriculum; ``max_example_len`` guards ``--seq-len``; ``marker_set`` is cross-checked).

Arms (``--variant``):

  * ``full``        -> plain causal attention (no ``document_chunk_attention``); the box markers in
                       the shard are just ordinary tokens. Upper-bound / O(N^2) arm.
  * ``chunked``     -> document-chunked mask on the full-attention blocks (GDN blocks are linear
                       attention and ignore ``chunk_ids``), no mixing. Eval-matched arm.
  * ``chunked-mix`` -> chunked + curriculum mask mixing 0.8 -> 0.0 (the project default recipe;
                       eval still uses the pure chunked mask).

Marker ids come from ``RESERVED_IDS["qwen3_5"]`` in :mod:`olmo_core.data.document_chunk_landmark`
-- never retyped. The shard must be built with ``--marker-set qwen3_5``
(``convert_unified_to_document_landmark.py``); this script hard-fails on a mismatch.

CPU-checkable dry run (no CUDA, nothing written)::

    PYTHONPATH=src python src/scripts/train/memexpress/ctc_suite/train_ctc_suite.py \\
      --task contradiction --data <shard-dir> --variant chunked-mix --model-scale 0.8b \\
      --dry-run --dry-run-world-size 8 --dry-run-n-examples 20000

Real run (via ``run_ctc_local.sbatch``; ``--variant chunked-mix`` forces ``--no-compile``)::

    torchrun --nproc_per_node=8 src/scripts/train/memexpress/ctc_suite/train_ctc_suite.py \\
      --task contradiction --data <shard-dir> --variant chunked-mix --model-scale 0.8b \\
      --run-name ctc-contra-chunkedmix-08b --save-checkpoint --no-compile
"""

import argparse
import datetime
import functools
import json
import os
import subprocess
from typing import Any, Dict, Optional

from olmo_core.config import DType
from olmo_core.data import TokenizerConfig
from olmo_core.data.composable import (
    ComposableDataLoaderConfig,
    LandmarkPackingInstanceSourceConfig,
    NumpyDocumentSourceConfig,
    PackingInstanceSourceConfig,
    PadToLengthInstanceSourceConfig,
)
from olmo_core.data.document_chunk_landmark import (
    RESERVED_IDS,  # canonical ids -- never retype
)
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.distributed.utils import get_rank, get_world_size
from olmo_core.float8 import Float8Config
from olmo_core.nn.attention import AttentionBackendName, AttentionType
from olmo_core.nn.attention.chunked_mask import mask_mix_standard_prob
from olmo_core.nn.lm_head import LMLossImplementation
from olmo_core.nn.rope import YaRNRoPEScalingConfig
from olmo_core.nn.transformer import (
    TransformerActivationCheckpointingMode,
    TransformerConfig,
)
from olmo_core.optim import LinearWithWarmup, OptimGroupOverride, SkipStepAdamWConfig
from olmo_core.train import (
    Duration,
    LoadStrategy,
    TrainerConfig,
    prepare_training_environment,
    teardown_training_environment,
)
from olmo_core.train.callbacks import (
    FlopMeterCallback,
    NestedFFNMoECallback,
    CheckpointerCallback,
    ConfigSaverCallback,
    GPUMemoryMonitorCallback,
    WandBCallback,
)
from olmo_core.train.train_module import (
    TransformerActivationCheckpointingConfig,
    TransformerContextParallelConfig,
    TransformerDataParallelConfig,
    TransformerDataParallelWrappingStrategy,
    TransformerTrainModuleConfig,
)
from olmo_core.utils import seed_all

try:  # package import (PYTHONPATH=src) or same-directory fallback (torchrun on the file path)
    from scripts.train.memexpress.ctc_suite.llama_configs import (
        LLAMA_MARKER_TOKENIZER,
        llama3_1_8B,
        llama3_2_3B,
    )
except ImportError:  # pragma: no cover
    import sys as _sys

    _sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from llama_configs import (  # type: ignore[no-redef]
        LLAMA_MARKER_TOKENIZER,
        llama3_1_8B,
        llama3_2_3B,
    )

try:  # package import (PYTHONPATH=src) or same-directory fallback (torchrun on the file path)
    from scripts.train.memexpress.ctc_suite.olmo3_configs import (
        OLMO3_MARKER_TOKENIZER,
        OLMO3_VOCAB_SIZE,
        olmo3_7B_ctc,
        olmo3_7B_ctc_swa,
    )
    from scripts.train.memexpress.ctc_suite.olmo_hybrid_configs import (
        OLMO_HYBRID_MARKER_TOKENIZER,
        OLMO_HYBRID_VOCAB_SIZE,
        olmo_hybrid_7B_ctc,
    )
except ImportError:  # pragma: no cover
    import sys as _sys

    _sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from olmo3_configs import (  # type: ignore[no-redef]
        OLMO3_MARKER_TOKENIZER,
        OLMO3_VOCAB_SIZE,
        olmo3_7B_ctc,
        olmo3_7B_ctc_swa,
    )
    from olmo_hybrid_configs import (  # type: ignore[no-redef]
        OLMO_HYBRID_MARKER_TOKENIZER,
        OLMO_HYBRID_VOCAB_SIZE,
        olmo_hybrid_7B_ctc,
    )

#: Supported model families. The trainer is family-agnostic: everything below (marker ids,
#: embedding size, tokenizer, model factory) is keyed on the family, which is auto-detected from
#: the shard's ``marker_set`` (or forced via ``--model-family``). ``qwen3_5`` is the GDN+attn
#: hybrid; ``qwen3`` is the plain dense/causal family. A shard built for one family MUST train with
#: that family -- a wrong-tokenizer shard produces plausible numbers, not a crash.
#: Per-family embedding-matrix size (rows), i.e. the converted base checkpoint's embedding size.
FAMILY_VOCAB_SIZE = {
    "qwen3_5": 248320,  # Qwen3.5-{0.8B,4B,9B}-Base embedding size
    "qwen3": 151936,  # Qwen3-{0.6B,1.7B,4B,8B}-Base embedding size
    "gemma": 262208,  # Gemma-3-{1B,4B,12B,27B} embedding size (262144 real ids + 64 padding rows)
    "llama": 128256,  # Llama-3.x embedding size (no padding rows: vocab == matrix rows)
    "olmo3": OLMO3_VOCAB_SIZE,  # dolma2 padded embedding size (100278 real ids + 74 padding rows)
}

#: Per-family HF tokenizer identifier (all sizes within a family share one tokenizer).
FAMILY_TOKENIZER = {
    "qwen3_5": "Qwen/Qwen3.5-0.8B-Base",
    "qwen3": "Qwen/Qwen3-4B",
    "gemma": "google/gemma-3-4b-pt",
    # Llama 3 has no ``<|box_start|>``/``<|box_end|>`` tokens, so the suite uses a local tokenizer
    # copy in which reserved slots 128002/128003 are RENAMED to those strings (ids unchanged) --
    # built by ``src/scripts/data/make_llama_marker_tokenizer.py``. Point at the patched copy, not
    # the stock repo, or the converter's marker-id verification fails.
    "llama": LLAMA_MARKER_TOKENIZER,
    # Same story for OLMo: dolma2 has no ``<|box_start|>``/``<|box_end|>``, so the suite uses a
    # local copy in which the unused ``<|extra_id_1|>``/``<|extra_id_2|>`` slots (100266/100267)
    # are RENAMED to those strings, ids unchanged.
    "olmo3": OLMO3_MARKER_TOKENIZER,
}

#: Per-family model factories, keyed by ``--model-scale``.
MODEL_FACTORIES = {
    "qwen3_5": {
        "0.8b": TransformerConfig.qwen3_5_0_8B,
        "2b": TransformerConfig.qwen3_5_2B,
        "4b": TransformerConfig.qwen3_5_4B,
        "9b": TransformerConfig.qwen3_5_9B,
    },
    "qwen3": {
        "0.6b": TransformerConfig.qwen3_0_6B,
        "1.7b": TransformerConfig.qwen3_1_7B,
        "4b": TransformerConfig.qwen3_4B,
        "8b": TransformerConfig.qwen3_8B,
    },
    # Gemma 3 (local/global sliding-window hybrid). ``gemma3_4B``'s factory defaults do NOT match
    # the released ``google/gemma-3-4b-pt`` checkpoint (8 query heads / head_dim 256, and the
    # global layers carry HF ``rope_scaling={"rope_type":"linear","factor":8}``), so the exact HF
    # dims are pinned here rather than inherited -- a silently mismatched shape loads the weights
    # wrong and produces plausible garbage instead of crashing.
    "gemma": {
        "4b": functools.partial(
            TransformerConfig.gemma3_4B,
            n_heads=8,
            head_dim=256,
            global_rope_linear_scaling_factor=8.0,
        ),
    },
    # Llama 3.x (plain dense/causal, GQA). olmo-core ships no 3B factory, so ``llama_configs``
    # builds one with the generic ``llama_like`` using dims read off ``meta-llama/Llama-3.2-3B``'s
    # own config.json, and hard-asserts the resulting parameter count against it. 8b maps onto
    # olmo-core's own ``llama3_8B`` but still overrides ``rope_scaling``: the class default factor
    # (32.0) is Llama-3.2's, and Llama-3.1-8B's is 8.0 -- see llama_configs.LLAMA3_1_8B_HF_SHAPE.
    # 8b is the SIZE CONTROL for the 3b contradiction result (0.362 at 2k vs ~0.83 for Qwen3.5-4B
    # and OLMo-3-7B): until a bigger Llama is measured on the identical task, "Llama is weak at
    # N^2 pair-finding" and "3B is too small for it" are indistinguishable.
    "llama": {
        "3b": llama3_2_3B,
        "8b": llama3_1_8B,
    },
    # OLMo 3 (``allenai/Olmo-3-1025-7B``), this repo's native family. ``olmo3_configs`` wraps the
    # stock ``olmo3_7B`` factory to disable sliding-window attention (DocumentChunkedAttention
    # refuses it, so the chunked arm cannot have it -- disabled in BOTH arms to keep the mask the
    # only manipulated variable) and to apply the checkpoint's YaRN scaling to every layer.
    "olmo3": {
        # Keeps Olmo 3's native 3:1 sliding:full backbone and chunks ONLY the 8 full-attention
        # layers -- the exact counterpart of chunking Qwen3.5's non-GDN blocks.
        "7b": olmo3_7B_ctc_swa,
        # Superseded no-sliding-window variant: disabling the windows costs the base model ~41x CE
        # before training (olmo3_swa_ablation.py). Kept so the first wave of runs reproduces.
        "7b-noswa": olmo3_7B_ctc,
        # ``allenai/Olmo-Hybrid-7B``: same size and same training data as Olmo-3-1025-7B, but a 3:1
        # LINEAR (Gated-DeltaNet):full backbone instead of 3:1 sliding:full. It belongs here rather
        # than in a family of its own because it shares dolma2, the patched marker tokenizer, the
        # marker ids and the 100352-row embedding -- so it trains on the SAME olmo3 shards, and the
        # shard's ``marker_set`` cross-check passes without an exemption. That shared everything is
        # the whole point: pairing 7b against 7b-hybrid varies the attention backbone and nothing
        # else, which no other entry in this table can do. Chunks the 8 full-attention layers only,
        # exactly as ``olmo3_7B_ctc_swa`` and the Qwen3.5 arms do.
        "7b-hybrid": olmo_hybrid_7B_ctc,
    },
}


def resolve_family(opts: argparse.Namespace) -> str:
    """Resolve the model family: explicit ``--model-family`` or auto-detect from the shard.

    Auto-detection reads ``metadata.json:marker_set`` (written by
    ``convert_unified_to_document_landmark.py --marker-set ...``), so the tokenizer that built the
    shard picks the model/marker ids that train on it -- they cannot silently mismatch. Falls back
    to ``qwen3_5`` only when the shard predates the ``marker_set`` field (all current shards write
    it), preserving the original Qwen3.5-only behavior.

    :param opts: Parsed CLI options (uses ``model_family`` and ``data``).

    :returns: A key of :data:`MODEL_FACTORIES` (``"qwen3_5"`` or ``"qwen3"``).

    :raises SystemExit: If the resolved family is unknown.
    """
    fam = opts.model_family
    if fam == "auto":
        fam = "qwen3_5"
        meta_path = os.path.join(opts.data, "metadata.json")
        if os.path.exists(meta_path):
            ms = json.load(open(meta_path)).get("marker_set")
            if ms:
                fam = ms
    if fam not in MODEL_FACTORIES:
        raise SystemExit(
            f"unknown model family {fam!r}; known: {sorted(MODEL_FACTORIES)}. "
            "Pass --model-family explicitly or rebuild the shard with a known --marker-set."
        )
    return fam


# Per-scale converted olmo distcp bases (model-only). Only 0.8B exists today; 4B/9B must be passed
# explicitly (--base-checkpoint / BASE_SRC) once converted -- repeat the marker audit first (§4).
DEFAULT_BASE_CHECKPOINT = {
    "0.8b": "/scratch/users/prasann/cpt_mix_ckpts/q35-08b-base-modelonly/model_and_optim",
}
SAVE_ROOT = "/scratch/users/prasann/olmo_ckpts"
WORK_DIR = "/scratch/users/prasann/longctx_sft_qwen/dataset-cache-ctc-suite"

# Hyperparams inherited from the source attn_explore scripts (NOT contradiction-specific).
LR = 5e-5
MEM_FREQ_SPARSE = (
    63  # landmark block = 64 (63 content + 1 landmark), matches every landmark lineage
)
NUM_EPOCHS = 3
WANDB_PROJECT = "memory-networks"


def read_shard_metadata(data_dir: str, seq_len: int, family: str) -> Dict[str, Any]:
    """Read and validate the shard's ``metadata.json``.

    :param data_dir: Shard dir from ``convert_unified_to_document_landmark.py``.
    :param seq_len: The training sequence length, checked against ``max_example_len``.
    :param family: The resolved model family; the shard's ``marker_set`` must match it.

    :returns: The parsed metadata dict.

    :raises SystemExit: If the metadata is missing, the marker set disagrees with ``family``, or
        ``seq_len`` is too short for the shard.
    """
    ids = RESERVED_IDS[family]
    meta_path = os.path.join(data_dir, "metadata.json")
    if not os.path.exists(meta_path):
        raise SystemExit(
            f"{meta_path} missing. The docchunk trainers REQUIRE the converter's metadata.json "
            "(num_instances drives the mask-mix anneal; max_example_len guards --seq-len)."
        )
    meta = json.load(open(meta_path))
    shard_marker_set = meta.get("marker_set")
    if shard_marker_set is not None and shard_marker_set != family:
        raise SystemExit(
            f"shard {data_dir} was built with --marker-set {shard_marker_set!r}, but this run "
            f"resolved to family {family!r}. A wrong-tokenizer shard produces plausible numbers, "
            "not a crash -- rebuild the shard or pass the matching --model-family, do not override."
        )
    for key, want in [("doc_start_id", ids.doc_start), ("doc_end_id", ids.doc_end)]:
        got = meta.get(key)
        if got is not None and int(got) != want:
            raise SystemExit(f"shard metadata {key}={got} != canonical {want} ({family})")
    if meta.get("max_example_len", 0) > seq_len:
        raise SystemExit(
            f"--seq-len={seq_len} < max example length {meta['max_example_len']} "
            f"(PadToLength would SKIP the long examples). Raise --seq-len."
        )
    return meta


def derive_batch_geometry(global_batch: int, world_size: int, micro_batch_instances: int) -> int:
    """Validate the batch geometry and return instances-per-rank-per-step.

    :param global_batch: Instances per optimizer step (across all ranks).
    :param world_size: Number of DP ranks.
    :param micro_batch_instances: Instances per rank per FORWARD.

    :returns: ``global_batch // world_size`` (each rank's share of a step).

    :raises SystemExit: If a rank's share is < 1 or ``micro_batch_instances`` exceeds it --
        olmo-core would otherwise silently reinterpret the batch (quiet optimization drift).
    """
    per_rank = global_batch // world_size
    if per_rank < 1 or micro_batch_instances > per_rank:
        raise SystemExit(
            f"--micro-batch-instances={micro_batch_instances} invalid: with --global-batch="
            f"{global_batch} and world_size={world_size} each rank holds {per_rank} instance(s). "
            f"Use --global-batch >= world_size and --micro-batch-instances in [1, {per_rank}]."
        )
    return per_rank


def derive_mask_mix_curriculum(
    *,
    n_examples: int,
    epochs: int,
    global_batch: int,
    world_size: int,
    micro_batch_instances: int,
    mix_start_p: float,
    mix_end_p: float,
) -> Dict[str, Any]:
    """Derive the curriculum's ``mix_total_forwards`` and hard-check the anneal lands.

    ``mix_total_forwards`` must be the number of FORWARDS a rank actually performs, because that
    is what the curriculum's counter counts. Two divisors, both learned the hard way:

      * ``world_size`` -- data is sharded across DP ranks (the original mask-mix-ngpu-anneal bug:
        p stalled at ``mix_start_p * (1 - 1/world_size)``, worse the more GPUs).
      * ``micro_batch_instances`` -- a forward carries this many instances, so raising it CUTS the
        forward count by the same factor (observed 2026-07-16: p ended at 0.601 instead of 0.0).

    The invariant: forwards_per_rank == optimizer steps * (per-rank instances / micro-batch).

    :param n_examples: Instances in the shard (``metadata.json: num_instances``).
    :param epochs: Training epochs.
    :param global_batch: Instances per optimizer step.
    :param world_size: Number of DP ranks.
    :param micro_batch_instances: Instances per rank per forward.
    :param mix_start_p: Curriculum start collapse probability.
    :param mix_end_p: Curriculum end collapse probability.

    :returns: Dict with ``mix_total_forwards``, ``steps``, ``forwards_per_rank``, ``final_p``.

    :raises SystemExit: If the predicted final ``p_standard`` misses ``mix_end_p`` by > 0.01 --
        a silent shortfall does not crash and does not look wrong in the loss, it just means the
        model never trained predominantly on its target mask, voiding the run.
    """
    per_rank = derive_batch_geometry(global_batch, world_size, micro_batch_instances)
    mix_total_forwards = max(1, (n_examples * epochs) // (world_size * micro_batch_instances))
    # Independently re-derive the forward count from the STEP count and assert the curriculum
    # will actually land on mix_end_p, instead of trusting one formula to agree with itself.
    steps = max(1, (n_examples * epochs) // global_batch)
    forwards_per_rank = steps * (per_rank // micro_batch_instances)
    final_p = mask_mix_standard_prob(
        forwards_per_rank - 1,
        mix_start_p=mix_start_p,
        mix_end_p=mix_end_p,
        mix_total_forwards=mix_total_forwards,
    )
    # Tolerance 0.01, not 1e-3: integer-division rounding legitimately leaves the last forward a
    # hair off. We are catching gross shortfalls like the observed 0.601, not rounding dust.
    if abs(final_p - mix_end_p) > 0.01:
        raise SystemExit(
            f"mask-mix curriculum would end at p_standard={final_p:.4f}, not mix_end_p="
            f"{mix_end_p}: mix_total_forwards={mix_total_forwards} disagrees with the "
            f"{forwards_per_rank} forwards this rank will actually run. The model would keep "
            f"training on PLAIN CAUSAL for {final_p:.0%} of forwards forever, silently voiding "
            "the full-vs-chunked comparison. Check the world_size / micro_batch_instances "
            "divisors."
        )
    return dict(
        mix_total_forwards=mix_total_forwards,
        steps=steps,
        forwards_per_rank=forwards_per_rank,
        final_p=final_p,
    )


def build_model_config(opts: argparse.Namespace) -> TransformerConfig:
    """Build the per-scale :class:`TransformerConfig` for the requested family/variant.

    :param opts: Parsed CLI options (``model_family``, ``model_scale``, ``variant``, mix knobs).

    :returns: The model config, with ``document_chunk_attention`` set for the chunked arms.
    """
    ids = RESERVED_IDS[opts.model_family]
    factory = MODEL_FACTORIES[opts.model_family][opts.model_scale]
    # Pin flash_2 by default so the saved config.json supports KV-cached generation at eval (the
    # qwen3_5 factories default attn_backend=None otherwise; matches the source attn_explore
    # scripts). ``--attn-backend torch`` is an escape hatch for clusters where flash-attn isn't
    # importable/compatible (e.g. a fresh env on an unverified GPU arch) -- SDPA works everywhere
    # but does not support KV-cached generation, so eval on a torch-backend checkpoint needs the
    # full-precompute path, not incremental decoding.
    qwen_kwargs: Dict[str, Any] = dict(
        vocab_size=opts.vocab_size, attn_backend=AttentionBackendName(opts.attn_backend)
    )
    # RoPE theta override (NTK-style context extension): raise the base rope_theta so the model's
    # RoPE frequencies cover the longer context. A plain factory param (applies to the base block,
    # no block_overrides), so it unambiguously takes effect at runtime -- unlike YaRN, whose
    # with_rope_scaling path writes per-layer overrides. For Qwen3-4B (native 32k, theta 1M) at
    # 256k, the NTK-aware value is ~8M. Off by default (0). Prefer this over --rope-yarn-factor
    # for large (>=8x) extensions where YaRN was found to plateau.
    if opts.rope_theta and opts.rope_theta > 0:
        qwen_kwargs["rope_theta"] = opts.rope_theta
    if opts.variant in ("chunked", "chunked-mix"):
        # Chunked mask on the full-attention blocks only; GDN blocks ignore chunk_ids.
        qwen_kwargs["document_chunked"] = True
        qwen_kwargs["cross_doc_mode"] = "chunked"
    model_config = factory(**qwen_kwargs)
    if opts.variant == "sparselandmark":
        # Swap the full-attention mixer for sparse landmark attention (same swap as
        # cpt/Qwen3.5-4B-sparse-landmark-dolma3longmino.py): full attention within a 64-token
        # block, past blocks visible only through their single landmark token. The qwen3_5
        # elementwise output gate on the mixer is KEPT (sparse landmark applies it, and w_g loads
        # from the base checkpoint). GDN blocks are untouched. Data must be landmark-inserted
        # (LandmarkPackingInstanceSource below), and the base checkpoint must have the landmark
        # embedding row (ids.landmark) repaired -- it is untrained in the raw conversion.
        blk = model_config.block
        mixer = blk["attn"].sequence_mixer if isinstance(blk, dict) else blk.sequence_mixer
        mixer.name = AttentionType.sparse_landmark
        mixer.mem_freq = MEM_FREQ_SPARSE
        mixer.num_landmarks = 1
    if opts.variant == "pooledkv":
        # Swap the full-attention mixer for pooled-doc-KV attention (same swap mechanics as
        # sparselandmark): plain causal topology, but most context documents' K/V collapse -- for
        # queries outside the document -- to a single mean-pooled slot with a +log(doc_len) logit
        # bias; gold + a few random docs keep real per-token KV (keep set via the
        # --pooled-gold-sidecar hook, else the seeded random fallback). GDN blocks are untouched.
        # Inference needs NO special path: the checkpoint runs ordinary full attention.
        blk = model_config.block
        mixer = blk["attn"].sequence_mixer if isinstance(blk, dict) else blk.sequence_mixer
        mixer.name = AttentionType.pooled_doc_kv
        mixer.pooled_keep_prob = opts.pooled_keep_prob
        mixer.pooled_keep_seed = opts.seed
        mixer.pooled_len_bias = not opts.pooled_no_len_bias
    # YaRN RoPE context extension for long-context runs (base native ctx is 32k for Qwen3; the
    # 256k rung needs factor ~8). Off by default (--rope-yarn-factor 0) so short-rung runs are
    # byte-identical. Mirrors sft_longctx/Qwen3-4B-dense-longctx-SFT.py (factor=2 for 64k).
    if opts.rope_yarn_factor and opts.rope_yarn_factor > 1:
        if opts.model_family == "qwen3_5":
            # with_rope_scaling refuses hybrid (named-block) models; the GDN-hybrid needs a
            # per-block/theta-based extension instead. Skip here and warn -- the hybrid's RoPE
            # extension to >native is a separate (open) modeling choice; the SFT still adapts.
            print(
                "[ctc-suite] WARNING: --rope-yarn-factor ignored for qwen3_5 hybrid "
                "(with_rope_scaling unsupported for named-block models); training at native RoPE.",
                flush=True,
            )
        else:
            model_config = model_config.with_rope_scaling(
                YaRNRoPEScalingConfig(
                    factor=float(opts.rope_yarn_factor),
                    beta_fast=32,
                    beta_slow=1,
                    old_context_len=opts.rope_old_context,
                )
            )
    # Fused-linear CE: never materialize float logits over the full 248k vocab. At seq-len
    # 40960 the unfused path needs ~38 GiB per rank just for logits.float() and OOMs H200s
    # (same setting as the proven 40k-seq sft_docchunk Beaker scripts).
    model_config.lm_head.loss_implementation = LMLossImplementation.fused_linear
    if opts.variant in ("chunked", "chunked-mix", "pooledkv"):
        mix_keys: Dict[str, Any] = {}
        if opts.variant == "chunked-mix":
            mix_keys = dict(
                mix_start_p=opts.mix_start_p,
                mix_end_p=opts.mix_end_p,
                mix_total_forwards=opts._mix_total_forwards,  # filled by the curriculum deriv
                mix_seed=opts.mix_seed,
                mix_log_interval=opts.mix_log_interval,
            )
        model_config.document_chunk_attention = {
            "doc_start_id": ids.doc_start,
            "doc_end_id": ids.doc_end,
            "eos_id": ids.eos,
            "mode": "chunked",
            **mix_keys,
        }
    return model_config


#: Scales that need FULL parameter sharding + activation checkpointing to fit 40960 on an 80GB
#: H100 (proven OOM on jupiter: 76.71 GiB allocated then OOM in the first dry-run batch at
#: shard_degree=1 / no AC). 0.8B already fits at shard_degree=1 / no AC on 141GB H200s -- keep it
#: on that path so its proven runs don't change. 2B is grouped with the large scales: it is 2.4B
#: params and the model-scale study runs it on 80GB A100s, where even 0.8B needs full sharding +
#: AC at 40960 (lambda_cluster.md).
_LARGE_SCALES = ("2b", "4b", "9b")


def resolve_activation_checkpointing(opts: argparse.Namespace) -> str:
    """Resolve ``--activation-checkpointing`` ("auto" -> scale-dependent default).

    :param opts: Parsed CLI options (``model_scale``, ``activation_checkpointing``).

    :returns: ``"full"`` or ``"none"``.
    """
    if opts.activation_checkpointing != "auto":
        return opts.activation_checkpointing
    return "full" if opts.model_scale in _LARGE_SCALES else "none"


def resolve_shard_degree(opts: argparse.Namespace, world_size: int) -> int:
    """Resolve ``--shard-degree`` ("auto" -> scale-dependent default).

    0.8B keeps its proven ``shard_degree=1`` default (single-GPU-resident params fit 141GB
    H200s); 4B/9B default to full FSDP over ``world_size`` (matches the proven 40k-seq docchunk
    pattern in ``_docchunk_5task_32k_nocpt_common.py``, needed to fit an 80GB H100).

    :param opts: Parsed CLI options (``model_scale``, ``shard_degree``).
    :param world_size: DP world size (real, or hypothetical under ``--dry-run``).

    :returns: The resolved ``shard_degree``.
    """
    if opts.shard_degree:
        return opts.shard_degree
    return world_size if opts.model_scale in _LARGE_SCALES else 1


def _build_ac_config(ac_mode: str, ac_budget: float):
    """Build the activation-checkpointing config for the resolved mode.

    ``full`` = checkpoint every block (most memory-frugal, but breaks under compile+CP -- see
    build_train_module_config). ``budget`` = FLOP-optimal checkpointing to hit ``ac_budget`` (the
    compile+CP-safe path, from the proven long-context recipe). ``none`` = no AC.
    """
    if ac_mode == "full":
        return TransformerActivationCheckpointingConfig(
            mode=TransformerActivationCheckpointingMode.full
        )
    if ac_mode == "budget":
        return TransformerActivationCheckpointingConfig(
            mode=TransformerActivationCheckpointingMode.budget,
            activation_memory_budget=ac_budget,
        )
    return None


def build_train_module_config(
    opts: argparse.Namespace, world_size: int
) -> TransformerTrainModuleConfig:
    """Build the :class:`TransformerTrainModuleConfig` (FSDP + activation checkpointing).

    Shared by ``dry_run`` (to print the resolved config) and ``build_and_fit`` so the two never
    drift apart.

    :param opts: Parsed CLI options.
    :param world_size: DP world size (real, or hypothetical under ``--dry-run``).

    :returns: The train-module config, unbuilt (no CUDA needed to construct it).
    """
    ac_mode = resolve_activation_checkpointing(opts)
    # Context parallelism (long-context arms): CP shards the SEQUENCE/activations across cp_degree
    # ranks; params/optimizer are sharded by the FSDP dp dimension, which is world_size//cp_degree.
    # Ulysses only (ring flash-attn isn't guaranteed in the image, and GDN blocks reject ring), so
    # cp_degree is bounded by n_kv_heads (dense=8, hybrid=4). Off by default (--cp-degree 0/1).
    cp_degree = opts.cp_degree if opts.cp_degree and opts.cp_degree > 1 else 1
    if world_size % cp_degree != 0:
        raise SystemExit(f"world_size {world_size} not divisible by --cp-degree {cp_degree}")
    dp_world_size = world_size // cp_degree
    shard_degree = resolve_shard_degree(opts, dp_world_size)
    cp_config = (
        TransformerContextParallelConfig.ulysses(degree=cp_degree) if cp_degree > 1 else None
    )
    return TransformerTrainModuleConfig(
        rank_microbatch_size=opts.micro_batch_instances * opts.seq_len,
        max_sequence_length=opts.seq_len,
        optim=SkipStepAdamWConfig(
            lr=opts.lr,
            weight_decay=0.0,
            betas=(0.9, 0.95),
            group_overrides=[
                OptimGroupOverride(params=["embeddings.weight"], opts=dict(weight_decay=0.0))
            ]
            + (
                [
                    # The router decides discrete routing from a cold start and wants a livelier
                    # LR than the pretrained backbone; the gains are 1-D and must not be decayed.
                    OptimGroupOverride(
                        params=[
                            "blocks.*.feed_forward._nffn_router.*",
                            "blocks.*.feed_forward._nffn_gain",
                        ],
                        opts=dict(lr=opts.router_lr, weight_decay=0.0),
                    )
                ]
                if opts.variant == "ffnmoe"
                else []
            ),
        ),
        scheduler=LinearWithWarmup(warmup_fraction=0.03, alpha_f=0.0),
        compile_model=opts.compile,
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.hsdp if cp_degree > 1 else DataParallelType.fsdp,
            param_dtype=DType.bfloat16,
            reduce_dtype=DType.float32,
            wrapping_strategy=TransformerDataParallelWrappingStrategy.full,
            shard_degree=shard_degree,
        ),
        cp_config=cp_config,
        # Activation checkpointing. FULL-block AC fits 4B/9B at seq_len=40960 on 80GB H100s.
        # BUT full-block AC is INCOMPATIBLE with torch.compile + Ulysses CP: the block boundary
        # straddles the CP all-to-all, so the recomputed activation has a different shape/dtype
        # than the saved one ("recomputed metadata != saved metadata" crash at the dry-run). The
        # proven long-context CP recipe (sft_longctx/Qwen3-4B-dense-longctx-SFT.py) uses BUDGET AC
        # instead, which places checkpoints FLOP-optimally and avoids the all-to-all boundary.
        ac_config=_build_ac_config(ac_mode, opts.ac_budget),
        float8_config=Float8Config(enabled=False),
        z_loss_multiplier=None,
        max_grad_norm=1.0,
    )


def build_provenance(opts: argparse.Namespace, world_size: int) -> Dict[str, Any]:
    """Assemble the run's ``provenance.json`` payload (results-hub-ingestible; never fabricated).

    :param opts: Parsed CLI options.
    :param world_size: DP world size of the run.

    :returns: The provenance dict.
    """
    repo_dir = os.path.dirname(os.path.abspath(__file__))
    try:
        git_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo_dir, capture_output=True, text=True, timeout=10
        ).stdout.strip()
    except Exception:
        git_commit = "unknown"
    return {
        "experiment": "ctc_suite",
        "script": os.path.abspath(__file__),
        "git_commit": git_commit or "unknown",
        "task": opts.task,
        "variant": opts.variant,
        "model_family": opts.model_family,
        "model_scale": opts.model_scale,
        "data": os.path.abspath(opts.data),
        "marker_set": {"family": opts.model_family, **RESERVED_IDS[opts.model_family]._asdict()},
        "world_size": world_size,
        "start_time": os.environ.get("CTC_LAUNCH_TS")
        or datetime.datetime.now().isoformat(timespec="seconds"),
        "args": {k: v for k, v in vars(opts).items() if not k.startswith("_")},
    }


def announce_wandb(opts: argparse.Namespace) -> None:
    """Print the wandb group URL at launch (project directive: always surface it)."""
    if not opts.wandb:
        return
    group = opts.wandb_group or opts.run_name
    entity = opts.wandb_entity or os.environ.get("WANDB_ENTITY")
    if entity:
        print(
            f"[ctc-suite] wandb group: https://wandb.ai/{entity}/{WANDB_PROJECT}/groups/{group}",
            flush=True,
        )
    else:
        print(
            f"[ctc-suite] wandb project={WANDB_PROJECT} group={group} "
            "(set --wandb-entity / WANDB_ENTITY for a direct group URL)",
            flush=True,
        )


def resolve_plan(opts: argparse.Namespace, world_size: int) -> Dict[str, Any]:
    """Resolve everything derivable without CUDA: metadata, batch geometry, curriculum, configs.

    :param opts: Parsed CLI options. ``opts._mix_total_forwards`` is filled as a side effect for
        the chunked-mix variant.
    :param world_size: DP world size (real, or hypothetical under ``--dry-run``).

    :returns: Dict with ``meta``, ``n_examples``, ``per_rank``, ``curriculum`` (or None),
        ``model_config``, ``instance_source_config``, ``data_loader_config``.
    """
    ids = RESERVED_IDS[opts.model_family]
    meta = read_shard_metadata(opts.data, opts.seq_len, opts.model_family)
    n_examples = int(opts.dry_run_n_examples or meta["num_instances"])
    if meta.get("task") and meta["task"] != opts.task:
        print(
            f"[ctc-suite] WARNING: --task {opts.task} != shard metadata task {meta['task']!r}",
            flush=True,
        )
    # Under context parallelism, instances are distributed over the DP dimension (world_size/cp),
    # not all ranks -- each CP group cooperatively processes ONE instance's sequence.
    cp_degree = opts.cp_degree if opts.cp_degree and opts.cp_degree > 1 else 1
    dp_world_size = world_size // cp_degree
    per_rank = derive_batch_geometry(opts.global_batch, dp_world_size, opts.micro_batch_instances)
    curriculum: Optional[Dict[str, Any]] = None
    if opts.variant == "chunked-mix":
        curriculum = derive_mask_mix_curriculum(
            n_examples=n_examples,
            epochs=opts.epochs,
            global_batch=opts.global_batch,
            world_size=dp_world_size,
            micro_batch_instances=opts.micro_batch_instances,
            mix_start_p=opts.mix_start_p,
            mix_end_p=opts.mix_end_p,
        )
        opts._mix_total_forwards = curriculum["mix_total_forwards"]

    # Family-specific tokenizer (its OWN vocab; pad == eos). The Qwen3.5 path keeps its own vocab
    # (NOT TokenizerConfig.qwen3()); the plain-Qwen3 path uses the Qwen3 vocab/eos.
    tokenizer_config = TokenizerConfig(
        vocab_size=opts.vocab_size,
        eos_token_id=ids.eos,
        pad_token_id=ids.eos,
        bos_token_id=None,
        identifier=FAMILY_TOKENIZER[opts.model_family],
    )
    # Instance source. Default (--no-pack): one example per instance, padded to seq_len (the proven
    # short-rung path). With --pack: greedily bin-pack WHOLE examples into seq_len windows (no
    # cross-example splitting) and emit cu_doc_lens for intra-example masking -- the only tractable
    # option for a mixed 8k..256k length distribution, where padding every short example up to a
    # 256k seq_len would be a ~32x compute waste. Packing composes with CP (proven in the
    # sft_longctx -packed- scripts: "Packing is supported under CP"). Same npy format either way.
    if opts.variant == "sparselandmark":
        # Landmark-packed data path (mirrors the proven singletask_ladder 3variant script):
        # first-fit bin-packs whole SFT examples into block-aligned landmark windows and inserts
        # the landmark token every MEM_FREQ_SPARSE content tokens, emitting doc_lens so the sparse
        # kernel's doc_id keeps examples from attending each other's landmarks.
        if opts.seq_len % (MEM_FREQ_SPARSE + 1) != 0:
            raise SystemExit(
                f"--variant sparselandmark requires --seq-len divisible by the landmark block "
                f"({MEM_FREQ_SPARSE + 1}); got {opts.seq_len}."
            )
        instance_source_config = LandmarkPackingInstanceSourceConfig(
            source=NumpyDocumentSourceConfig(
                source_paths=[f"{opts.data}/token_ids_part_*.npy"],
                tokenizer=tokenizer_config,
                label_mask_paths=[f"{opts.data}/labels_mask_*.npy"],
                expand_glob=True,
            ),
            sequence_length=opts.seq_len,
            mem_freq=MEM_FREQ_SPARSE,
            mem_id=ids.landmark,
            pad_id=ids.pad,
        )
    elif opts.pack:
        # HARD GUARD: the packer builds a SegmentTree over max_sequence_length, which asserts
        # log2(N) is an integer. A non-power-of-2 --seq-len (e.g. 40960) therefore dies with a bare
        # "N should be a power of 2" -- but only AFTER the base checkpoint loads and the mesh is
        # built (~15 min at 4B), and only on rank 0, so every other rank then hangs until the 900 s
        # gloo timeout. One bad argument costs a full node-hour and reports itself as a distributed
        # timeout. Fail here instead, before anything expensive happens.
        if opts.seq_len & (opts.seq_len - 1) != 0:
            lo = 1 << (opts.seq_len.bit_length() - 1)
            raise SystemExit(
                f"--pack requires a power-of-2 --seq-len (the instance packer's SegmentTree "
                f"asserts it); got {opts.seq_len}. Use {lo} or {lo * 2} "
                f"(and keep --seq-len >= the shard's max_example_len)."
            )
        instance_source_config = PackingInstanceSourceConfig.from_npy(
            f"{opts.data}/token_ids_part_*.npy",
            tokenizer=tokenizer_config,
            sequence_length=opts.seq_len,
            max_sequence_length=opts.seq_len,
            label_mask_paths=[f"{opts.data}/labels_mask_*.npy"],
            expand_glob=True,
        )
    else:
        instance_source_config = PadToLengthInstanceSourceConfig.from_npy(
            f"{opts.data}/token_ids_part_*.npy",
            tokenizer=tokenizer_config,
            sequence_length=opts.seq_len,
            label_mask_paths=[f"{opts.data}/labels_mask_*.npy"],
            expand_glob=True,
        )
    data_loader_config = ComposableDataLoaderConfig(
        tokenizer=tokenizer_config,
        work_dir=opts.work_dir or WORK_DIR,
        global_batch_size=opts.global_batch * opts.seq_len,
        seed=34521 + opts.seed,
        num_workers=opts.num_workers,
    )
    return dict(
        meta=meta,
        n_examples=n_examples,
        per_rank=per_rank,
        curriculum=curriculum,
        model_config=build_model_config(opts),
        tokenizer_config=tokenizer_config,
        instance_source_config=instance_source_config,
        data_loader_config=data_loader_config,
    )


def dry_run(opts: argparse.Namespace) -> None:
    """Build all configs without CUDA / distributed init and print the resolved plan."""
    world_size = opts.dry_run_world_size
    plan = resolve_plan(opts, world_size)
    model_config = plan["model_config"]
    print("[ctc-suite] DRY RUN -- configs only, no CUDA, nothing trained, nothing written")
    print(
        f"  task={opts.task} variant={opts.variant} scale={opts.model_scale} "
        f"run_name={opts.run_name}"
    )
    print(
        f"  data={os.path.abspath(opts.data)} n_examples={plan['n_examples']}"
        + (
            f" (OVERRIDE; shard has {plan['meta']['num_instances']})"
            if opts.dry_run_n_examples
            else " (metadata)"
        )
    )
    ids = RESERVED_IDS[opts.model_family]
    print(
        f"  family={opts.model_family} marker_set={opts.model_family} "
        f"doc_start={ids.doc_start} doc_end={ids.doc_end} eos={ids.eos} "
        f"vocab_size={opts.vocab_size}"
    )
    print(
        f"  model: {opts.model_family}_{opts.model_scale} n_layers={model_config.n_layers} "
        f"params={model_config.num_params:,} "
        f"document_chunk_attention={model_config.document_chunk_attention}"
    )
    print(
        f"  seq_len={opts.seq_len} epochs={opts.epochs} lr={opts.lr} "
        f"global_batch={opts.global_batch} instances "
        f"({opts.global_batch * opts.seq_len:,} tokens) "
        f"micro_batch_instances={opts.micro_batch_instances}"
    )
    print(f"  world_size={world_size} (hypothetical) -> per-rank instances/step={plan['per_rank']}")
    train_module_config = build_train_module_config(opts, world_size)
    ac_config = train_module_config.ac_config
    print(
        f"  train_module: dp_name={train_module_config.dp_config.name} "
        f"shard_degree={train_module_config.dp_config.shard_degree} "
        f"AC={ac_config.mode if ac_config is not None else 'none'} "
        f"rank_microbatch_size={train_module_config.rank_microbatch_size:,} tokens"
    )
    if plan["curriculum"] is not None:
        c = plan["curriculum"]
        print(
            f"  mask-mix curriculum: p {opts.mix_start_p} -> {opts.mix_end_p} over "
            f"mix_total_forwards={c['mix_total_forwards']} | steps={c['steps']} "
            f"forwards_per_rank={c['forwards_per_rank']} "
            f"predicted final p_standard={c['final_p']:.4f}  [HARD-CHECK PASSED]"
        )
    else:
        print(f"  mask-mix: none (variant={opts.variant})")
    print(f"  data_loader: work_dir={plan['data_loader_config'].work_dir}")
    base = opts.base_checkpoint or DEFAULT_BASE_CHECKPOINT.get(opts.model_scale)
    print(f"  base_checkpoint={base or 'MISSING (required for a real run at this scale)'}")
    announce_wandb(opts)
    prov = build_provenance(opts, world_size)
    print(f"  provenance preview: {json.dumps(prov, indent=2, default=str)}")


def _tolerant_base_load(base_checkpoint: str, model, save_folder: str) -> None:
    """Load ``base_checkpoint`` into ``model`` allowing keys the base lacks (the ffnmoe router/gains,
    the softtoken projector), and REFUSE if anything else is missing -- a missing backbone key
    would otherwise silently train from init.

    torch's DCP loader rejects a state dict that asks for keys the checkpoint does not have
    BEFORE any ``strict`` option applies, so the model state dict is FILTERED to the checkpoint's
    keys, loaded, and set back non-strictly; the filtered-out (new) parameters keep their init.

    Resumes are unaffected: if the save folder already holds a step checkpoint, ``fit()`` loads
    it afterwards (it carries the new keys), overriding this base load.
    """
    import torch.distributed.checkpoint as dist_cp
    import torch.distributed.checkpoint.state_dict as dist_cp_sd

    from olmo_core.distributed.checkpoint import RemoteFileSystemReader, _prepare_state_dict
    from olmo_core.utils import gc_cuda

    reader = RemoteFileSystemReader(base_checkpoint)
    ckpt_keys = set(reader.read_metadata().state_dict_metadata)
    state_dict = _prepare_state_dict(model, None)
    model_sd = state_dict["model"]
    missing = sorted(k for k in model_sd if f"model.{k}" not in ckpt_keys)
    allowed = ("._nffn_router.", "._nffn_gain", "pooled_projector.", "._pooled_projector")
    bad = [k for k in missing if not any(a in k for a in allowed)]
    if bad:
        raise SystemExit(
            f"[ctc-suite] base {base_checkpoint} lacks {len(bad)} backbone keys (first: {bad[:3]}); "
            "refusing to train from init"
        )
    filtered = {"model": {k: v for k, v in model_sd.items() if k not in set(missing)}}
    dist_cp.state_dict_loader.load(filtered, checkpoint_id=base_checkpoint, storage_reader=reader)
    dist_cp_sd.set_model_state_dict(
        model, filtered["model"], options=dist_cp_sd.StateDictOptions(strict=False)
    )
    gc_cuda()
    print(
        f"[ctc-suite] tolerant base load from {base_checkpoint}: {len(filtered['model'])} keys "
        f"loaded, {len(missing)} new keys kept at init (e.g. {missing[:2]})",
        flush=True,
    )


def build_and_fit(opts: argparse.Namespace) -> None:
    """Build the model / data loader / trainer and run the fit (real training entrypoint).

    :param opts: Parsed CLI options.
    """
    save_folder = opts.save_folder or f"{SAVE_ROOT}/{opts.run_name}"
    base_checkpoint = opts.base_checkpoint or DEFAULT_BASE_CHECKPOINT.get(opts.model_scale)
    if not base_checkpoint:
        raise SystemExit(
            f"--base-checkpoint required for --model-scale {opts.model_scale} (no default base "
            "converted yet; run the marker audit on it first, see the plan §4)."
        )
    # Guard the distcp marker: staging a base via cp can drop the hidden .metadata, after which
    # olmo-core silently trains FROM SCRATCH (CE ~ ln(vocab)) instead of from the base.
    if not os.path.exists(os.path.join(base_checkpoint, ".metadata")):
        raise SystemExit(
            f"base checkpoint {base_checkpoint} missing .metadata -> would silently train from "
            "scratch. Point at the model_and_optim SUBDIR of the converted base."
        )

    world_size = max(1, get_world_size())
    plan = resolve_plan(opts, world_size)
    print(
        f"[ctc-suite] task={opts.task} variant={opts.variant} scale={opts.model_scale} "
        f"n_examples={plan['n_examples']} epochs={opts.epochs} seq_len={opts.seq_len} "
        f"global_batch={opts.global_batch} world_size={world_size} "
        f"curriculum={plan['curriculum']}",
        flush=True,
    )
    announce_wandb(opts)

    if get_rank() == 0:
        os.makedirs(save_folder, exist_ok=True)
        with open(os.path.join(save_folder, "provenance.json"), "w") as f:
            json.dump(build_provenance(opts, world_size), f, indent=2, default=str)

    train_module_config = build_train_module_config(opts, world_size)

    trainer_config = (
        TrainerConfig(
            save_folder=save_folder,
            save_overwrite=True,
            # ffnmoe / softtoken add NEW parameters (router+gains / projector) that the plain
            # base does not have; the trainer's own load is strict, so those arms load the base
            # themselves below with strict=False (new keys keep their init: router -> full rung,
            # projector -> identity). load_path stays None for them so fit() does not re-load.
            load_path=None if opts.variant in ("ffnmoe", "softtoken") else base_checkpoint,
            # ...and "always" would then demand a checkpoint that no longer exists: those arms
            # resume from the save folder only if a step checkpoint is there.
            load_strategy=(
                LoadStrategy.if_available
                if opts.variant in ("ffnmoe", "softtoken")
                else LoadStrategy.always
            ),
            load_trainer_state=False,
            load_optim_state=False,
            metrics_collect_interval=1,
            cancel_check_interval=1,
            max_duration=Duration.epochs(opts.epochs),
            hard_stop=Duration.steps(opts.max_steps) if opts.max_steps else None,
            # no_checkpoints=True would ALSO skip the base-checkpoint LOAD block (trainer.fit
            # gates loading on `not no_checkpoints`) -> silent train-from-scratch. Keep False;
            # mid-run saving is controlled by the explicit CheckpointerCallback below, and the
            # --save-checkpoint block after fit() writes the model-only checkpoint for eval.
            no_checkpoints=False,
        )
        .with_callback("gpu_monitor", GPUMemoryMonitorCallback())
        .with_callback("config_saver", ConfigSaverCallback())
        # Register the checkpointer EXPLICITLY. Leaving it out does not disable it -- Trainer does
        # `callbacks.setdefault("checkpointer", CheckpointerCallback())` (trainer.py), so omitting it
        # silently activates the DEFAULT save_interval=250. At 4B that wrote a full
        # model+optim+train-state checkpoint (~55G) every 250 steps: a 750-step run produced 165G of
        # step*/ dirs that nothing reads, since eval loads the model-only `model_and_optim` saved
        # after fit(). Default to no mid-run checkpoints; pass --save-interval N to get resume
        # points back for a long run on a preemptible queue.
        # ``enabled=False`` when --no-final-checkpoint: CheckpointerCallback.post_train()
        # otherwise saves a FULL model+optim+train-state checkpoint unconditionally at the end —
        # 82G and ~25 min over lambda's ~60MB/s NFS, for state nothing downstream reads (eval and
        # rebasing both use the model-only export from the --save-checkpoint block after fit()).
        # The base-checkpoint LOAD is unaffected: trainer.fit gates it on ``no_checkpoints``, not
        # on this callback (see the note above).
        .with_callback(
            "checkpointer",
            CheckpointerCallback(
                save_interval=opts.save_interval, enabled=not opts.no_final_checkpoint
            ),
        )
    )
    if opts.wandb:
        trainer_config = trainer_config.with_callback(
            "wandb",
            WandBCallback(
                name=opts.run_name,
                group=opts.wandb_group or opts.run_name,
                entity=opts.wandb_entity,
                project=WANDB_PROJECT,
                enabled=True,
                cancel_check_interval=10,
                config={
                    "experiment": "ctc_suite",
                    "task": opts.task,
                    "variant": opts.variant,
                    "model_scale": opts.model_scale,
                    "seq_len": opts.seq_len,
                    "epochs": opts.epochs,
                    "lr": opts.lr,
                    "global_batch": opts.global_batch,
                    "seed": opts.seed,
                    "n_examples": plan["n_examples"],
                    "mix_start_p": opts.mix_start_p if opts.variant == "chunked-mix" else None,
                    "mix_end_p": opts.mix_end_p if opts.variant == "chunked-mix" else None,
                    "base_checkpoint": base_checkpoint,
                    "data": opts.data,
                },
            ),
        )

    seed_all(12536 + opts.seed)
    model = plan["model_config"].build(init_device="meta")
    ids = RESERVED_IDS[opts.model_family]
    # Forwards per rank over the whole run = steps x grad-accum: the horizon the ffnmoe schedules
    # (target / exploration / layer curriculum) and the softtoken mixing curriculum anneal over.
    gbs_examples = max(1, opts.global_batch)
    if opts.pack:
        # Packed rows hold many examples: the step count is tokens / (rows x seq_len), not
        # examples / rows. Using the example count here over-estimated the schedule horizon
        # ~20x on the first Qwen3.5 FFN runs (target still 0.84 at the last step).
        n_tok = int(plan["meta"].get("num_tokens") or plan["meta"].get("total_tokens") or 0)
        steps_total = max(1, -(-n_tok * opts.epochs // (gbs_examples * opts.seq_len))) if n_tok else -(-plan["n_examples"] * opts.epochs // gbs_examples)
    else:
        steps_total = -(-plan["n_examples"] * opts.epochs // gbs_examples)
    accum = max(1, opts.global_batch // (world_size * opts.micro_batch_instances))
    total_calls = max(1, steps_total * accum)
    if opts.variant == "ffnmoe":
        model.enable_nested_ffn_moe(
            start_layer=opts.ffn_moe_start_layer,
            divisors=[float(x) for x in opts.ffn_moe_divisors.split(",")],
            include_null=not opts.ffn_moe_no_null,
            target_cost=opts.ffn_moe_target,
            budget_weight=opts.ffn_moe_budget_weight,
            hinge_power=opts.ffn_moe_hinge_power,
            target_anneal_calls=int(total_calls * opts.ffn_moe_target_anneal_frac),
            explore_prob=opts.ffn_moe_explore,
            explore_anneal_calls=int(total_calls * opts.ffn_moe_explore_anneal_frac),
            recon_frac=opts.ffn_moe_recon_frac,
            recon_weight=opts.ffn_moe_recon_weight,
            entropy_weight=opts.ffn_moe_entropy_weight,
            seed=opts.seed,
            layer_curriculum_calls=int(total_calls * opts.ffn_moe_layer_curriculum_frac),
            width_multiple=opts.ffn_moe_width_multiple,
        )
        print(
            f"[ctc-suite] ffnmoe: routed from layer {opts.ffn_moe_start_layer}, rungs="
            f"{model._nested_ffn_moe['widths']}, target {opts.ffn_moe_target} annealed over "
            f"{int(total_calls * opts.ffn_moe_target_anneal_frac)}/{total_calls} calls",
            flush=True,
        )
    if opts.variant == "softtoken":
        model.enable_pooled_soft_tokens(
            ids.doc_start,
            ids.doc_end,
            ids.eos,
            placeholder_id=ids.landmark,
            keep_prob=opts.st_keep_prob,
            keep_seed=opts.seed,
            aux_match_weight=opts.st_aux_weight,
            detach_soft_kv=not opts.st_no_detach_soft_kv,
            distill_prob=opts.st_distill_prob,
            distill_weight=opts.st_distill_weight,
        )
        print(
            f"[ctc-suite] softtoken: detach={not opts.st_no_detach_soft_kv} "
            f"distill_prob={opts.st_distill_prob} keep_mode={opts.st_keep_mode} "
            f"n_random={opts.st_n_random_range or opts.st_n_random} keep_frac={opts.st_keep_frac} "
            f"gold_blind={opts.st_gold_blind} keep_prob={opts.st_keep_prob}",
            flush=True,
        )
    train_module = train_module_config.build(model)
    if opts.variant == "softtoken" and not opts.st_gold_blind:
        from olmo_core.nn.attention.pooled_doc_kv import (
            install_pooled_doc_keep,
            make_fingerprint_keep_docs_fn,
        )

        sidecar = opts.st_gold_sidecar or os.path.join(opts.data, "gold_fingerprints.json")
        with open(sidecar) as f:
            gold_table = json.load(f)
        keep_fn = make_fingerprint_keep_docs_fn(
            gold_table,
            doc_start_id=ids.doc_start,
            doc_end_id=ids.doc_end,
            eos_id=ids.eos,
            n_random=opts.st_n_random,
            n_random_range=(
                tuple(int(x) for x in opts.st_n_random_range.split(","))
                if opts.st_n_random_range
                else None
            ),
            n_random_frac=opts.st_keep_frac,
            mode=opts.st_keep_mode,
            n_gold=opts.st_n_gold,
            seed=opts.seed,
            mix_start_p=opts.st_mix_start_p,
            mix_end_p=opts.st_mix_end_p,
            mix_total_calls=int(total_calls * opts.st_mix_anneal_frac),
        )
        holder = install_pooled_doc_keep(train_module.model, keep_fn)
        if holder.n_attached == 0:
            raise SystemExit("[ctc-suite] softtoken: gold keep hook attached to nothing")
        print(
            f"[ctc-suite] softtoken: gold keep hook on {holder.n_attached} module(s), "
            f"{len(gold_table)} fingerprints from {sidecar}",
            flush=True,
        )
    if opts.variant == "ffnmoe":
        trainer_config = trainer_config.with_callback(
            "ffn_moe", NestedFFNMoECallback(calls_per_step=accum)
        )
    # Method-aware training FLOPs for every arm (records/flop-scaling-ffn-kv-plan.md §5).
    trainer_config = trainer_config.with_callback(
        "flop_meter", FlopMeterCallback(seq_len=opts.seq_len, pad_id=ids.eos)  # rows are padded with EOS (see pad_token_id above)
    )
    if opts.variant == "pooledkv" and opts.pooled_gold_sidecar:
        # Gold-aware keep set: a forward pre-hook resolves each row's gold docs by content
        # fingerprint and marks gold + --pooled-n-random random negatives as keeping real KV.
        from olmo_core.nn.attention.pooled_doc_kv import (
            install_pooled_doc_keep,
            make_fingerprint_keep_docs_fn,
        )

        ids = RESERVED_IDS[opts.model_family]
        with open(opts.pooled_gold_sidecar) as f:
            gold_table = json.load(f)
        keep_fn = make_fingerprint_keep_docs_fn(
            gold_table,
            doc_start_id=ids.doc_start,
            doc_end_id=ids.doc_end,
            eos_id=ids.eos,
            n_random=opts.pooled_n_random,
            mode=opts.pooled_keep_mode,
            seed=opts.seed,
            n_gold=opts.pooled_n_gold,
        )
        holder = install_pooled_doc_keep(train_module.model, keep_fn)
        if holder.n_attached == 0:
            raise SystemExit(
                "[ctc-suite] --pooled-gold-sidecar given but no PooledDocKVAttention layers were "
                "found on the model (variant/config mismatch?)."
            )
        print(
            f"[ctc-suite] pooledkv: gold keep-set hook installed on {holder.n_attached} layers "
            f"({len(gold_table)} fingerprints, mode={opts.pooled_keep_mode}, "
            f"n_random={opts.pooled_n_random})",
            flush=True,
        )
    source = plan["instance_source_config"].build(plan["data_loader_config"].work_dir)
    data_loader = plan["data_loader_config"].build(
        source, dp_process_group=train_module.dp_process_group
    )
    trainer = trainer_config.build(train_module, data_loader)
    if opts.variant in ("ffnmoe", "softtoken"):
        _tolerant_base_load(base_checkpoint, train_module.model, save_folder)
    trainer.fit()

    # Save a model-only checkpoint in the eval loader's expected layout (config.json +
    # model_and_optim/) so the docchunk eval scripts load it directly.
    if opts.save_checkpoint:
        from olmo_core.distributed.checkpoint import save_model_and_optim_state

        save_model_and_optim_state(
            f"{save_folder}/model_and_optim", train_module.model, save_overwrite=True
        )
        if get_rank() == 0:
            model_dict = plan["model_config"].as_config_dict()
            if opts.attn_backend == "torch":
                # The soft-token arm trains with the SDPA backend (the flash_2 path is ~20x slower
                # on its compacted rows), but SDPA has no KV-cached generation: record flash_2 in
                # the export so the evaluators build the (identical-weight) model for fast decode.
                def _swap(o):
                    if isinstance(o, dict):
                        for k, v in o.items():
                            if k == "backend" and v == "torch":
                                o[k] = "flash_2"
                            else:
                                _swap(v)
                    elif isinstance(o, list):
                        for v in o:
                            _swap(v)
                _swap(model_dict)
            experiment = {
                "model": model_dict,
                "dataset": {"tokenizer": plan["tokenizer_config"].as_config_dict()},
                # Recorded so the eval scores with the routing the run trained with (the
                # evaluator reads this block and enables the router before loading).
                "ffn_moe": (
                    {
                        "start_layer": opts.ffn_moe_start_layer,
                        "divisors": opts.ffn_moe_divisors,
                        "include_null": not opts.ffn_moe_no_null,
                        "width_multiple": opts.ffn_moe_width_multiple,
                    }
                    if opts.variant == "ffnmoe"
                    else None
                ),
                "softtoken": (
                    {"n_random": opts.st_n_random, "n_random_range": opts.st_n_random_range,
                     "keep_frac": opts.st_keep_frac, "keep_prob": opts.st_keep_prob,
                     "keep_mode": opts.st_keep_mode, "gold_blind": opts.st_gold_blind}
                    if opts.variant == "softtoken"
                    else None
                ),
            }
            with open(f"{save_folder}/config.json", "w") as f:
                json.dump(experiment, f)
            print(f"[ctc-suite] saved model-only checkpoint -> {save_folder}", flush=True)


def parse_args() -> argparse.Namespace:
    """Build the CLI parser and parse arguments.

    :returns: Parsed options (with the private ``_mix_total_forwards`` slot pre-initialized).
    """
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--task", required=True, help="suite task name (provenance + wandb; see plan)")
    ap.add_argument(
        "--data",
        required=True,
        help="shard dir from convert_unified_to_document_landmark.py --marker-set qwen3_5 "
        "(token_ids_part_*.npy + labels_mask_*.npy + metadata.json)",
    )
    ap.add_argument(
        "--model-family",
        choices=["auto", *sorted(MODEL_FACTORIES)],
        default="auto",
        help="auto (default) = detect from the shard's metadata marker_set; or force qwen3 "
        "(plain dense/causal) / qwen3_5 (GDN+attn hybrid). Determines the model factory, marker "
        "ids, tokenizer, and embedding size.",
    )
    ap.add_argument(
        "--model-scale",
        default="0.8b",
        help="scale key within the resolved family: qwen3_5={0.8b,4b,9b}; qwen3={0.6b,1.7b,4b,8b}",
    )
    ap.add_argument(
        "--variant",
        choices=["full", "chunked", "chunked-mix", "sparselandmark", "pooledkv", "ffnmoe", "softtoken"],
        required=True,
        help="full = plain causal (no document_chunk_attention); chunked = pure document-chunked "
        "mask; chunked-mix = chunked + curriculum mask mixing (mix_start_p -> mix_end_p); "
        "sparselandmark = AttentionType.sparse_landmark on the full-attn blocks + landmark-packed "
        "data; pooledkv = AttentionType.pooled_doc_kv on the full-attn blocks (train-time "
        "per-document KV pooling: gold + a few random docs keep real KV, the rest collapse to a "
        "mean-pooled slot; inference is ordinary full attention); ffnmoe = nested-width FFN "
        "router (olmo_core.nn.nested_ffn_moe: per-token choice of a prefix-sliced FFN width or "
        "null under a budget hinge); softtoken = pooled-doc soft tokens (the whole stack runs on "
        "a compacted sequence: gold + n_random docs keep real tokens, every other doc collapses "
        "to ONE projected soft token; inference is ordinary full attention). Both add small new "
        "parameters (router+gains / projector) that are initialized at load, so they run from "
        "the plain base -- no baked base needed (tolerant load, see build_and_fit).",
    )
    # ---- ffnmoe (records/flop-scaling-ffn-kv-plan.md; recipe = ffnmoe/README.md v10/v12) ----
    ap.add_argument("--ffn-moe-start-layer", type=int, default=12, help="first routed layer (0 = all)")
    ap.add_argument("--ffn-moe-divisors", default="1,16,64,256,1024,9728", help="rung ladder")
    ap.add_argument("--ffn-moe-width-multiple", type=int, default=1, help="1 allows a width-1 rung")
    ap.add_argument("--ffn-moe-no-null", action="store_true")
    ap.add_argument("--ffn-moe-target", type=float, default=0.01, help="budget: mean FFN cost on routed layers")
    ap.add_argument("--ffn-moe-budget-weight", type=float, default=1.0)
    ap.add_argument("--ffn-moe-hinge-power", type=int, default=1)
    ap.add_argument("--ffn-moe-target-anneal-frac", type=float, default=0.3, help="0 = hinge active from step 0 (stage 2 of the two-stage recipe)")
    ap.add_argument("--ffn-moe-explore", type=float, default=0.1)
    ap.add_argument("--ffn-moe-explore-anneal-frac", type=float, default=0.3)
    ap.add_argument("--ffn-moe-recon-frac", type=float, default=0.02)
    ap.add_argument("--ffn-moe-recon-weight", type=float, default=0.0)
    ap.add_argument("--ffn-moe-entropy-weight", type=float, default=0.0)
    ap.add_argument("--ffn-moe-layer-curriculum-frac", type=float, default=0.0)
    ap.add_argument("--router-lr", type=float, default=1e-3, help="ffnmoe: router/gain LR (backbone uses --lr)")
    # ---- softtoken (records/pooled-doc-kv-handoff.md; v20 = --st-n-random 128, v22 = 256) ----
    ap.add_argument("--st-n-random", type=int, default=128, help="random non-gold docs kept real per example")
    ap.add_argument("--st-n-random-range", default="", help="lo,hi log-uniform breadth per call (overrides --st-n-random)")
    ap.add_argument(
        "--st-keep-frac",
        type=float,
        default=None,
        help="keep a FIXED FRACTION of each example's non-gold docs real (gold always kept); "
        "context-length invariant -- the FLOP-scaling study's KV arms (overrides n-random/range)",
    )
    ap.add_argument("--st-keep-mode", default="gold_plus_random")
    ap.add_argument("--st-n-gold", type=int, default=0)
    ap.add_argument("--st-keep-prob", type=float, default=0.1, help="gold-blind fallback keep prob (no sidecar / --st-gold-blind)")
    ap.add_argument("--st-gold-blind", action="store_true", help="ignore the gold sidecar: keep docs by --st-keep-prob only (oolong)")
    ap.add_argument("--st-gold-sidecar", default=None, help="default <data>/gold_fingerprints.json")
    ap.add_argument("--st-no-detach-soft-kv", action="store_true", help="the winning recipe DETACHES; this is the ablation")
    ap.add_argument("--st-distill-prob", type=float, default=0.0)
    ap.add_argument("--st-distill-weight", type=float, default=1.0)
    ap.add_argument("--st-aux-weight", type=float, default=0.0)
    ap.add_argument("--st-mix-start-p", type=float, default=0.0)
    ap.add_argument("--st-mix-end-p", type=float, default=0.0)
    ap.add_argument("--st-mix-anneal-frac", type=float, default=1.0)
    ap.add_argument(
        "--pooled-gold-sidecar",
        default=None,
        help="pooledkv only: gold sidecar JSON ({content_fingerprint: gold ids or pairs}, e.g. "
        "from build_gold_sidecar_from_shard.py). Installs the gold-aware keep-set hook; without "
        "it the keep set is a gold-blind random --pooled-keep-prob fraction (control arm).",
    )
    ap.add_argument(
        "--pooled-n-random",
        type=int,
        default=2,
        help="pooledkv + sidecar: random non-gold docs kept real per example",
    )
    ap.add_argument(
        "--pooled-keep-mode",
        default="gold_plus_random",
        help="pooledkv + sidecar: select_keep_docs policy (gold_plus_random / gold_subsample / "
        "random_only / random_nongold / gold_pair / gold_halves)",
    )
    ap.add_argument(
        "--pooled-n-gold",
        type=int,
        default=0,
        help="pooledkv + sidecar: n_gold for the gold_subsample / gold_halves policies",
    )
    ap.add_argument(
        "--pooled-keep-prob",
        type=float,
        default=0.1,
        help="pooledkv without sidecar: per-doc probability of keeping real KV (seeded hash)",
    )
    ap.add_argument(
        "--pooled-no-len-bias",
        action="store_true",
        help="pooledkv: drop the +log(doc_len) slot logit bias (ablation; the biased form is the "
        "principled 'L copies of the mean KV entry' equivalence)",
    )
    ap.add_argument(
        "--seq-len", type=int, default=40960, help="fits the 32k rung + prompt/CoT overhead"
    )
    ap.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument(
        "--global-batch",
        type=int,
        default=8,
        help="instances per optimizer step across all ranks (global batch tokens = this * seq-len)",
    )
    ap.add_argument("--save-folder", default=None, help=f"default {SAVE_ROOT}/<run-name>")
    ap.add_argument("--run-name", default="ctc-suite-smoke")
    ap.add_argument("--wandb-group", default=None, help="default: the run name")
    ap.add_argument("--wandb-entity", default=None)
    ap.add_argument(
        "--no-wandb",
        dest="wandb",
        action="store_false",
        help="disable wandb (ON by default; needs WANDB_API_KEY)",
    )
    ap.add_argument(
        "--base-checkpoint",
        default=None,
        help="model_and_optim distcp subdir of the converted per-scale base (default only for "
        f"0.8b: {DEFAULT_BASE_CHECKPOINT['0.8b']})",
    )
    ap.add_argument(
        "--work-dir",
        default=None,
        help=f"data-loader cache dir (default {WORK_DIR}); use node-local /data for speed",
    )
    ap.add_argument(
        "--micro-batch-instances",
        type=int,
        default=1,
        help="instances per rank per FORWARD (rank_microbatch_size = this * seq-len). Pure "
        "throughput knob (grad accumulation is exact); cannot exceed --global-batch / world_size",
    )
    ap.add_argument("--num-workers", type=int, default=2, help="dataloader workers per rank")
    ap.add_argument("--max-steps", type=int, default=0, help="hard-stop after N steps (0 = full)")
    ap.add_argument(
        "--vocab-size",
        type=int,
        default=None,
        help="embedding-matrix size; default is the resolved family's base size "
        "(qwen3_5=248320, qwen3=151936). Override only to match a non-standard base.",
    )
    ap.add_argument(
        "--activation-checkpointing",
        choices=["auto", "full", "budget", "none"],
        default="auto",
        help="activation checkpointing. auto (default) = full for 4b/9b, none for 0.8b. "
        "USE 'budget' for context-parallel (--cp-degree) runs: full-block AC is incompatible with "
        "compile+CP (recompute-metadata mismatch at the dry-run); budget is the proven CP recipe.",
    )
    ap.add_argument(
        "--ac-budget",
        type=float,
        default=0.7,
        help="activation-memory budget for --activation-checkpointing budget (proven CP value 0.7; "
        "lower = more checkpointing / less memory if a long-context run OOMs).",
    )
    ap.add_argument(
        "--shard-degree",
        type=int,
        default=0,
        help="FSDP shard_degree. 0 (default/auto) = 1 for 0.8b (proven working), world_size for "
        "4b/9b (full parameter sharding, needed to fit seq_len=40960 on an 80GB H100). Override "
        "to force a specific value. Under CP this shards over the dp dimension (world_size/cp).",
    )
    ap.add_argument(
        "--cp-degree",
        type=int,
        default=0,
        help="context-parallel (Ulysses) degree for long-context arms. 0/1 = no CP (default). "
        "Shards the sequence over cp_degree GPUs; bounded by n_kv_heads (dense qwen3=8, hybrid "
        "qwen3_5=4). world_size must be divisible by it; the dp/FSDP dim becomes world_size/cp.",
    )
    ap.add_argument(
        "--rope-yarn-factor",
        type=float,
        default=0.0,
        help="YaRN RoPE context-extension factor (0/1 = off). E.g. 8 extends Qwen3's native 32k "
        "to 256k; 2 for 64k. Required whenever --seq-len exceeds the base's native context.",
    )
    ap.add_argument(
        "--rope-old-context",
        type=int,
        default=32768,
        help="base native context length for YaRN scaling (Qwen3 = 32768).",
    )
    ap.add_argument(
        "--rope-theta",
        type=float,
        default=0.0,
        help="override base rope_theta for NTK-style context extension (0 = factory default). "
        "For Qwen3-4B (native 32k) at 256k use ~8e6. Cleaner + more robust than --rope-yarn-factor "
        "for large extensions; do not combine with --rope-yarn-factor.",
    )
    ap.add_argument(
        "--pack",
        action="store_true",
        help="bin-pack whole examples into seq_len windows (PackingInstanceSource) instead of one "
        "padded example per instance. Required for a mixed-length (e.g. 8k..256k) shard so short "
        "examples don't waste compute padded up to the max seq_len. Composes with CP.",
    )
    ap.add_argument(
        "--attn-backend",
        choices=["flash_2", "torch"],
        default="flash_2",
        help="attention backend for the full-attention blocks (default flash_2; use torch/SDPA "
        "on a cluster where flash-attn is not importable or not verified for the GPU arch)",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=0,
        help="run-to-run seed offset (shifts data order + torch seed) for noise-floor runs",
    )
    ap.add_argument("--mix-start-p", type=float, default=0.80)
    ap.add_argument("--mix-end-p", type=float, default=0.0)
    ap.add_argument("--mix-seed", type=int, default=42)
    ap.add_argument("--mix-log-interval", type=int, default=5)
    ap.add_argument(
        "--save-checkpoint",
        action="store_true",
        help="after fit, save model-only checkpoint (config.json + model_and_optim) for eval",
    )
    ap.add_argument(
        "--save-interval",
        type=int,
        default=None,
        help="steps between mid-run model+optim checkpoints. Default None = none written (eval only "
        "needs the post-fit model-only save, and at 4B each mid-run checkpoint costs ~55G). Set an "
        "integer to get resume points for a long run on a preemptible queue.",
    )
    ap.add_argument(
        "--no-final-checkpoint",
        action="store_true",
        help="skip the checkpointer's end-of-training FULL (model+optim+train-state) save; the "
        "post-fit model-only export from --save-checkpoint still runs. Use on lambda, where the "
        "full save costs ~82G of user quota and ~25 min of NFS writes that nothing reads. "
        "Incompatible with resuming the run later.",
    )
    ap.add_argument("--no-compile", dest="compile", action="store_false")
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="build configs (model, data loader, curriculum math) WITHOUT CUDA/distributed and "
        "print the resolved plan; trains nothing, writes nothing",
    )
    ap.add_argument(
        "--dry-run-world-size",
        type=int,
        default=8,
        help="hypothetical DP world size for --dry-run curriculum math",
    )
    ap.add_argument(
        "--dry-run-n-examples",
        type=int,
        default=0,
        help="--dry-run only: pretend the shard has this many examples (0 = use metadata)",
    )
    opts = ap.parse_args()
    opts._mix_total_forwards = 0
    # Resolve the model family (explicit or auto from the shard) and rebind opts.model_family to
    # the concrete family, so every downstream function keys off it deterministically.
    opts.model_family = resolve_family(opts)
    if opts.model_scale not in MODEL_FACTORIES[opts.model_family]:
        ap.error(
            f"--model-scale {opts.model_scale!r} is not defined for family "
            f"{opts.model_family!r}; choose from {sorted(MODEL_FACTORIES[opts.model_family])}"
        )
    if opts.vocab_size is None:
        opts.vocab_size = FAMILY_VOCAB_SIZE[opts.model_family]
    if not opts.dry_run and opts.dry_run_n_examples:
        ap.error("--dry-run-n-examples is only valid with --dry-run")
    if opts.variant == "chunked-mix" and opts.compile:
        # The python-seeded mix coin + counter are not torch.compile-capturable.
        print("[ctc-suite] chunked-mix: forcing --no-compile (mix coin is not compilable)")
        opts.compile = False
    if opts.variant == "sparselandmark" and opts.compile:
        # SparseLandmarkAttention is a Triton custom-autograd Function; keep compile off (same
        # posture as every sparse-landmark CPT/SFT script) rather than risk AC+compile metadata
        # mismatches mid-sweep.
        print("[ctc-suite] sparselandmark: forcing --no-compile (triton custom-autograd mixer)")
        opts.compile = False
    if opts.variant == "pooledkv":
        if opts.pack:
            # The keep-set fingerprint + role reconstruction assume ONE example per padded row
            # (everything after the first EOS is PAD); packed rows would fingerprint-miss every
            # example and silently keep all docs real.
            ap.error("--variant pooledkv requires the padded (no --pack) data path")
        if opts.compile:
            # Data-dependent shapes (per-batch doc count) + host-side keep-set resolution.
            print("[ctc-suite] pooledkv: forcing --no-compile (data-dependent pooled-KV shapes)")
            opts.compile = False
    if opts.variant != "pooledkv" and (opts.pooled_gold_sidecar or opts.pooled_no_len_bias):
        ap.error("--pooled-* options are only valid with --variant pooledkv")
    if opts.variant == "softtoken":
        if opts.pack:
            ap.error("--variant softtoken requires the padded (no --pack) data path (per-row fingerprints)")
        if opts.compile:
            print("[ctc-suite] softtoken: forcing --no-compile (data-dependent compacted shapes)")
            opts.compile = False
    if opts.variant == "ffnmoe" and opts.compile:
        print("[ctc-suite] ffnmoe: forcing --no-compile (data-dependent per-rung GEMM shapes)")
        opts.compile = False
    return opts


def main() -> None:
    """CLI entrypoint: dry-run without distributed init, otherwise train under torchrun."""
    opts = parse_args()
    if opts.dry_run:
        dry_run(opts)
        return
    prepare_training_environment()
    try:
        build_and_fit(opts)
    finally:
        teardown_training_environment()


if __name__ == "__main__":
    main()
