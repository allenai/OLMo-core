"""
Tiny CPU models for the five attention variants under inference-consistency test, plus the token
layouts each one needs to see its own structure.

Each variant declares what agreement it is *entitled to*. That is the substance of this module: the
dense path computes one function and must reproduce it exactly whether it prefills or decodes, while
the structured variants deliberately serve different semantics at decode time than a single
teacher-forced forward computes -- landmark decode modes treat generated tokens as one trailing local
block and apply hard top-k retrieval that the batched forward never applies. Asserting equality
across all five would either fail on correct code or force the tolerance so wide it stops catching
the dense regressions it exists for, so :class:`VariantSpec` carries an ``expect`` field and the
tests branch on it.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

import torch

from olmo_core.config import DType
from olmo_core.generate.generation_module import TransformerGenerationModule
from olmo_core.generate.generation_module.config import GenerationConfig
from olmo_core.nn.attention import AttentionBackendName, AttentionType
from olmo_core.nn.transformer import TransformerConfig

__all__ = [
    "Ids",
    "IDS",
    "VariantSpec",
    "VARIANTS",
    "get_variant",
    "document_prompt",
    "plain_prompt",
]


@dataclass(frozen=True)
class Ids:
    """Reserved token ids for the structured layouts. Kept well clear of the content range."""

    doc_start: int = 900
    doc_end: int = 901
    summary: int = 902
    eos: int = 903
    pad: int = 904
    landmark: int = 905


IDS = Ids()

VOCAB_SIZE = 1024
D_MODEL = 128
N_HEADS = 4
N_LAYERS = 2
# block_size = mem_freq + num_landmarks = 16. The fused landmark kernel tiles by block_size and
# tl.dot needs tile dims >= 16, so mem_freq < 15 is rejected at build time.
MEM_FREQ = 15
N_SUMMARY_TOKENS = 3

# Content tokens live below the reserved block, so a forced gold continuation can never collide with
# EOS (which would stop the decode loop early and misalign the trace).
CONTENT_LO, CONTENT_HI = 10, 800


def plain_prompt(n_tokens: int, *, seed: int = 0) -> torch.Tensor:
    """
    An unstructured content-token prompt, for variants that read no layout from the token stream.

    :param n_tokens: Prompt length.
    :param seed: RNG seed.

    :returns: Token ids of shape ``(1, n_tokens)``.
    """
    g = torch.Generator().manual_seed(seed)
    return torch.randint(CONTENT_LO, CONTENT_HI, (1, n_tokens), generator=g)


def document_prompt(
    *,
    n_docs: int = 3,
    doc_len: int = 9,
    instruction_len: int = 5,
    query_len: int = 4,
    n_summary_tokens: int = 0,
    seed: int = 0,
) -> torch.Tensor:
    """
    Build ``[instruction][<doc_start> doc <doc_end> (<summ> * k)] * n_docs [query]``.

    This is the layout both :class:`~olmo_core.nn.attention.DocumentChunkedAttention` and
    :class:`~olmo_core.nn.attention.summary_token.SummaryTokenAttention` reconstruct their masks
    from -- neither takes the structure as an argument; both derive it from the boundary tokens
    inside :meth:`~olmo_core.nn.transformer.Transformer.forward`. That derivation running identically
    on the prefill stream and on the single-token decode stream is a large part of what these tests
    check.

    :param n_docs: Number of context documents.
    :param doc_len: Content tokens per document.
    :param instruction_len: Length of the leading instruction span.
    :param query_len: Length of the trailing query span.
    :param n_summary_tokens: Summary tokens appended after each document (0 for the non-summary
        variants).
    :param seed: RNG seed for the content tokens.

    :returns: Token ids of shape ``(1, L)``.
    """
    g = torch.Generator().manual_seed(seed)

    def rand(n: int) -> List[int]:
        return torch.randint(CONTENT_LO, CONTENT_HI, (n,), generator=g).tolist()

    ids: List[int] = rand(instruction_len)
    for _ in range(n_docs):
        ids += [IDS.doc_start] + rand(doc_len) + [IDS.doc_end]
        ids += [IDS.summary] * n_summary_tokens
    ids += rand(query_len)
    return torch.tensor([ids])


def gold_continuation(n: int, *, seed: int = 1234) -> List[int]:
    """
    A gold continuation to force through the decode loop.

    :param n: Number of tokens.
    :param seed: RNG seed.

    :returns: Token ids, guaranteed clear of the reserved ids.
    """
    g = torch.Generator().manual_seed(seed)
    return torch.randint(CONTENT_LO, CONTENT_HI, (n,), generator=g).tolist()


def _base_kwargs(**overrides) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = dict(
        d_model=D_MODEL,
        vocab_size=VOCAB_SIZE,
        n_layers=N_LAYERS,
        n_heads=N_HEADS,
        # The torch backend is the only one that runs on CPU; flash kernels are GPU-only and these
        # tests are meant to run in ordinary CI.
        use_flash=False,
        attn_backend=AttentionBackendName.torch,
        dtype=DType.float32,
    )
    kwargs.update(overrides)
    return kwargs


def _build_module(
    config: TransformerConfig,
    *,
    generation_config: GenerationConfig,
    seed: int = 0,
) -> TransformerGenerationModule:
    torch.manual_seed(seed)
    model = config.build()
    model.eval()
    return TransformerGenerationModule(
        model=model,
        generation_config=generation_config,
        device=torch.device("cpu"),
    )


def _gen_config(**overrides) -> GenerationConfig:
    cfg: Dict[str, Any] = dict(
        do_sample=False,
        eos_token_id=IDS.eos,
        pad_token_id=IDS.pad,
        use_cache=True,
    )
    cfg.update(overrides)
    return GenerationConfig(**cfg)


# --------------------------------------------------------------------------------------------- #
# Variant builders
# --------------------------------------------------------------------------------------------- #


def build_dense() -> TransformerGenerationModule:
    """Plain causal attention -- the control. Its two paths compute the same function exactly."""
    config = TransformerConfig.llama_like(**_base_kwargs())
    return _build_module(config, generation_config=_gen_config())


def build_document_chunked() -> TransformerGenerationModule:
    """
    Dense attention restricted by the chunked-document mask, with ``chunk_ids`` reconstructed from
    the boundary tokens on every forward.
    """
    config = TransformerConfig.llama_like(**_base_kwargs(document_chunked=True))
    config.document_chunk_attention = dict(
        doc_start_id=IDS.doc_start,
        doc_end_id=IDS.doc_end,
        eos_id=IDS.eos,
        pad_id=IDS.pad,
    )
    return _build_module(config, generation_config=_gen_config())


def build_summary_token() -> TransformerGenerationModule:
    """
    Summary-token attention: each context document is followed by a short summary run, and a
    document may read only itself plus earlier summary runs.

    The serving mask mode is left at its default. That default is the thing to keep an eye on: the
    mask a summary-token model is served at inference is a *decision*
    (:meth:`~olmo_core.nn.transformer.Transformer.set_summary_eval_mask_mode`), not a property of the
    checkpoint, and serving the restricted mask to a causally-trained arm has silently voided results
    before.
    """
    config = TransformerConfig.llama_like(**_base_kwargs())
    assert not isinstance(config.block, dict)
    mixer = config.block.sequence_mixer
    mixer.name = AttentionType.summary_token
    mixer.n_summary_tokens = N_SUMMARY_TOKENS
    mixer.summary_visible_tokens = N_SUMMARY_TOKENS
    config.summary_token_attention = dict(
        doc_start_id=IDS.doc_start,
        doc_end_id=IDS.doc_end,
        summary_token_id=IDS.summary,
        eos_id=IDS.eos,
        pad_id=IDS.pad,
    )
    return _build_module(config, generation_config=_gen_config())


def build_sparse_landmark() -> TransformerGenerationModule:
    """Sparse landmark attention: landmark-only routing across blocks, local attention within."""
    config = TransformerConfig.llama_like(**_base_kwargs(sparse_landmark=True, mem_freq=MEM_FREQ))
    return _build_module(
        config,
        generation_config=_gen_config(
            landmark_mem_id=IDS.landmark,
            landmark_pad_id=IDS.pad,
        ),
    )


def build_compressive_landmark() -> TransformerGenerationModule:
    """
    Compressive landmark attention: each past block's landmark token also folds its value into the
    output as a compressed summary of that block.
    """
    config = TransformerConfig.llama_like(
        **_base_kwargs(fast_compressive_landmark=True, mem_freq=MEM_FREQ)
    )
    return _build_module(
        config,
        generation_config=_gen_config(
            landmark_mem_id=IDS.landmark,
            landmark_pad_id=IDS.pad,
        ),
    )


# --------------------------------------------------------------------------------------------- #
# Specs
# --------------------------------------------------------------------------------------------- #


@dataclass
class VariantSpec:
    """
    One variant under test: how to build it, what to feed it, and what agreement it owes.

    :param name: Variant name, used to parametrize the tests and label reports.
    :param build: Builds the generation module.
    :param make_prompt: Builds the content-space prompt.
    :param n_gold: Number of gold tokens to force through the decode loop.
    :param expect: ``"identical"`` if the generation path must reproduce the teacher-forced forward
        pass to numerical tolerance, or ``"gap"`` if the two paths compute deliberately different
        functions and the test records the divergence instead of failing on it.
    :param why: For ``expect="gap"``, the reason the paths differ by design. Stated here so a future
        reader can tell a designed gap from a regression that was papered over.
    :param atol: Tolerance on the max absolute gold-token logprob delta, for ``expect="identical"``.
    :param kl_atol: Tolerance on max per-step KL, for ``expect="identical"``.
    :param gap_max_kl_budget: For ``expect="gap"``, an upper bound on max per-step KL. The gap is
        designed, but it is not unbounded: a decode path that has stopped attending to the prompt at
        all also produces a "gap", and without a ceiling this test would wave it through. The value
        is an observed baseline on the tiny random models here, not a target.
    :param is_landmark: Whether generation rewrites the prompt with landmark tokens (the harness
        needs the model-space prompt to align the trace).
    :param block_multiple: Right-pad the reference forward to a multiple of this. The eager landmark
        forward rejects a partial trailing block, and a forced continuation rarely ends on one.
    :param requires_gpu: Whether the cached decode path needs CUDA. Only the eager sparse-landmark
        path runs on CPU: the shared KV cache lives behind the flash-attention backend, and
        ``TorchAttentionBackend`` refuses to cache at all, so every variant that reaches it through
        :class:`~olmo_core.nn.attention.Attention` is GPU-only.
    :param requires_flash: Whether the variant additionally needs flash-attention 2 installed.
    :param generation_kwargs: Extra overrides forwarded to ``generate_batch``.
    """

    name: str
    build: Callable[[], TransformerGenerationModule]
    make_prompt: Callable[[], torch.Tensor]
    n_gold: int = 12
    expect: str = "identical"
    why: Optional[str] = None
    atol: float = 2e-5
    kl_atol: float = 1e-8
    gap_max_kl_budget: Optional[float] = None
    is_landmark: bool = False
    block_multiple: Optional[int] = None
    requires_gpu: bool = True
    requires_flash: bool = True
    generation_kwargs: Dict[str, Any] = field(default_factory=dict)


#: Disable hard top-k landmark retrieval at decode time. Both keys are needed:
#: ``landmark_top_k_blocks=None`` alone still defers to ``landmark_top_k_fraction``, which
#: **defaults to 0.1**, so a config that only clears the block count still runs top-k retrieval.
#: With both cleared the decode gates densely over all past blocks, as the batched forward does.
LANDMARK_NO_TOPK: Dict[str, Any] = {
    "landmark_top_k_blocks": None,
    "landmark_top_k_fraction": None,
}


def landmark_drift_gold_budget(model_space_prompt_len: int, block_size: int) -> int:
    """
    The largest gold continuation that can be compared before the two paths disagree by construction.

    Landmark slots are fixed by absolute position (``pos % block_size == block_size - 1``), not by
    the token stream, and :meth:`TransformerGenerationModule.generate_batch` never inserts landmark
    tokens among *generated* tokens -- its decode modes deliberately treat the continuation as one
    growing local block. So once the continuation is long enough to occupy the next landmark slot,
    the eager forward reads that generated **content** token as a landmark while decode does not, and
    the two are computing different attention structures. That is a designed difference, not a cache
    bug.

    The disagreement starts one step later than the slot itself: a query *at* the slot still reaches
    its own block's content locally, so it agrees. Only queries that must look back *through* the
    spurious landmark diverge. Compared query positions run ``P-1 .. P-2+n_gold``, so the budget is
    how many of those land at or before the next landmark slot.

    Note ``src/corpus_reasoning/eval/eval_lc_native_docchunk.py`` avoids this in its own decode loop
    by feeding a real landmark token after every ``mem_freq`` generated tokens, explicitly "so the
    periodic ``is_mem`` structure ... matches". ``generate_batch`` has no such injection.

    :param model_space_prompt_len: Length of the landmark-rewritten prompt, ``P``.
    :param block_size: ``mem_freq + num_landmarks``.

    :returns: The maximum ``n_gold`` that can be compared without landmark drift.
    """
    # The first landmark slot occupied by a GENERATED token: the smallest position >= P congruent to
    # block_size - 1. Anchoring on the last *prompt* position instead is wrong whenever the prompt
    # ends exactly on a slot -- which is what "generation_only" guarantees, since it pads the prompt
    # to end on a landmark. That slot holds a real landmark token and causes no drift; the next one
    # does.
    p = model_space_prompt_len
    offset = (block_size - 1 - p) % block_size
    first_generated_slot = p + offset
    # Queries run P-1 .. P-2+n_gold, and the one sitting *at* the slot still agrees.
    return first_generated_slot - p + 2


VARIANTS: Dict[str, VariantSpec] = {
    # The harness's own control. With ``use_cache=False`` the decode loop re-runs the whole sequence
    # every step, so it must reproduce the teacher-forced forward to floating-point noise -- there is
    # no cache, no mask rebuild, nothing left that could legitimately differ. If this one drifts, the
    # trace alignment or the metrics are wrong and no other row in this table can be believed. It
    # needs no GPU, so it guards the rest even in CPU-only CI.
    "dense_nocache": VariantSpec(
        name="dense_nocache",
        build=build_dense,
        make_prompt=lambda: plain_prompt(48),
        expect="identical",
        atol=1e-5,
        kl_atol=1e-6,
        requires_gpu=False,
        requires_flash=False,
        generation_kwargs={"use_cache": False},
    ),
    "dense": VariantSpec(
        name="dense",
        build=build_dense,
        make_prompt=lambda: plain_prompt(48),
        expect="identical",
    ),
    "document_chunked": VariantSpec(
        name="document_chunked",
        build=build_document_chunked,
        make_prompt=lambda: document_prompt(n_docs=3, doc_len=9),
        expect="identical",
        # Generated tokens carry the FREE role, which the chunked mask lets read the whole causal
        # prefix -- so the cached decode and the full masked forward should agree here, and a gap
        # would mean the role reconstruction disagrees between a full stream and a 1-token step.
    ),
    "summary_token": VariantSpec(
        name="summary_token",
        build=build_summary_token,
        make_prompt=lambda: document_prompt(n_docs=3, doc_len=9, n_summary_tokens=N_SUMMARY_TOKENS),
        expect="identical",
    ),
    # --- Landmark, hard top-k retrieval DISABLED -------------------------------------------------
    # With top-k off the decode gates densely over every past block, which is what the batched
    # forward does, so the two paths must agree exactly -- provided the continuation also stays
    # inside the prompt's final block (see ``n_gold`` below). This is the real correctness assertion
    # for the landmark KV-cache decode, and nothing was covering it before: with the default config
    # every landmark comparison was contaminated by top-k and could only ever be "measured".
    "sparse_landmark_no_topk": VariantSpec(
        name="sparse_landmark_no_topk",
        build=build_sparse_landmark,
        make_prompt=lambda: plain_prompt(48),
        expect="identical",
        atol=1e-4,
        kl_atol=1e-5,
        is_landmark=True,
        block_multiple=MEM_FREQ + 1,
        requires_gpu=False,
        requires_flash=False,
        generation_kwargs=LANDMARK_NO_TOPK,
    ),
    "compressive_landmark_no_topk": VariantSpec(
        name="compressive_landmark_no_topk",
        build=build_compressive_landmark,
        make_prompt=lambda: plain_prompt(48),
        expect="identical",
        atol=1e-4,
        kl_atol=1e-5,
        is_landmark=True,
        block_multiple=MEM_FREQ + 1,
        generation_kwargs=LANDMARK_NO_TOPK,
    ),
    # --- Landmark, production config (hard top-k ON by default) ----------------------------------
    "sparse_landmark": VariantSpec(
        name="sparse_landmark",
        build=build_sparse_landmark,
        make_prompt=lambda: plain_prompt(48),
        expect="gap",
        why=(
            "Hard top-k landmark retrieval is applied on single-query decode steps only, never "
            "during the batched prefill, so decode attends to a strict subset of the blocks the "
            "teacher-forced forward gates over. GenerationConfig.landmark_top_k_fraction defaults "
            "to 0.1, so this is the configuration real evals run. Turning top-k off closes this gap "
            "entirely -- see the sparse_landmark_no_topk row, which asserts exact agreement."
        ),
        is_landmark=True,
        gap_max_kl_budget=0.5,
        block_multiple=MEM_FREQ + 1,
        # The eager sparse-landmark path needs neither flash attention nor CUDA, so this is the one
        # variant whose consistency check runs in ordinary CPU CI.
        requires_gpu=False,
        requires_flash=False,
    ),
    "compressive_landmark": VariantSpec(
        name="compressive_landmark",
        build=build_compressive_landmark,
        make_prompt=lambda: plain_prompt(48),
        expect="gap",
        why=(
            "Same top-k divergence as sparse landmark, plus the compressive summary term, whose "
            "per-block landmark contribution is assembled by a separate decode routine from the one "
            "the batched forward uses."
        ),
        is_landmark=True,
        gap_max_kl_budget=0.5,
        block_multiple=MEM_FREQ + 1,
    ),
}


def get_variant(name: str) -> VariantSpec:
    """
    Look up a variant spec by name.

    :param name: The variant name.

    :returns: The spec.

    :raises KeyError: If the name is not a known variant.
    """
    if name not in VARIANTS:
        raise KeyError(f"unknown variant {name!r}; known: {sorted(VARIANTS)}")
    return VARIANTS[name]


def model_space_prompt(
    spec: VariantSpec,
    gm: TransformerGenerationModule,
    prompt_ids: torch.Tensor,
) -> Optional[torch.Tensor]:
    """
    Return the token stream the model will actually see for ``prompt_ids``, or ``None`` when
    generation passes the prompt through unchanged.

    Landmark generation rewrites the prompt -- inserting a memory token after every ``mem_freq``
    content tokens, and padding the final block under ``generation_only`` -- so the reference forward
    has to be run over the rewritten stream, not the caller's prompt.

    :param spec: The variant spec.
    :param gm: The generation module (source of the landmark geometry and decode mode).
    :param prompt_ids: The content-space prompt.

    :returns: The model-space prompt, or ``None``.
    """
    if not spec.is_landmark:
        return None

    from olmo_core.generate.generation_module.transformer.generation_module import (
        _build_landmark_prompt,
    )

    layers = gm._landmark_attention_layers()
    if not layers:
        raise ValueError(f"{spec.name} is marked is_landmark but the model has no landmark layers.")
    mem_freq = int(getattr(layers[0], "mem_freq"))
    num_landmarks = int(getattr(layers[0], "num_landmarks", 1))
    cfg = gm._generation_config
    pad_id = cfg.landmark_pad_id if cfg.landmark_pad_id is not None else cfg.pad_token_id
    assert cfg.landmark_mem_id is not None, f"{spec.name} needs GenerationConfig.landmark_mem_id"
    return _build_landmark_prompt(
        prompt_ids,
        mem_freq,
        cfg.landmark_mem_id,
        mode=cfg.landmark_decode_mode,
        pad_id=pad_id,
        num_landmarks=num_landmarks,
    )


def sequences_for(spec: VariantSpec, seed: int = 1234) -> Sequence[Any]:
    """
    Build the ``(prompt_ids, gold)`` pair for a variant.

    :param spec: The variant spec.
    :param seed: RNG seed for the gold continuation.

    :returns: ``(prompt_ids, gold)``.
    """
    return spec.make_prompt(), gold_continuation(spec.n_gold, seed=seed)
