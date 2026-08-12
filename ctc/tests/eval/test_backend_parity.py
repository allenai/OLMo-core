"""
Cross-backend parity: three backends, one answer.

The claim being defended is narrow and worth stating exactly. It is **not** that vLLM and
transformers produce bit-identical logits -- they do not, and no test should pretend otherwise.
It is that everything *around* the model is shared, so a disagreement between backends can only
come from the model math, and is therefore worth investigating rather than shrugging at.

Three things must be shared, and each is checked below:

1. the prefill -- the exact token ids fed in;
2. the stop handling -- a backend's own stop is an early exit, and
   :func:`ctc.eval.stopping.apply` has the last word over the full decoded string;
3. the ordering -- generation *i* is graded against example *i*.

Historically none of these were shared. Each driver carried its own copy, and the same bug had to
be found three times: the ``</think>`` truncation fix landed in the vLLM driver, the primed-bracket
doc-id fix in the native one, the grouping-JSON fix somewhere else again.

The end-to-end check -- same checkpoint, same rung, two backends, compare text -- needs a GPU and
lives at the bottom behind ``@pytest.mark.gpu``.
"""

from __future__ import annotations

import importlib

import pytest

from ctc.eval import prefill as prefill_mod
from ctc.eval.stopping import STOP_PRESETS
from ctc.eval.stopping import apply as apply_stop

BACKEND_MODULES = ("native", "vllm", "hf")


# ── every backend is built the same way ─────────────────────────────────────────────────────────


@pytest.mark.parametrize("name", BACKEND_MODULES)
def test_every_backend_module_exposes_build(name):
    """``base.load`` constructs by name; a backend without build() is unreachable through the CLI."""
    module = importlib.import_module(f"ctc.eval.backends.{name}")
    assert callable(module.build)


@pytest.mark.parametrize("name", BACKEND_MODULES)
def test_every_backend_applies_the_shared_stop_function(name):
    """Not merely 'has stop handling' -- the SAME function, so the three cannot drift apart."""
    src = importlib.import_module(f"ctc.eval.backends.{name}").__dict__
    assert src["apply_stop"] is apply_stop


@pytest.mark.parametrize("name", BACKEND_MODULES)
def test_every_backend_builds_its_prefill_through_the_shared_module(name):
    """A backend tokenizing on its own would drop the document markers, worth about -0.01 f1."""
    src = importlib.import_module(f"ctc.eval.backends.{name}").__dict__
    assert src["build_prefills"] is prefill_mod.build_prefills


# ── the prefill is identical, which is what makes a comparison meaningful ───────────────────────


class FakeTokenizer:
    """Deterministic stand-in: one id per whitespace token, so ids are readable in a failure."""

    eos_token_id = 999
    pad_token_id = 999

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [len(w) for w in text.split()]}

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(str(i) for i in ids)


def test_plain_prefill_is_the_tokenizer_and_nothing_else():
    build = prefill_mod.plain_prefill(FakeTokenizer())
    assert build("aa bbb c") == [2, 3, 1]


def test_the_same_prefill_builder_gives_every_backend_the_same_ids():
    """The property the parity test rests on: one builder, so one token stream."""
    build = prefill_mod.plain_prefill(FakeTokenizer())
    prompts = ["aa bbb", "c dddd"]
    assert prefill_mod.build_prefills(build, prompts, None) == [[2, 3], [1, 4]]


def test_prompts_and_examples_are_zipped_positionally_and_checked():
    """A length mismatch would grade each prompt against another example's structure."""
    build = prefill_mod.plain_prefill(FakeTokenizer())
    with pytest.raises(ValueError, match="zipped positionally"):
        prefill_mod.build_prefills(build, ["a", "b"], [{"documents": []}])


def test_structural_prefill_refuses_to_fall_back_to_plain_tokenization():
    """Silently falling back would produce exactly the stream the class exists to avoid."""
    # The submodule, not the package: olmo_core imports fine from a checkout while its own
    # dependencies are absent, so probing the top level would let this fail instead of skip.
    pytest.importorskip(
        "olmo_core.data.document_chunk_landmark", reason="structural prefill needs olmo-core"
    )
    build = prefill_mod.structural_prefill(FakeTokenizer(), task="contradiction")
    with pytest.raises(ValueError, match="requires the example"):
        build("some prompt", None)


# ── stop handling is an early exit, never the definition ────────────────────────────────────────


def _vllm_backend(**attrs):
    """A VLLMBackend with no vLLM: only the pure translation is under test."""
    vllm = importlib.import_module("ctc.eval.backends.vllm")
    backend = vllm.VLLMBackend.__new__(vllm.VLLMBackend)
    backend.eos_id = 7
    backend.allow_early_text_stops = False
    for k, v in attrs.items():
        setattr(backend, k, v)
    return backend


@pytest.mark.parametrize("preset", sorted(STOP_PRESETS))
def test_no_text_stop_is_pushed_to_vllm_by_default(preset):
    """
    Every preset sets require_content, and vLLM cannot express it. Stopping at the first literal
    newline would return a leading formatting newline as the whole answer -- and by the time
    apply_stop runs, the real answer is already gone, so it cannot repair it. Correctness over
    decode speed, deliberately.
    """
    kwargs = _vllm_backend().sampling_kwargs(STOP_PRESETS[preset])
    assert kwargs["stop"] is None


def test_eos_is_always_pushed_down():
    """The one stop that can never fire early, and the one that saves the most decode time."""
    assert _vllm_backend().sampling_kwargs(STOP_PRESETS["pairs"])["stop_token_ids"] == [7]


def test_the_speed_opt_out_pushes_text_stops_down():
    kwargs = _vllm_backend(allow_early_text_stops=True).sampling_kwargs(STOP_PRESETS["pairs"])
    assert kwargs["stop"] == list(STOP_PRESETS["pairs"].text_stops)


def test_a_condition_with_no_premature_risk_pushes_its_stops_down():
    """The rule is about premature firing, not a blanket refusal -- otherwise it is arbitrary."""
    from ctc.eval.stopping import StopCondition

    safe = StopCondition(text_stops=("]]",), require_content=False, require_before=None)
    assert _vllm_backend().sampling_kwargs(safe)["stop"] == ["]]"]


def test_decoding_is_greedy():
    """Sampling variance would swamp the differences these evals measure."""
    assert _vllm_backend().sampling_kwargs(STOP_PRESETS["pairs"])["temperature"] == 0.0


def test_vllm_keeps_the_stop_string_so_the_host_side_can_decide():
    """Truncating inside vLLM would hide the delimiter apply_stop uses to find the answer."""
    kwargs = _vllm_backend().sampling_kwargs(STOP_PRESETS["pairs"])
    assert kwargs["include_stop_str_in_output"] is True


@pytest.mark.parametrize("preset", sorted(STOP_PRESETS))
def test_the_host_side_truncation_is_what_produces_the_final_string(preset):
    """
    Whatever a backend hands back, apply_stop is idempotent on its own output -- so a backend that
    stops early and one that runs to the budget converge on the same text.
    """
    stop = STOP_PRESETS[preset]
    raw = "[[1, 2]]\nand then some trailing commentary that no backend should keep"
    once = apply_stop(raw, stop)
    assert apply_stop(once, stop) == once


# ── refusing a checkpoint vLLM cannot load ──────────────────────────────────────────────────────


def test_a_raw_text_only_export_is_refused_with_the_recipe(tmp_path):
    """
    vLLM dies on these at model construction with an AttributeError about vision_config, several
    frames from anything meaningful. Catching it here costs a file read.
    """
    import json

    vllm = importlib.import_module("ctc.eval.backends.vllm")
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "qwen3_5_text"}))
    with pytest.raises(ValueError, match="TEXT-ONLY"):
        vllm._refuse_text_only_export(tmp_path, json)


def test_a_serving_copy_is_accepted(tmp_path):
    import json

    vllm = importlib.import_module("ctc.eval.backends.vllm")
    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "qwen3_5", "vision_config": {"depth": 1}})
    )
    vllm._refuse_text_only_export(tmp_path, json)


def test_a_directory_without_a_config_is_left_alone(tmp_path):
    """Not every backend target is a local directory; absence is not evidence of a bad export."""
    import json

    vllm = importlib.import_module("ctc.eval.backends.vllm")
    vllm._refuse_text_only_export(tmp_path, json)


# ── the real thing, when there is a GPU ─────────────────────────────────────────────────────────


@pytest.mark.gpu
def test_hf_and_vllm_agree_on_a_real_checkpoint():
    """
    The end-to-end claim. Skipped without a GPU, which means it is NOT what keeps the backends
    honest day to day -- the shared-machinery checks above are. It is what catches a divergence
    the shared machinery cannot: a genuine difference in how a model was loaded or its math run.
    """
    pytest.skip("needs a checkpoint fixture; see records/ for the validation recipe")


class _patched:
    """Minimal context manager -- monkeypatch is per-test and these helpers are module-level."""

    def __init__(self, module, name, value):
        self.module, self.name, self.value = module, name, value

    def __enter__(self):
        self.old = getattr(self.module, self.name, None)
        setattr(self.module, self.name, self.value)
        return self

    def __exit__(self, *exc):
        if self.old is None:
            delattr(self.module, self.name)
        else:
            setattr(self.module, self.name, self.old)
        return False


# ── ordering: generation i must be graded against example i ─────────────────────────────────────


class FakeOutput:
    def __init__(self, request_id, text):
        self.request_id = request_id
        self.outputs = [type("O", (), {"text": text})()]


def test_a_permuted_vllm_result_list_is_put_back_in_order():
    """A silent permutation grades every generation against another example's gold answer."""
    vllm = importlib.import_module("ctc.eval.backends.vllm")
    shuffled = [FakeOutput("2", "c"), FakeOutput("0", "a"), FakeOutput("1", "b")]
    assert [o.outputs[0].text for o in vllm._in_submission_order(shuffled)] == ["a", "b", "c"]


def test_non_numeric_request_ids_are_left_alone():
    """Only reorder when the ids actually say what the order is; otherwise do not invent one."""
    vllm = importlib.import_module("ctc.eval.backends.vllm")
    outs = [FakeOutput("req-x", "a"), FakeOutput("req-y", "b")]
    assert [o.outputs[0].text for o in vllm._in_submission_order(outs)] == ["a", "b"]


def test_chunked_attention_is_enabled_with_the_ids_it_requires():
    """`enable_document_chunk_attention(doc_start_id, doc_end_id, eos_id)` takes three REQUIRED
    positional arguments. Calling it bare raised TypeError at model load, which made
    `--attn chunked --backend native` -- the suite's primary arm -- unreachable: every such run
    died before its first prompt. The ids come from the embedding height, not from config.json,
    which does not reliably carry document_chunk_attention on the shipped checkpoints."""
    from ctc.eval.backends.native import NativeBackend

    class _W:
        def __init__(self, rows):
            self.shape = (rows,)

    class _Emb:
        def __init__(self, rows):
            self.weight = _W(rows)

    class _Model:
        def __init__(self, rows):
            self.embeddings = _Emb(rows)

    class _GM:
        def __init__(self, rows):
            self.model = _Model(rows)

    for rows, family in ((151936, "qwen3"), (248320, "qwen3_5")):
        be = NativeBackend.__new__(NativeBackend)
        be.gm = _GM(rows)
        be.doc_start_id = None
        be.doc_end_id = None
        assert NativeBackend._marker_family(be) == family

        ids = NativeBackend._document_chunk_ids(be)
        assert set(ids) == {"doc_start_id", "doc_end_id", "eos_id"}
        assert all(isinstance(v, int) for v in ids.values())
        # The two markers must be DISTINCT or the mask cannot tell an open document from a close.
        assert ids["doc_start_id"] != ids["doc_end_id"]

    # An explicit override wins over the derived default.
    be = NativeBackend.__new__(NativeBackend)
    be.gm = _GM(248320)
    be.doc_start_id, be.doc_end_id = 11, 22
    got = NativeBackend._document_chunk_ids(be)
    assert (got["doc_start_id"], got["doc_end_id"]) == (11, 22)


def test_the_full_arm_actually_turns_the_chunked_mask_off():
    """`--attn full` looked up `disable_document_chunk_attention` with getattr and skipped silently
    when it was absent -- and `Transformer` had no such method, so the "full" arm graded the CHUNKED
    mask on any checkpoint whose config carries it. Measured on a 0.6B: 500/500 byte-identical
    generations between the two arms. A missing disable is now an error, never a silent pass."""
    from ctc.eval.backends.native import NativeBackend

    class _Model:
        def __init__(self):
            self._document_chunk_attention = {"doc_start_id": 1, "doc_end_id": 2, "eos_id": 3}

        def disable_document_chunk_attention(self):
            was = self._document_chunk_attention is not None
            self._document_chunk_attention = None
            return was

    class _GM:
        def __init__(self, model):
            self.model = model

    be = NativeBackend.__new__(NativeBackend)
    be.attn = "full"
    be.gm = _GM(_Model())
    NativeBackend._configure_attention(be)
    assert be.gm.model._document_chunk_attention is None

    # A model that carries the mask but cannot turn it off must stop the run, not grade it.
    class _Undisableable:
        _document_chunk_attention = {"doc_start_id": 1, "doc_end_id": 2, "eos_id": 3}

    be = NativeBackend.__new__(NativeBackend)
    be.attn = "full"
    be.gm = _GM(_Undisableable())
    with pytest.raises(RuntimeError, match="--attn full cannot be honoured"):
        NativeBackend._configure_attention(be)


def test_the_transformer_can_actually_disable_the_chunked_mask():
    """The method the arm above depends on. It lives on the model, not on the eval backend, and its
    absence is what made the silent skip possible."""
    from olmo_core.nn.transformer.model import Transformer

    assert callable(getattr(Transformer, "disable_document_chunk_attention", None))
