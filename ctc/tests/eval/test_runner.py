"""
The eval loop.

Runs against a fake backend -- ``generate`` is just a callable returning strings -- which is
possible only because the runner is backend-agnostic, and is the same property that makes the
cross-backend parity test meaningful. Everything here is verified without a GPU.
"""

from __future__ import annotations

import json

import pytest

from ctc import tasks
from ctc.eval.runner import (
    MIN_EVAL_SIZE,
    EvalConfig,
    load_examples,
    run_task,
    standard_error,
)
from ctc.format import registry

tasks.load_all()


def _example(pairs=((1, 2),), n_docs=2):
    return {
        "documents": [{"text": f"Claim {i}."} for i in range(n_docs)],
        "queries": ["Find contradicting claims."],
        "answers": [""],
        "gold_doc_indices": [list(p) for p in pairs],
        "source": "pubmed",
    }


def _rung_file(tmp_path, examples, name="rung_2048.jsonl"):
    p = tmp_path / name
    p.write_text("\n".join(json.dumps(e) for e in examples) + "\n")
    return p


def _cfg(tmp_path, data_path, **kw):
    base = dict(
        ckpt=tmp_path / "ckpt",
        task=registry.get("contradiction"),
        rung="2k",
        data_path=data_path,
        ignore_fingerprint=True,
    )
    base.update(kw)
    return EvalConfig(**base)


# ── the happy path ──────────────────────────────────────────────────────────────────────────────

def test_perfect_predictions_score_one(tmp_path):
    data = _rung_file(tmp_path, [_example() for _ in range(3)])
    out = run_task(_cfg(tmp_path, data), lambda ps: ["[[1, 2]]"] * len(ps))
    assert out.primary == 1.0
    assert out.eval_size == 3
    assert out.parse_rate == 1.0


def test_wrong_predictions_score_zero(tmp_path):
    data = _rung_file(tmp_path, [_example() for _ in range(3)])
    out = run_task(_cfg(tmp_path, data), lambda ps: ["[[3, 4]]"] * len(ps))
    assert out.primary == 0.0
    assert out.parse_rate == 1.0  # it parsed fine, it was just wrong


def test_stop_rule_is_applied_to_generations(tmp_path):
    """The rambling no-cot checkpoint: correct answer, then noise."""
    data = _rung_file(tmp_path, [_example()])
    out = run_task(
        _cfg(tmp_path, data),
        lambda ps: ["[[1, 2]] and now let me explain my reasoning at length"] * len(ps),
    )
    assert out.primary == 1.0
    assert out.generations[0]["cleaned"] == "[[1, 2]]"


# ── parse rate is reported separately from score ────────────────────────────────────────────────

def test_unparseable_output_lowers_parse_rate_not_just_score(tmp_path):
    """Both score zero. Without parse_rate, a decoding regression looks like a weaker model."""
    data = _rung_file(tmp_path, [_example() for _ in range(4)])
    out = run_task(_cfg(tmp_path, data), lambda ps: ["complete gibberish"] * len(ps))
    assert out.primary == 0.0
    assert out.parse_rate == 0.0
    assert any("parse_rate" in w for w in out.warnings)


def test_full_parse_rate_produces_no_parse_warning(tmp_path):
    data = _rung_file(tmp_path, [_example()])
    out = run_task(_cfg(tmp_path, data), lambda ps: ["[[1, 2]]"] * len(ps))
    assert not any("parse_rate" in w for w in out.warnings)


# ── the silent-zero trap ────────────────────────────────────────────────────────────────────────

def test_overlong_prompts_raise_rather_than_scoring_zero(tmp_path):
    """354/500 examples were once skipped and scored 0.0 -- in both arms, so it read as 'no gap'."""
    data = _rung_file(tmp_path, [_example() for _ in range(2)])
    cfg = _cfg(tmp_path, data, max_length=10)
    with pytest.raises(ValueError, match="indistinguishable"):
        run_task(cfg, lambda ps: ["[[1, 2]]"] * len(ps), count_tokens=lambda t: len(t.split()))


def test_overlong_prompts_can_be_allowed_but_are_counted(tmp_path):
    data = _rung_file(tmp_path, [_example() for _ in range(2)])
    cfg = _cfg(tmp_path, data, max_length=10, allow_truncated=True)
    out = run_task(cfg, lambda ps: ["[[1, 2]]"] * len(ps), count_tokens=lambda t: len(t.split()))
    assert out.truncated == 2
    assert any("NOT averaged in as zeros" in w for w in out.warnings)


def test_missing_tokenizer_is_disclosed_not_silently_skipped(tmp_path):
    data = _rung_file(tmp_path, [_example()])
    out = run_task(_cfg(tmp_path, data), lambda ps: ["[[1, 2]]"] * len(ps))
    assert any("not audited" in w for w in out.warnings)


def test_prompt_length_distribution_is_recorded(tmp_path):
    data = _rung_file(tmp_path, [_example(), _example(n_docs=8)])
    out = run_task(
        _cfg(tmp_path, data), lambda ps: ["[[1, 2]]"] * len(ps),
        count_tokens=lambda t: len(t.split()),
    )
    assert out.prompt_tokens["max"] > out.prompt_tokens["min"]


# ── uncertainty travels with the number ─────────────────────────────────────────────────────────

def test_small_eval_sets_are_flagged(tmp_path):
    data = _rung_file(tmp_path, [_example() for _ in range(3)])
    out = run_task(_cfg(tmp_path, data), lambda ps: ["[[1, 2]]"] * len(ps))
    assert any(f"below {MIN_EVAL_SIZE}" in w for w in out.warnings)
    assert "⚠" in out.summary()


def test_summary_carries_the_error_bar(tmp_path):
    data = _rung_file(tmp_path, [_example() for _ in range(3)])
    out = run_task(_cfg(tmp_path, data), lambda ps: ["[[1, 2]]"] * len(ps))
    assert "±" in out.summary()
    assert "eval_size=" in out.summary()


def test_result_uses_eval_size_not_n(tmp_path):
    """`n` is reserved for corpus size in this project; reusing it caused real confusion."""
    data = _rung_file(tmp_path, [_example()])
    out = run_task(_cfg(tmp_path, data), lambda ps: ["[[1, 2]]"] * len(ps))
    d = out.to_dict()
    assert "eval_size" in d and "n" not in d


@pytest.mark.parametrize(
    "mean,n,expected",
    [(0.70, 488, 0.021), (0.95, 488, 0.010)],
)
def test_standard_error_matches_the_documented_values(mean, n, expected):
    assert standard_error(mean, n) == pytest.approx(expected, abs=0.001)


def test_standard_error_of_nothing_is_zero():
    assert standard_error(0.5, 0) == 0.0


# ── provenance ──────────────────────────────────────────────────────────────────────────────────

def test_provenance_records_what_is_needed_to_reproduce(tmp_path):
    data = _rung_file(tmp_path, [_example()])
    out = run_task(_cfg(tmp_path, data), lambda ps: ["[[1, 2]]"] * len(ps))
    for k in ("ckpt", "data_path", "git_commit", "query_position", "stop", "max_new_tokens"):
        assert k in out.provenance


def test_disabled_fingerprint_check_is_recorded_in_the_result(tmp_path):
    """Turning the guard off must show up in the file, not only in the command line."""
    data = _rung_file(tmp_path, [_example()])
    out = run_task(_cfg(tmp_path, data), lambda ps: ["[[1, 2]]"] * len(ps))
    assert any("UNVERIFIED" in w for w in out.warnings)


def test_missing_fingerprint_is_recorded_rather_than_fatal(tmp_path):
    """Checkpoints predating fingerprinting must still be gradable -- but say so."""
    data = _rung_file(tmp_path, [_example()])
    cfg = _cfg(tmp_path, data, ignore_fingerprint=False)
    out = run_task(cfg, lambda ps: ["[[1, 2]]"] * len(ps))
    assert any("UNVERIFIED" in w for w in out.warnings)


def test_limit_marks_the_run_as_a_preview(tmp_path):
    data = _rung_file(tmp_path, [_example() for _ in range(10)])
    out = run_task(_cfg(tmp_path, data, limit=2), lambda ps: ["[[1, 2]]"] * len(ps))
    assert out.eval_size == 2
    assert any("preview" in w for w in out.warnings)


# ── input validation ────────────────────────────────────────────────────────────────────────────

def test_empty_rung_file_raises(tmp_path):
    """Otherwise the result is eval_size=0 with metrics 0.0, which looks like total failure."""
    p = tmp_path / "empty.jsonl"
    p.write_text("")
    with pytest.raises(ValueError, match="no examples"):
        load_examples(p)


def test_missing_rung_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_examples(tmp_path / "nope.jsonl")


def test_backend_returning_the_wrong_count_raises(tmp_path):
    data = _rung_file(tmp_path, [_example() for _ in range(3)])
    with pytest.raises(ValueError, match="generations for"):
        run_task(_cfg(tmp_path, data), lambda ps: ["[[1, 2]]"])


# ── output ──────────────────────────────────────────────────────────────────────────────────────

def test_result_round_trips_through_json(tmp_path):
    data = _rung_file(tmp_path, [_example()])
    out = run_task(_cfg(tmp_path, data), lambda ps: ["[[1, 2]]"] * len(ps))
    path = out.write(tmp_path / "out" / "result.json")
    assert json.loads(path.read_text())["primary_value"] == 1.0


def test_generations_are_kept_by_default(tmp_path):
    """Every grading bug in this project was found by reading generations."""
    data = _rung_file(tmp_path, [_example()])
    out = run_task(_cfg(tmp_path, data), lambda ps: ["[[1, 2]]"] * len(ps))
    assert out.generations and "raw" in out.generations[0]


def test_generations_can_be_suppressed(tmp_path):
    data = _rung_file(tmp_path, [_example()])
    out = run_task(_cfg(tmp_path, data, dump_generations=False), lambda ps: ["[[1, 2]]"] * len(ps))
    assert out.generations == []


# ── backend-agnosticism, the property the parity test depends on ────────────────────────────────

def test_identical_text_from_two_backends_scores_identically(tmp_path):
    """The runner contributes everything except the text, so only the text can move the score."""
    data = _rung_file(tmp_path, [_example() for _ in range(3)])
    texts = ["[[1, 2]]", "[[1, 2]] rambling", "<think>hmm</think>[[1, 2]]"]
    a = run_task(_cfg(tmp_path, data, backend="native"), lambda ps: texts)
    b = run_task(_cfg(tmp_path, data, backend="vllm"), lambda ps: texts)
    assert a.metrics == b.metrics
    assert a.backend != b.backend
