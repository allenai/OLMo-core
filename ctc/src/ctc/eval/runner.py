"""
The eval loop: checkpoint plus task data in, a results file out.

Backend-agnostic by construction. The prompt, the stop rule, the parser and the scorer all come
from :mod:`ctc.format` and the task's spec, so the only thing a backend contributes is *text*. A
score that moves when only ``--backend`` moves is therefore a backend bug, and that is exactly what
the parity test asserts.

Three failure modes are designed against here, because each one previously produced a plausible
number rather than an error:

**An undersized budget scores 0.0 instead of failing.** The pre-migration evaluators skipped any
prompt longer than ``max_length - max_new`` and recorded it as an empty generation, which parses
cleanly to a score of zero. Rung labels bound *corpus size*, not prompt length -- measured
contradiction prompts at the "4k" rung spanned 3,457 to 23,796 tokens -- so a budget derived from
the label silently skipped 354 of 500 examples. It happened in both arms at once, so it read as
"no dense-vs-chunked gap". Here an over-long prompt is a hard error unless explicitly allowed, and
if allowed it is counted and reported rather than averaged in as a zero.

**A parse failure is indistinguishable from a wrong answer.** Both score zero. ``parse_rate`` is
therefore always reported: if it moves, the problem is decoding or truncation, not capability.

**A number is quoted without its uncertainty.** Every result carries ``eval_size`` and a standard
error, and small eval sets are flagged in the file itself rather than left for the reader to
notice.
"""

from __future__ import annotations

import json
import math
import platform
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from ..format import metrics as metrics_mod
from ..format.fingerprint import check_or_explain_missing
from ..format.registry import TaskSpec
from .stopping import STOP_PRESETS
from .stopping import apply as apply_stop

__all__ = ["EvalConfig", "EvalOutcome", "run_task", "standard_error", "load_examples"]

#: Below this, a result is flagged in the file. Small eval sets inflate noise into apparent
#: findings, so the warning travels with the number rather than living in someone's memory.
MIN_EVAL_SIZE = 500


def standard_error(mean: float, n: int) -> float:
    """
    Binomial standard error for a right/wrong-graded metric.

    :param mean: The metric value, in ``[0, 1]``.
    :param n: Number of examples it was averaged over.

    :returns: The standard error, or ``0.0`` when ``n`` is zero. At 488 examples this is about
        ±0.021 at 0.70 and ±0.010 at 0.95 -- larger than several differences that have been
        reported as findings, which is why it is emitted rather than computed on request.
    """
    if n <= 0:
        return 0.0
    p = min(max(mean, 0.0), 1.0)
    return math.sqrt(p * (1.0 - p) / n)


def load_examples(path: Path, limit: Optional[int] = None) -> List[Dict]:
    """
    Read a rung file.

    :param path: A JSONL file of unified-format examples.
    :param limit: Keep only the first N. Anything less than the full file makes the result a
        preview, and :func:`run_task` records that it was truncated.

    :returns: The examples.

    :raises FileNotFoundError: If the file is absent.
    :raises ValueError: If the file is empty -- silently grading nothing would otherwise produce a
        results file with ``eval_size: 0`` and metrics of 0.0, which looks like total failure.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"no such rung file: {path}")
    out = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
            if limit is not None and len(out) >= limit:
                break
    if not out:
        raise ValueError(f"{path} contains no examples")
    return out


@dataclass
class EvalConfig:
    """
    One grading run.

    :param ckpt: Checkpoint directory.
    :param task: The task spec.
    :param rung: Rung label being graded.
    :param data_path: The rung file.
    :param attn: Attention mask to grade under: ``full``, ``chunked`` or ``landmark``.
    :param backend: Backend name, recorded in the result.
    :param max_length: Total sequence budget. ``None`` sizes it from the measured prompt
        distribution, which is the only safe way -- see the module docstring.
    :param max_new_tokens: Decode budget; defaults to the task's.
    :param query_position: Prompt layout knob, recorded in the fingerprint.
    :param limit: Grade only the first N examples.
    :param allow_truncated: Permit prompts that exceed the budget, counting them instead of
        raising. Off by default: a skipped example scores a clean 0.0 and looks like a wrong answer.
    :param ignore_fingerprint: Skip the train/eval format check. The result records that
        compatibility was unverified.
    :param dump_generations: Keep raw generations. On by default -- every grading bug in this
        project was found by reading generations for a task that scored near zero.
    """

    ckpt: Path
    task: TaskSpec
    rung: str
    data_path: Path
    attn: str = "full"
    backend: str = "native"
    max_length: Optional[int] = None
    max_new_tokens: Optional[int] = None
    query_position: str = "both"
    limit: Optional[int] = None
    allow_truncated: bool = False
    ignore_fingerprint: bool = False
    dump_generations: bool = True


@dataclass
class EvalOutcome:
    """The result of one grading run, and everything needed to interpret it later."""

    task: str
    rung: str
    attn: str
    backend: str
    eval_size: int
    metrics: Dict[str, float]
    standard_errors: Dict[str, float]
    primary_metric: str
    parse_rate: float
    cot_label: str = "no-cot"
    warnings: List[str] = field(default_factory=list)
    truncated: int = 0
    prompt_tokens: Dict[str, int] = field(default_factory=dict)
    provenance: Dict[str, Any] = field(default_factory=dict)
    generations: List[Dict] = field(default_factory=list)

    @property
    def primary(self) -> float:
        """:returns: The task's headline number."""
        return self.metrics[self.primary_metric]

    def summary(self) -> str:
        """
        :returns: A one-line summary, carrying the error bar and any small-eval warning -- so the
            number cannot be read off without them.
        """
        se = self.standard_errors.get(self.primary_metric, 0.0)
        line = (
            f"{self.task} @{self.rung} [{self.attn}/{self.backend}] "
            f"{self.primary_metric}={self.primary:.3f} ±{se:.3f}  eval_size={self.eval_size}"
        )
        if self.eval_size < MIN_EVAL_SIZE:
            line += f"  ⚠ eval_size={self.eval_size} only"
        if self.parse_rate < 1.0:
            line += f"  parse_rate={self.parse_rate:.3f}"
        return line

    def to_dict(self) -> Dict[str, Any]:
        """:returns: A JSON-ready mapping."""
        d = asdict(self)
        d["primary_value"] = self.primary
        return d

    def write(self, path: Path) -> Path:
        """
        :param path: Destination ``.json``.

        :returns: The path written.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n")
        return path


def _jsonable(value: Any) -> Any:
    """
    Make a parsed answer safe to ``json.dumps``.

    Half the suite parses to a ``set`` -- ``parse_id_set`` backs the whole retrieval/absence family
    -- and ``json`` cannot serialise one. Because the result file is written *after* the rung is
    graded, an unserialisable answer meant a task generated all its examples, scored them, and then
    threw the entire run away at the write step. Sorted rather than listed in iteration order so a
    dumped generation is stable run to run and two dumps can be diffed.

    :param value: A parsed answer, gold value, or any nesting of them.

    :returns: The same value with sets replaced by sorted lists.
    """
    if isinstance(value, (set, frozenset)):
        try:
            return sorted(value)
        except TypeError:  # mixed types: order is arbitrary, but it must still serialise
            return sorted(value, key=repr)
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _git_commit() -> Optional[str]:
    """:returns: The current commit, so a result can be traced to the code that produced it."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        return out.stdout.strip()
    except (subprocess.SubprocessError, OSError):
        return None


def run_task(
    cfg: EvalConfig,
    generate: Callable[[Sequence[str]], Sequence[str]],
    *,
    count_tokens: Optional[Callable[[str], int]] = None,
) -> EvalOutcome:
    """
    Grade one task at one rung.

    :param cfg: What to run.
    :param generate: ``(prompts, examples) -> texts``. The whole of a backend's contribution;
        everything else comes from the task spec, which is what makes backends comparable.

        ``examples`` accompanies the prompts because tokenization is a backend concern and some
        backends need STRUCTURE, not just text: the native path wraps each document in markers, and
        document boundaries are not reliably recoverable from a flat string. Backends that plain-
        tokenize ignore it.
    :param count_tokens: ``(text) -> n``. Used for the length audit. Without it the audit is
        skipped and the result says so -- an unaudited run cannot rule out the silent-zero failure.

    :returns: The outcome.

    :raises ValueError: If prompts exceed the budget and ``allow_truncated`` is not set.
    """
    spec = cfg.task
    stop = STOP_PRESETS[spec.stop]
    max_new = cfg.max_new_tokens or spec.max_new_tokens

    warnings: List[str] = []

    # --- format compatibility -------------------------------------------------------------------
    if cfg.ignore_fingerprint:
        warnings.append(
            "format fingerprint check DISABLED; train/eval format compatibility is UNVERIFIED"
        )
    else:
        eval_fp = spec.fingerprint(query_position=cfg.query_position)
        missing = check_or_explain_missing(eval_fp, Path(cfg.ckpt), strict=False)
        if missing:
            warnings.append(missing)

    examples = load_examples(Path(cfg.data_path), limit=cfg.limit)
    if cfg.limit is not None:
        warnings.append(f"limited to the first {cfg.limit} examples; this is a preview, not a run")

    prompts = [spec.build_prompt(ex, query_position=cfg.query_position) for ex in examples]

    # --- length audit ---------------------------------------------------------------------------
    # The failure this guards: a prompt longer than the budget was skipped and scored as an empty
    # generation, i.e. a clean 0.0. It hit both arms equally and read as "no gap between them".
    prompt_tokens: Dict[str, int] = {}
    truncated = 0
    if count_tokens is not None:
        lengths = [count_tokens(p) for p in prompts]
        prompt_tokens = {
            "min": min(lengths),
            "max": max(lengths),
            "mean": int(sum(lengths) / len(lengths)),
        }
        budget = cfg.max_length if cfg.max_length is not None else max(lengths) + max_new + 512
        over = [n for n in lengths if n + max_new > budget]
        truncated = len(over)
        if truncated:
            msg = (
                f"{truncated}/{len(lengths)} prompts exceed max_length={budget} with "
                f"max_new={max_new} (longest prompt {max(lengths)}). Rung labels bound corpus "
                f"size, not prompt length."
            )
            if not cfg.allow_truncated:
                raise ValueError(
                    msg + " These would be skipped and scored 0.0, which is indistinguishable "
                    "from wrong answers. Raise --max-length, or pass allow_truncated to count "
                    "them instead."
                )
            warnings.append(msg + " Counted as truncated, NOT averaged in as zeros.")
    else:
        warnings.append(
            "no tokenizer supplied, so prompt lengths were not audited; an over-long prompt "
            "would be scored 0.0 rather than reported"
        )

    # --- generate and score ---------------------------------------------------------------------
    raw = list(generate(prompts, examples))
    if len(raw) != len(prompts):
        raise ValueError(f"backend returned {len(raw)} generations for {len(prompts)} prompts")

    per_example: List[Dict[str, float]] = []
    generations: List[Dict] = []
    n_parsed = 0

    for ex, prompt, text in zip(examples, prompts, raw):
        cleaned = apply_stop(text, stop)
        parsed = spec.parse(cleaned, len(ex["documents"]))
        scored = spec.score(parsed, ex["gold_doc_indices"])
        per_example.append(scored)
        n_parsed += int(scored.get("parsed", 1.0))
        if cfg.dump_generations:
            generations.append(
                {
                    "raw": text,
                    "cleaned": cleaned,
                    "parsed": _jsonable(parsed),
                    "gold": _jsonable(ex["gold_doc_indices"]),
                    "prompt_chars": len(prompt),
                    **{k: v for k, v in scored.items() if k != "parsed"},
                }
            )

    keys = sorted({k for r in per_example for k in r})
    agg = metrics_mod.aggregate(per_example, keys)
    eval_size = len(per_example)

    if eval_size < MIN_EVAL_SIZE:
        warnings.append(
            f"eval_size={eval_size} is below {MIN_EVAL_SIZE}; quote it inline with the error bar"
        )

    parse_rate = n_parsed / eval_size if eval_size else 0.0
    if parse_rate < 1.0:
        warnings.append(
            f"parse_rate={parse_rate:.3f} -- {eval_size - n_parsed} generation(s) did not parse. "
            "Read them before treating the score as a capability measurement."
        )

    return EvalOutcome(
        task=spec.name,
        rung=cfg.rung,
        attn=cfg.attn,
        backend=cfg.backend,
        eval_size=eval_size,
        metrics=agg,
        standard_errors={k: standard_error(v, eval_size) for k, v in agg.items()},
        primary_metric=spec.primary_metric,
        parse_rate=parse_rate,
        warnings=warnings,
        truncated=truncated,
        prompt_tokens=prompt_tokens,
        provenance={
            "ckpt": str(cfg.ckpt),
            "data_path": str(cfg.data_path),
            "git_commit": _git_commit(),
            "host": platform.node(),
            "query_position": cfg.query_position,
            "stop": spec.stop,
            "max_new_tokens": max_new,
            "max_length": cfg.max_length,
        },
        generations=generations,
    )
