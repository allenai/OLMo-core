"""
Results I/O for the CTC suite (``records/ctc-suite-scaling-plan.md`` §8).

Every eval writes its result through :func:`write_result`, which

1. validates the record against the §8 schema (required fields incl. nested provenance),
2. writes the pretty per-rung JSON to ``<out_root>/<task>/<model>_<arm>/rung_<T>.json``, and
3. appends the record as ONE line to ``<out_root>/all_results.jsonl`` (flock + ``O_APPEND``
   single-``write`` so concurrent eval jobs cannot interleave).

Small-eval policy (project directive): ``eval_size < 500`` is refused unless the record carries
``small_eval_ok: true`` AND a ``small_eval_warning`` string quoting the binomial SE (build it with
:func:`binomial_se` / :func:`small_eval_warning`). The contradiction held-out set (488) also needs
the flag -- the point is that sub-500 numbers are never written silently.

Self-test (no GPU, no deps beyond stdlib)::

    PYTHONPATH=src python -m scripts.eval.ctc_suite.results_io --selftest
"""

import argparse
import fcntl
import json
import math
import os
import re
import tempfile
from typing import Any, Dict, Tuple

#: Top-level fields every §8 record must carry.
REQUIRED_FIELDS = (
    "task",
    "complexity_class",
    "model",
    "arm",
    "rung_tokens",
    "metric_name",
    "metric_value",
    "eval_size",
    "cot_label",
)

#: Required keys inside ``record["provenance"]``.
REQUIRED_PROVENANCE = ("git_commit", "ckpt_path", "eval_backend")

#: Below this eval size a record is refused unless it opts in via ``small_eval_ok``.
MIN_EVAL_SIZE = 500

#: Basename of the flat append-only log under ``out_root``.
ALL_RESULTS_NAME = "all_results.jsonl"


def binomial_se(p: float, n: int) -> float:
    """
    Binomial standard error ``sqrt(p * (1 - p) / n)`` of a rate metric.

    :param p: The observed rate (metric value in ``[0, 1]``).
    :param n: The number of eval examples the rate was computed over.

    :returns: The standard error (0.0 when ``n <= 0`` is refused with ``ValueError``).

    :raises ValueError: If ``n`` is not positive or ``p`` is outside ``[0, 1]``.
    """
    if n <= 0:
        raise ValueError(f"binomial_se needs n > 0, got n={n}")
    if not (0.0 <= p <= 1.0):
        raise ValueError(f"binomial_se needs p in [0, 1], got p={p}")
    return math.sqrt(p * (1.0 - p) / n)


def small_eval_warning(p: float, n: int) -> str:
    """
    Render the inline warning string a sub-500 record must carry.

    :param p: The metric value (a rate in ``[0, 1]``).
    :param n: The eval size.

    :returns: e.g. ``"eval_size=488 only (binomial SE +/-0.021 at p=0.70)"``.
    """
    se = binomial_se(min(max(p, 0.0), 1.0), n)
    return f"eval_size={n} only (binomial SE +/-{se:.3f} at p={p:.2f})"


def _sanitize_component(value: Any) -> str:
    """Turn a record field into a safe single path component (no separators / weird chars)."""
    s = str(value)
    s = re.sub(r"[^A-Za-z0-9._+=@-]+", "-", s).strip("-.")
    if not s:
        raise ValueError(f"field value {value!r} sanitizes to an empty path component")
    return s


def validate_record(record: Dict[str, Any]) -> None:
    """
    Validate a record against the §8 schema; raise on any violation.

    Checks: required top-level fields, required ``provenance`` keys, numeric
    ``rung_tokens``/``metric_value``/``eval_size``, and the small-eval policy
    (``eval_size < 500`` requires ``small_eval_ok=True`` + a ``small_eval_warning`` string).

    :param record: The result record (see the module docstring for the schema).

    :raises ValueError: If the record is not a valid §8 record.
    """
    if not isinstance(record, dict):
        raise ValueError(f"record must be a dict, got {type(record).__name__}")
    missing = [k for k in REQUIRED_FIELDS if k not in record]
    if missing:
        raise ValueError(f"record is missing required fields: {missing}")
    prov = record.get("provenance")
    if not isinstance(prov, dict):
        raise ValueError("record is missing the required 'provenance' dict")
    missing_prov = [k for k in REQUIRED_PROVENANCE if not prov.get(k)]
    if missing_prov:
        raise ValueError(f"record['provenance'] is missing required keys: {missing_prov}")

    for key in ("rung_tokens", "eval_size"):
        if not isinstance(record[key], int) or record[key] <= 0:
            raise ValueError(f"record[{key!r}] must be a positive int, got {record[key]!r}")
    if not isinstance(record["metric_value"], (int, float)):
        raise ValueError(f"record['metric_value'] must be numeric, got {record['metric_value']!r}")

    eval_size = record["eval_size"]
    if eval_size < MIN_EVAL_SIZE:
        if record.get("small_eval_ok") is not True:
            raise ValueError(
                f"eval_size={eval_size} < {MIN_EVAL_SIZE}: refusing to record a small eval without "
                "small_eval_ok=True in the record (project directive: sub-500 evals must be "
                "explicit, never silent)."
            )
        warning = record.get("small_eval_warning")
        if not isinstance(warning, str) or not warning.strip():
            raise ValueError(
                f"eval_size={eval_size} < {MIN_EVAL_SIZE} with small_eval_ok=True still requires a "
                "non-empty 'small_eval_warning' string quoting the binomial SE -- build it with "
                "results_io.small_eval_warning(metric_value, eval_size)."
            )


def rung_json_path(out_root: str, record: Dict[str, Any]) -> str:
    """
    The per-rung JSON path for a record: ``<out_root>/<task>/<model>_<arm>/rung_<T>.json``.

    :param out_root: The results root (normally ``results/ctc_suite``).
    :param record: A record carrying at least ``task``, ``model``, ``arm``, ``rung_tokens``.

    :returns: The absolute or ``out_root``-relative file path (not created yet).
    """
    task = _sanitize_component(record["task"])
    model = _sanitize_component(record["model"])
    arm = _sanitize_component(record["arm"])
    return os.path.join(out_root, task, f"{model}_{arm}", f"rung_{int(record['rung_tokens'])}.json")


def append_jsonl(out_root: str, record: Dict[str, Any]) -> str:
    """
    Append ``record`` as one line to ``<out_root>/all_results.jsonl``, safely under concurrency.

    An exclusive ``fcntl.flock`` on a sibling ``.lock`` file serializes appenders, and the payload
    itself goes down as a single ``os.write`` on an ``O_APPEND`` fd -- either guard alone is enough
    on a local filesystem; together they also survive an appender that skips the lock.

    :param out_root: The results root directory (created if needed).
    :param record: The (already validated) record.

    :returns: The path of the JSONL file appended to.
    """
    os.makedirs(out_root, exist_ok=True)
    jsonl_path = os.path.join(out_root, ALL_RESULTS_NAME)
    line = json.dumps(record, sort_keys=True) + "\n"
    lock_path = jsonl_path + ".lock"
    with open(lock_path, "w") as lock_f:
        fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX)
        fd = os.open(jsonl_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        try:
            os.write(fd, line.encode("utf-8"))
        finally:
            os.close(fd)
        fcntl.flock(lock_f.fileno(), fcntl.LOCK_UN)
    return jsonl_path


def write_result(out_root: str, record: Dict[str, Any]) -> Tuple[str, str]:
    """
    Validate ``record`` and write it to both §8 sinks.

    The per-rung JSON is written atomically (temp file + ``os.replace``) to
    ``<out_root>/<task>/<model>_<arm>/rung_<T>.json`` (re-running an eval overwrites its rung
    file), and the record is appended as one line to ``<out_root>/all_results.jsonl``
    (append-only log; readers should keep the LAST line per (task, model, arm, rung)).

    :param out_root: The results root (normally ``results/ctc_suite``).
    :param record: The §8 record; see :func:`validate_record` for requirements.

    :returns: ``(per_rung_json_path, all_results_jsonl_path)``.

    :raises ValueError: If the record fails validation (nothing is written).
    """
    validate_record(record)
    per_rung = rung_json_path(out_root, record)
    os.makedirs(os.path.dirname(per_rung), exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(per_rung), suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(record, f, indent=2, sort_keys=True)
            f.write("\n")
        os.replace(tmp, per_rung)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
    jsonl_path = append_jsonl(out_root, record)
    return per_rung, jsonl_path


def _dummy_record(eval_size: int = 500) -> Dict[str, Any]:
    """A schema-complete dummy record for the self-test."""
    return {
        "task": "contradiction",
        "complexity_class": "N2",
        "model": "qwen3.5-0.8b",
        "arm": "chunked",
        "mask_mix": {"mode": "curriculum", "start_p": 0.8, "end_p": 0.0},
        "trained_ctx": "2k-32k-joint-uniform",
        "rung_tokens": 8192,
        "n_docs": 180,
        "metric_name": "set_f1",
        "metric_value": 0.71,
        "aux_metrics": {"precision": 0.70, "recall": 0.72, "parse_rate": 0.99},
        "eval_size": eval_size,
        "cot_label": "no-cot",
        "provenance": {
            "git_commit": "deadbeef",
            "data_path": "data/contra_eval_rung8192.jsonl",
            "ckpt_path": "/tmp/ckpt/step1100",
            "eval_backend": "native",
            "launcher": "selftest",
            "date": "2026-07-18",
        },
    }


def _selftest() -> None:
    """Round-trip a dummy record through both sinks in a temp dir; assert every §8 guarantee."""
    with tempfile.TemporaryDirectory(prefix="ctc_results_io_") as root:
        rec = _dummy_record()
        per_rung, jsonl_path = write_result(root, rec)
        expected = os.path.join(root, "contradiction", "qwen3.5-0.8b_chunked", "rung_8192.json")
        assert per_rung == expected, (per_rung, expected)
        with open(per_rung) as f:
            assert json.load(f) == rec, "per-rung JSON did not round-trip"
        # Append twice more; the log must hold one line per write, each parsing back to the record.
        write_result(root, rec)
        write_result(root, rec)
        with open(jsonl_path) as f:
            lines = f.read().splitlines()
        assert len(lines) == 3 and all(json.loads(ln) == rec for ln in lines), "jsonl log wrong"

        # Small-eval policy: <500 refused bare, refused with flag but no warning, ok with both.
        small = _dummy_record(eval_size=488)
        for patch, should_pass in (
            ({}, False),
            ({"small_eval_ok": True}, False),
            ({"small_eval_ok": True, "small_eval_warning": small_eval_warning(0.71, 488)}, True),
        ):
            trial = {**small, **patch}
            try:
                write_result(root, trial)
                ok = True
            except ValueError:
                ok = False
            assert ok == should_pass, f"small-eval policy mismatch for patch={patch}"

        # Missing-field validation must refuse before writing anything.
        broken = _dummy_record()
        del broken["metric_name"]
        try:
            write_result(root, broken)
            raise AssertionError("record without metric_name was accepted")
        except ValueError:
            pass
        broken = _dummy_record()
        del broken["provenance"]["git_commit"]
        try:
            write_result(root, broken)
            raise AssertionError("record without provenance.git_commit was accepted")
        except ValueError:
            pass

        se = binomial_se(0.70, 488)
        assert abs(se - 0.0207) < 1e-3, se
        print(
            f"[results_io selftest] PASS  per_rung={os.path.relpath(per_rung, root)}  "
            f"jsonl_lines=4  binomial_se(0.70, 488)={se:.4f}"
        )


def main() -> None:
    """CLI entry point (``--selftest`` only for now)."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--selftest", action="store_true", help="round-trip a dummy record in a tmp dir"
    )
    args = ap.parse_args()
    if args.selftest:
        _selftest()
    else:
        ap.error("nothing to do (pass --selftest)")


if __name__ == "__main__":
    main()
