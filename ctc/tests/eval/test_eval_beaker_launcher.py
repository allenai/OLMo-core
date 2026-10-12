"""
The Beaker eval launcher's argument surface and the command it hands the node.

Everything here is checkable without Beaker, weka or a GPU, and each check stands in for a launch
that would otherwise cost an hour of turnaround to discover: a results directory shared between
checkpoints, a flag the eval understands that the launcher cannot pass, a switch that is documented
as a default but cannot be turned off, or a combination the eval rejects only after the job starts.

The launcher and ``ctc-eval`` are deliberately the same experiment run in two places, so the tests
that matter are the ones comparing what the launcher *says* with what it *emits*.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).parents[3] / "src" / "scripts" / "ctc"
sys.path.insert(0, str(SCRIPTS))

eval_beaker = pytest.importorskip("eval_beaker", reason="launcher not importable")

ROOT = "/weka/oe-training-default/ai2-llm"


def _args(*argv):
    return eval_beaker.build_parser().parse_args(list(argv))


def _command(*argv):
    return eval_beaker.remote_command(_args(*argv), ROOT)


def _help(option):
    action = next(a for a in eval_beaker.build_parser()._actions if option in a.option_strings)
    return action.help


# ── where results land ──────────────────────────────────────────────────────────────────────────


def test_results_default_beside_the_checkpoint(tmp_path):
    """``--ckpt .../step1100`` grades into that run's own directory, not a shared one."""
    args = _args("--ckpt", "/weka/ck/prasanns/run-a/step1100")
    assert eval_beaker.default_results_dir(args, ROOT) == "/weka/ck/prasanns/run-a/ctc_eval"


def test_a_run_name_and_its_checkpoint_path_agree():
    """Both spellings of the same run must resolve to the same results directory."""
    by_name = eval_beaker.default_results_dir(_args("run-a"), ROOT)
    by_path = eval_beaker.default_results_dir(
        _args("--ckpt", f"{ROOT}/checkpoints/prasanns/run-a/step1100"), ROOT
    )
    assert by_name == by_path == f"{ROOT}/checkpoints/prasanns/run-a/ctc_eval"


@pytest.mark.parametrize(
    "argv",
    [
        ("run-a",),
        ("--ckpt", "/weka/ck/prasanns/run-a/step1100"),
        ("--ckpt", "/weka/ck/prasanns/run-a/step1100/"),
    ],
    ids=["run-name", "ckpt", "ckpt-trailing-slash"],
)
def test_no_launch_ever_defaults_to_the_shared_results_directory(argv):
    """The shared ``checkpoints/prasanns/ctc_eval`` is what made filename collisions likely."""
    command = _command(*argv)
    assert f"{ROOT}/checkpoints/prasanns/ctc_eval" not in command
    assert "/ctc_eval" in command


def test_the_results_dir_help_describes_what_the_code_does():
    """These two disagreed: the help promised ``<ckpt>/../ctc_eval`` and the code ignored --ckpt."""
    help_text = _help("--results-dir")
    assert "<ckpt>/../" in help_text and eval_beaker.RESULTS_SUBDIR in help_text
    args = _args("--ckpt", "/weka/ck/prasanns/run-a/step1100")
    promised = f"{Path(args.ckpt).parent}/{eval_beaker.RESULTS_SUBDIR}"
    assert eval_beaker.default_results_dir(args, ROOT) == promised


def test_an_explicit_results_dir_still_wins():
    assert '--out "/weka/elsewhere"' in _command("run-a", "--results-dir", "/weka/elsewhere")


def test_check_bundle_without_a_run_writes_nowhere_shared():
    """It passes --dry-run and produces no results, so it must not mkdir a shared weka directory."""
    command = _command("--check-bundle")
    assert eval_beaker.CHECK_BUNDLE_RESULTS in command
    assert f"{ROOT}/checkpoints/prasanns/ctc_eval" not in command
    assert "--dry-run" in command


# ── --keep-going has an off switch ──────────────────────────────────────────────────────────────


def test_keep_going_defaults_on_for_an_unattended_sweep():
    assert _args("run-a").keep_going is True
    assert "--keep-going" in _command("run-a")


def test_keep_going_can_be_turned_off():
    """It was ``store_true, default=True``, i.e. on with no way to say otherwise."""
    assert _args("run-a", "--no-keep-going").keep_going is False
    assert "--keep-going" not in _command("run-a", "--no-keep-going")


# ── --backend and --mem-freq reach the node ─────────────────────────────────────────────────────


def test_the_backend_is_the_nodes_choice_unless_named():
    assert "--backend" not in _command("run-a")


@pytest.mark.parametrize("backend", ["native", "vllm", "hf"])
def test_a_named_backend_is_forwarded(backend):
    assert f"--backend {backend}" in _command("run-a", "--backend", backend)


def test_an_unknown_backend_is_rejected_at_parse_time():
    with pytest.raises(SystemExit):
        _args("run-a", "--backend", "tensorrt")


def test_mem_freq_is_always_stated():
    """A landmark checkpoint's block size changes what it is graded as; it should not be implicit."""
    assert "--mem-freq 63" in _command("run-a")
    assert "--mem-freq 31" in _command("run-a", "--attn", "landmark", "--mem-freq", "31")


def test_the_launcher_forwards_the_same_mem_freq_default_as_the_eval():
    from ctc.eval import cli

    assert _args("run-a").mem_freq == cli.build_parser().parse_args(["--ckpt", "x"]).mem_freq


# ── --share-prefix is native-only, and says so here rather than on the node ─────────────────────


def test_share_prefix_with_the_native_backend_is_reachable():
    command = _command("run-a", "--backend", "native", "--share-prefix")
    assert "--share-prefix" in command and "--backend native" in command


@pytest.mark.parametrize("backend", ["vllm", "hf"])
def test_share_prefix_on_another_backend_fails_before_submission(backend):
    """``ctc.eval.cli`` raises the same SystemExit, but only after the job has been scheduled."""
    with pytest.raises(SystemExit) as excinfo:
        eval_beaker.main(["run-a", "--backend", backend, "--share-prefix"])
    message = str(excinfo.value)
    assert "--share-prefix" in message and backend in message


# ── the overwrite escape hatch is reachable from here too ───────────────────────────────────────


def test_overwrite_is_off_by_default_and_forwarded_when_asked():
    assert "--overwrite" not in _command("run-a")
    assert "--overwrite" in _command("run-a", "--overwrite")


# ── flags the eval understands are the flags the launcher can send ──────────────────────────────


def test_every_forwarded_flag_exists_on_the_eval_cli():
    """The two halves are one experiment; a flag the eval does not know is a job that dies at once."""
    from ctc.eval import cli

    known = {opt for action in cli.build_parser()._actions for opt in action.option_strings}
    command = _command(
        "run-a",
        "--backend",
        "native",
        "--share-prefix",
        "--mem-freq",
        "31",
        "--tag",
        "t",
        "--overwrite",
        "--limit",
        "5",
        "--ignore-format-fingerprint",
        "--allow-truncated",
    )
    invocation = next(line for line in command.splitlines() if "ctc.eval.cli" in line)
    sent = {word for word in invocation.split() if word.startswith("--")}
    assert sent <= known, f"not understood by ctc-eval: {sorted(sent - known)}"
