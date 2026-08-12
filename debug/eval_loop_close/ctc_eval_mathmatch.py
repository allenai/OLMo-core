"""
``ctc-eval``, plus one ladder row for ``mathmatch``.

The smoke checkpoints this repo can actually train (``debug/train_smoke/README.md``) were trained on
``mathmatch``, and ``ctc.eval.bundles.BUNDLE`` has no entry for it -- so ``ctc-eval --tasks
mathmatch`` exits with *"unknown task(s) mathmatch"* before anything else happens. The nine ladders
it does carry are the CTC suite's, whose files live on weka.

This registers the one missing row and then delegates to :func:`ctc.eval.cli.main` **unchanged**:
planning, the collision check, backend selection and load, the runner, the fingerprint guard, the
result writing and the summary all run exactly as shipped. Nothing else is patched, and nothing is
patched inside the run.

``mathmatch`` is deliberately NOT added to ``BUNDLE`` in the source tree: ``GROUPS["all"]`` is
``tuple(BUNDLE)``, so an entry there joins every ``--tasks all`` sweep and resolves to a weka file
that does not exist.

Usage mirrors ctc-eval, with the rung file supplied here::

    python debug/eval_loop_close/ctc_eval_mathmatch.py \
        --data /data/prasann/ctc_eval_loop/data/mathmatch/eval_2k.jsonl \
        --ckpt /data/prasann/ctc_smoke/ckpt_eval_loop/step30 --tasks mathmatch --attn chunked ...
"""

from __future__ import annotations

import sys
from pathlib import Path


def main(argv: list) -> int:
    """
    Register the ``mathmatch`` rung, then run ``ctc-eval``.

    :param argv: ``ctc-eval`` arguments, plus ``--data <rung file>`` which is consumed here.

    :returns: ``ctc-eval``'s exit status.
    """
    if "--data" not in argv:
        raise SystemExit("--data <rung .jsonl> is required (it is this shim's whole ladder)")
    at = argv.index("--data")
    data = Path(argv[at + 1]).resolve()
    argv = argv[:at] + argv[at + 2 :]
    if not data.exists():
        raise SystemExit(f"no such rung file: {data}")

    from ctc.eval import bundles, cli

    rows = sum(1 for line in data.open() if line.strip())
    # The bundle root is the file's directory and the rung path is its name, so `--bundle` still
    # selects the data and the result records the root it came from.
    bundles.BUNDLE["mathmatch"] = bundles.BundleTask(
        name="mathmatch",
        spec="mathmatch",
        rungs=(("2k", data.name),),
        group="smoke",
        eval_size=rows,
        note="Smoke ladder, registered by debug/eval_loop_close/ctc_eval_mathmatch.py.",
    )
    argv = [*argv, "--bundle", str(data.parent)]
    print(f"[shim] mathmatch 2k -> {data}  ({rows} rows)")
    return cli.main(argv)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
