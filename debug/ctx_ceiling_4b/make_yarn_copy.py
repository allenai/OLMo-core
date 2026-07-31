"""
Build a **YaRN serving copy** of an olmo_core checkpoint step dir, so the native long-context
eval can score rungs past the model's trained context.

``eval_lc_native.py`` warns (and the numbers are meaningless) when a selected rung exceeds the
checkpoint's trained ``max_position_embeddings``: without RoPE extension the model reads garbage
positions and the rung looks like a long-context collapse that is really a config error. This
script produces the extended copy that warning tells you to point ``--model-path`` at.

The copy is **config-only**: it writes a patched ``config.json`` whose every
:class:`~olmo_core.nn.rope.RoPEConfig` gains a :class:`~olmo_core.nn.rope.YaRNRoPEScalingConfig`,
and symlinks ``model_and_optim`` back at the original weights. A copy is therefore ~15 KB, not
~57 GB, and stays valid as long as the source step dir exists.

Pick ``--factor`` as the smallest multiple of the trained context that covers the target rung
(e.g. a 262,144-trained model needs factor 2 for the 512k rung, 4 for 1M, 8 for 2M). Over-scaling
a short rung needlessly degrades it, so prefer one copy per rung over one copy for all.

Usage::

    python debug/ctx_ceiling_4b/make_yarn_copy.py \\
        --src /weka/.../checkpoints/amandab/<run>/step560 \\
        --factor 2                       # -> <run>/step560_yarn2, covers 524288

    # explicit destination + a dry run that only prints the patch
    python debug/ctx_ceiling_4b/make_yarn_copy.py --src .../step560 --factor 4 \\
        --dest .../step560_yarn4 --dry-run
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

ROPE_CLASS = "olmo_core.nn.rope.RoPEConfig"
YARN_CLASS = "olmo_core.nn.rope.YaRNRoPEScalingConfig"


def find_rope_configs(node, path="model"):
    """Yield ``(path, dict)`` for every serialized ``RoPEConfig`` under ``node``.

    Walking the tree rather than indexing a fixed key keeps this working across model families
    (Qwen3.5's hybrid ``block.attn`` / ``block.gdn`` split vs a flat ``block``) and across the
    landmark attention variants, which nest their mixer config differently.

    :param node: A parsed ``config.json`` subtree.
    :param path: Dotted path of ``node``, used only for error messages.

    :returns: Generator of ``(dotted_path, rope_config_dict)`` pairs.
    """
    if isinstance(node, dict):
        if node.get("_CLASS_") == ROPE_CLASS:
            yield path, node
        for key, value in node.items():
            yield from find_rope_configs(value, f"{path}.{key}")
    elif isinstance(node, list):
        for i, value in enumerate(node):
            yield from find_rope_configs(value, f"{path}[{i}]")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--src", required=True, help="source step dir (has config.json + model_and_optim/)."
    )
    ap.add_argument(
        "--factor", type=float, required=True, help="YaRN context expansion multiplier."
    )
    ap.add_argument(
        "--old-context-len",
        type=int,
        default=0,
        help="the context the model was TRAINED at; YaRN's ramp is computed against it. Default 0 "
        "= read max_sequence_length from the checkpoint's train_module config, which is what the "
        "run actually trained at.",
    )
    ap.add_argument(
        "--beta-fast", type=int, default=32, help="high-frequency ramp cutoff (repo default 32)."
    )
    ap.add_argument(
        "--beta-slow", type=int, default=1, help="low-frequency ramp cutoff (repo default 1)."
    )
    ap.add_argument(
        "--dest",
        default="",
        help="destination dir. Default: <src>_yarn<factor> alongside the source.",
    )
    ap.add_argument(
        "--copy-weights",
        action="store_true",
        help="physically copy model_and_optim instead of symlinking it. Only needed when the copy "
        "must outlive the source (~57 GB for a 4B checkpoint).",
    )
    ap.add_argument("--force", action="store_true", help="overwrite an existing destination.")
    ap.add_argument("--dry-run", action="store_true", help="print the patch, write nothing.")
    args = ap.parse_args()

    src = Path(args.src).resolve()
    cfg_path = src / "config.json"
    weights = src / "model_and_optim"
    if not cfg_path.is_file():
        sys.exit(f"ERROR: no config.json under {src}")
    if not weights.is_dir():
        sys.exit(f"ERROR: no model_and_optim/ under {src}")

    with cfg_path.open() as f:
        config = json.load(f)
    if "model" not in config:
        sys.exit(f"ERROR: {cfg_path} has no 'model' key (not an olmo_core experiment config?)")

    # The trained context length. `max_sequence_length` on the train module is what the run was
    # actually trained at, which is the right YaRN reference point -- deriving it from the rung
    # label instead would silently change the ramp when the source run changes.
    old_ctx = args.old_context_len
    if not old_ctx:
        old_ctx = (config.get("train_module") or {}).get("max_sequence_length") or 0
        if not old_ctx:
            sys.exit(
                "ERROR: could not read train_module.max_sequence_length from the checkpoint; "
                "pass --old-context-len explicitly."
            )
        print(f"[yarn] trained context (train_module.max_sequence_length) = {old_ctx}")

    ropes = list(find_rope_configs(config["model"]))
    if not ropes:
        sys.exit("ERROR: no RoPEConfig found under config['model'] -- nothing to extend.")

    # An already-scaled source would compose two scalings and silently produce a model neither
    # config describes, so refuse rather than stack them.
    already = [p for p, r in ropes if r.get("scaling")]
    if already and not args.force:
        sys.exit(
            f"ERROR: {len(already)} RoPE config(s) already carry a scaling (e.g. {already[0]}). "
            "This source is already extended; re-run from the unscaled checkpoint, or pass "
            "--force if you really mean to replace the existing scaling."
        )

    scaling = {
        "factor": args.factor,
        "beta_fast": args.beta_fast,
        "beta_slow": args.beta_slow,
        "old_context_len": old_ctx,
        "_CLASS_": YARN_CLASS,
    }
    for path, rope in ropes:
        rope["scaling"] = dict(scaling)
        print(
            f"[yarn] patched {path}  (theta={rope.get('theta')}, "
            f"partial_rotary_factor={rope.get('partial_rotary_factor')})"
        )

    reach = int(old_ctx * args.factor)
    print(
        f"[yarn] factor={args.factor} x old_context_len={old_ctx} -> nominal reach {reach} tokens"
    )

    dest = Path(args.dest) if args.dest else src.parent / f"{src.name}_yarn{args.factor:g}"
    if args.dry_run:
        print(
            f"[yarn] DRY RUN: would write {dest}/config.json and link model_and_optim -> {weights}"
        )
        return

    if dest.exists():
        if not args.force:
            sys.exit(f"ERROR: {dest} already exists (pass --force to overwrite).")
        shutil.rmtree(dest)
    dest.mkdir(parents=True)

    with (dest / "config.json").open("w") as f:
        json.dump(config, f, indent=2)

    if args.copy_weights:
        shutil.copytree(weights, dest / "model_and_optim")
    else:
        # Relative link so the pair stays valid if the run dir is moved or re-mounted at a
        # different path (weka vs a local /scratch copy).
        os.symlink(os.path.relpath(weights, dest), dest / "model_and_optim")

    print(f"[yarn] wrote {dest}")
    print(f"[yarn] eval it with:  --ckpt {dest}")


if __name__ == "__main__":
    main()
