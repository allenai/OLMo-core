"""
Assemble the exact multi-rank run on CPU, without a GPU, and build the data source.

horton is contended, so a 2-GPU slot is worth minutes of waiting; spending one on a
``TypeError`` in config assembly is the avoidable version of the 2026-08-11 failure, where
``data_loader.build(dataset)`` died *after* the model was already on the card. Everything up to the
model build is CPU work and can be checked for free while the job queues.

Usage::

    python debug/train_smoke/preflight_configs.py --data <shard dir> --base <repaired base>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
for _src in (_REPO / "src", _REPO / "ctc" / "src", _REPO / "src" / "scripts" / "ctc" / "train"):
    if _src.is_dir() and str(_src) not in sys.path:
        sys.path.insert(0, str(_src))

from options import DataSpec, TrainOptions  # noqa: E402
from recipe import build_experiment  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--work-dir", default="/data/prasann/ctc_work")
    args = ap.parse_args()

    options = TrainOptions(
        run_name="preflight",
        data=[DataSpec(path=args.data)],
        base=args.base,
        arch="chunked",
        model="qwen3_0_6B",
        tokenizer="qwen3",
        seq_len=2048,
        max_steps=30,
        gpus_per_node=2,
        lr=1e-5,
        save_folder="/data/prasann/ctc_smoke/ckpt_preflight",
        save_interval=30,
        ephemeral_save_interval=10,
        wandb_project="",
        mode="sft",
    )
    model, train_module, dataset, data_loader, trainer_cfg = build_experiment(
        options, save_folder=str(options.save_folder), work_dir=args.work_dir
    )
    print("model vocab_size      :", model.vocab_size)
    print("document_chunk_attn   :", model.document_chunk_attention)
    print("dp shard_degree       :", train_module.dp_config.shard_degree)
    print("trainer load_path     :", trainer_cfg.load_path)
    print("trainer load_strategy :", trainer_cfg.load_strategy)

    # The 2026-08-11 defect: the loader takes BUILT sources, not their configs, and the failure
    # surfaced only after the model was on the GPU. Build it here instead.
    source = dataset.build(args.work_dir)
    print("instances             :", len(source))
    loader = data_loader.build(source)
    print("loader                :", type(loader).__name__)
    print("PREFLIGHT OK")


if __name__ == "__main__":
    main()
