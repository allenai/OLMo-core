"""
Training options: the arguments, and nothing that needs a GPU.

Kept separate from :mod:`recipe` so the argument surface can be read, tested and reasoned about
without importing olmo-core or torch. Everything here is a plain dataclass and a parser.

The pre-migration tree had 161 launchers, most of which differed only in the values below --
20 CPT scripts of ~210 lines each, one per (architecture, learning rate, node count). These options
are chosen to be exactly the axes that actually varied.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

__all__ = ["TrainOptions", "DataSpec", "parse_data_spec", "build_parser", "options_from_args"]

#: Attention layouts a run can request. ``full`` is plain causal; the rest are the document-aware
#: masks. The data must have been converted to match -- ``landmark`` needs landmark-emit shards,
#: the others dense-emit -- which is why the format fingerprint is checked at step 0.
ARCHITECTURES = ("full", "chunked", "chunked-mix", "hierarchical", "landmark")


@dataclass(frozen=True)
class DataSpec:
    """
    One shard directory and its mixing weight.

    :param path: Directory holding ``token_ids_part_*.npy`` (and, for SFT, ``labels_mask_*.npy``).
    :param weight: Relative sampling weight. Normalised across all specs, so the numbers are ratios
        rather than probabilities -- ``2`` and ``1`` mean the same as ``0.667`` and ``0.333``.
    :param label: Name for metrics; defaults to the directory's basename.
    """

    path: str
    weight: float = 1.0
    label: Optional[str] = None

    @property
    def name(self) -> str:
        """:returns: The metrics label."""
        return self.label or self.path.rstrip("/").rsplit("/", 1)[-1]


def parse_data_spec(spec: str) -> DataSpec:
    """
    Parse one ``--data`` value: ``path``, ``path:weight``, or ``path:weight:label``.

    :param spec: The raw argument.

    :returns: The parsed spec.

    :raises ValueError: If the weight is not a positive number.
    """
    path, _, rest = spec.partition(":")
    if not rest:
        return DataSpec(path=path)
    weight_str, _, label = rest.partition(":")
    try:
        weight = float(weight_str)
    except ValueError:
        raise ValueError(
            f"bad --data {spec!r}: expected path, path:weight or path:weight:label"
        ) from None
    if weight <= 0:
        raise ValueError(f"bad --data {spec!r}: weight must be positive, got {weight}")
    return DataSpec(path=path, weight=weight, label=label or None)


@dataclass
class TrainOptions:
    """
    Everything a run varies. One object, passed to :mod:`recipe`.

    :param run_name: Names the run, the save folder and the W&B group.
    :param data: Shard directories with mixing weights.
    :param base: Checkpoint to start from. ``None`` trains from scratch, which for these
        experiments is almost always a mistake -- say it explicitly with ``--from-scratch``.
    :param arch: One of :data:`ARCHITECTURES`.
    :param model: ``TransformerConfig`` factory name, e.g. ``qwen3_4B``.
    :param tokenizer: Tokenizer family for vocab sizing and the reserved ids.
    :param seq_len: Instance length. For ``landmark`` it must be a multiple of ``mem_freq + 1``.
    :param mem_freq: Landmark spacing; block size is ``mem_freq + 1``.
    :param mix_start_p: (``chunked-mix``) curriculum start: the per-example probability of
        collapsing to plain causal on the first forward. The reference runs used 0.80.
    :param mix_end_p: (``chunked-mix``) curriculum end probability, reached on the last
        forward. The reference runs annealed to 0.0.
    :param mix_seed: (``chunked-mix``) seed for the per-example collapse draws.
    :param lr: Peak learning rate.
    :param warmup_fraction: Fraction of the run spent warming up.
    :param max_steps: Stop after this many steps. Mutually exclusive with ``max_tokens``.
    :param max_tokens: Stop after this many tokens. The natural budget for CPT.
    :param nodes: Node count; world size is ``nodes * gpus_per_node``.
    :param gpus_per_node: GPUs per node.
    :param save_folder: Where checkpoints go. Defaults to a cluster-appropriate root.
    :param save_interval: Permanent checkpoint interval, in steps.
    :param ephemeral_save_interval: Rolling checkpoint interval, in steps.
    :param seed: Data-order seed.
    :param cluster: Beaker cluster to launch on. ``None`` means run locally under torchrun.
    :param wandb_project: W&B project; empty disables W&B.
    :param fingerprint: Write the training data's format fingerprint into every checkpoint. On by
        default -- without it a checkpoint's eval format cannot be verified later.
    :param overrides: Dotted ``key=value`` overrides applied to the assembled config last.
    """

    run_name: str
    data: List[DataSpec] = field(default_factory=list)
    base: Optional[str] = None
    arch: str = "full"
    model: str = "qwen3_4B"
    tokenizer: str = "qwen3"
    seq_len: int = 40960
    mem_freq: int = 63
    mix_start_p: float = 0.80
    mix_end_p: float = 0.0
    mix_seed: int = 0
    lr: float = 1e-5
    warmup_fraction: float = 0.03
    max_steps: Optional[int] = None
    max_tokens: Optional[int] = None
    nodes: int = 1
    gpus_per_node: int = 8
    save_folder: Optional[str] = None
    save_interval: int = 100_000
    ephemeral_save_interval: int = 500
    seed: int = 34521
    cluster: Optional[str] = None
    wandb_project: str = "memory-networks"
    fingerprint: bool = True
    overrides: List[str] = field(default_factory=list)

    #: Set by the entry point, not the user: SFT trains on answer tokens only and pads one example
    #: per window; CPT trains on everything and packs.
    mode: str = "sft"

    def __post_init__(self) -> None:
        if self.arch not in ARCHITECTURES:
            raise ValueError(f"--arch must be one of {ARCHITECTURES}, got {self.arch!r}")
        if not self.data:
            raise ValueError("at least one --data shard directory is required")
        if (self.max_steps is None) == (self.max_tokens is None):
            raise ValueError("pass exactly one of --max-steps or --max-tokens")
        if self.arch == "landmark" and self.seq_len % (self.mem_freq + 1):
            raise ValueError(
                f"--seq-len {self.seq_len} must be a multiple of the landmark block size "
                f"(mem-freq+1 = {self.mem_freq + 1})"
            )
        # olmo-core rejects this too, but only once the trainer config is assembled -- which is
        # after the model is built on a GPU. Lowering --save-interval for a short run and leaving
        # the ephemeral default at 500 is the natural way to hit it, so it is checked here, for
        # free, before anything is scheduled.
        if self.ephemeral_save_interval >= self.save_interval:
            raise ValueError(
                f"--ephemeral-save-interval {self.ephemeral_save_interval} must be less than "
                f"--save-interval {self.save_interval}; the rolling checkpoint is meant to be the "
                f"more frequent of the two"
            )

    @property
    def world_size(self) -> int:
        """:returns: Total GPUs."""
        return self.nodes * self.gpus_per_node

    @property
    def global_batch_tokens(self) -> int:
        """:returns: Tokens per optimizer step -- one instance per GPU, gradient accumulation 1."""
        return self.world_size * self.seq_len

    @property
    def is_local(self) -> bool:
        """:returns: True when this runs under torchrun rather than Beaker."""
        return self.cluster is None

    def weights(self) -> List[Tuple[DataSpec, float]]:
        """
        :returns: Each spec with its normalised ratio.
        """
        total = sum(d.weight for d in self.data)
        return [(d, d.weight / total) for d in self.data]


def build_parser(description: str, *, mode: str) -> argparse.ArgumentParser:
    """
    The shared argument surface for both entry points.

    :param description: Help text for this entry point.
    :param mode: ``"sft"`` or ``"cpt"``; only changes defaults, not the options.

    :returns: The parser.
    """
    p = argparse.ArgumentParser(
        description=description, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("run_name", help="names the run, its save folder and its W&B group")
    p.add_argument(
        "--data",
        action="append",
        required=True,
        metavar="DIR[:WEIGHT[:LABEL]]",
        help="shard directory, repeatable. Weights are ratios and are normalised, so "
        "'--data a:2 --data b:1' mixes 2:1.",
    )
    p.add_argument("--base", help="checkpoint to start from")
    p.add_argument(
        "--from-scratch",
        action="store_true",
        help="train from random init. Required to omit --base, because doing so silently is almost "
        "never what anyone means here.",
    )
    p.add_argument("--arch", default="full", choices=ARCHITECTURES)
    p.add_argument("--model", default="qwen3_4B", help="TransformerConfig factory name")
    p.add_argument("--tokenizer", default="qwen3", choices=("qwen3", "qwen3_5"))
    p.add_argument("--seq-len", type=int, default=40960)
    p.add_argument("--mem-freq", type=int, default=63, help="(landmark) block = mem-freq + 1")
    p.add_argument(
        "--mix-start-p",
        type=float,
        default=0.80,
        help="(chunked-mix) curriculum start collapse probability (default 0.80, the "
        "reference runs' value)",
    )
    p.add_argument(
        "--mix-end-p",
        type=float,
        default=0.0,
        help="(chunked-mix) curriculum end collapse probability (default 0.0)",
    )
    p.add_argument("--mix-seed", type=int, default=0, help="(chunked-mix) collapse-draw seed")
    p.add_argument("--lr", type=float, default=1e-5 if mode == "sft" else 1e-4)
    p.add_argument("--warmup-fraction", type=float, default=0.03)
    p.add_argument("--max-steps", type=int)
    p.add_argument("--max-tokens", type=int)
    p.add_argument("--nodes", type=int, default=1)
    p.add_argument("--gpus-per-node", type=int, default=8)
    p.add_argument("--save-folder")
    p.add_argument("--save-interval", type=int, default=100_000)
    p.add_argument("--ephemeral-save-interval", type=int, default=500)
    p.add_argument("--seed", type=int, default=34521)
    p.add_argument(
        "--cluster",
        help="Beaker cluster (e.g. ai2/jupiter-cirrascale-2). Omit to run locally under torchrun.",
    )
    p.add_argument("--wandb-project", default="memory-networks", help="empty string disables W&B")
    p.add_argument(
        "--no-fingerprint",
        action="store_true",
        help="do not write the training data's format fingerprint into checkpoints. Only for a "
        "throwaway run: without it, eval cannot verify what format this checkpoint expects.",
    )
    p.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="dotted config override, applied last, e.g. --override train_module.optim.lr=6e-5",
    )
    return p


def options_from_args(args: argparse.Namespace, *, mode: str) -> TrainOptions:
    """
    Turn parsed arguments into :class:`TrainOptions`.

    :param args: Parsed arguments.
    :param mode: ``"sft"`` or ``"cpt"``.

    :returns: The options.

    :raises SystemExit: If ``--base`` is missing without ``--from-scratch``.
    """
    if not args.base and not args.from_scratch:
        raise SystemExit(
            "no --base checkpoint given. Pass one, or --from-scratch to say you meant it. Note that "
            "any Qwen3 base needs its marker embeddings repaired before document-chunked training: "
            "the reserved marker rows are bit-identical out of the box, so the model cannot tell an "
            "open-document marker from a close one."
        )
    return TrainOptions(
        run_name=args.run_name,
        data=[parse_data_spec(d) for d in args.data],
        base=args.base,
        arch=args.arch,
        model=args.model,
        tokenizer=args.tokenizer,
        seq_len=args.seq_len,
        mem_freq=args.mem_freq,
        mix_start_p=args.mix_start_p,
        mix_end_p=args.mix_end_p,
        mix_seed=args.mix_seed,
        lr=args.lr,
        warmup_fraction=args.warmup_fraction,
        max_steps=args.max_steps,
        max_tokens=args.max_tokens,
        nodes=args.nodes,
        gpus_per_node=args.gpus_per_node,
        save_folder=args.save_folder,
        save_interval=args.save_interval,
        ephemeral_save_interval=args.ephemeral_save_interval,
        seed=args.seed,
        cluster=args.cluster,
        wandb_project=args.wandb_project,
        fingerprint=not args.no_fingerprint,
        overrides=list(args.override),
        mode=mode,
    )


def describe(options: TrainOptions) -> str:
    """
    :param options: The resolved options.

    :returns: A short human summary, printed before a run so a launch is auditable from the log.
    """
    budget = f"{options.max_steps} steps" if options.max_steps else f"{options.max_tokens:,} tokens"
    mix = "  ".join(f"{d.name}={r:.3f}" for d, r in options.weights())
    where = f"beaker:{options.cluster}" if options.cluster else "local (torchrun)"
    return (
        f"{options.run_name}  [{options.mode}]\n"
        f"  arch      {options.arch} / {options.model} / seq_len {options.seq_len}\n"
        f"  base      {options.base or 'RANDOM INIT'}\n"
        f"  data      {mix}\n"
        f"  budget    {budget} at {options.world_size} GPU(s), "
        f"{options.global_batch_tokens:,} tok/step\n"
        f"  lr        {options.lr:g} (warmup {options.warmup_fraction:.0%})\n"
        f"  where     {where}\n"
        f"  fingerprint {'on' if options.fingerprint else 'OFF'}"
    )
