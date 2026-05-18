#!/usr/bin/env python
import argparse
import multiprocessing
from pathlib import Path
from typing import Iterable, Sequence

from olmo_core.data.misc.code_fresh_export import (
    CODE_FRESH_LANGUAGES,
    build_documents_and_stats,
    default_num_procs,
    flatten_documents,
    get_export_dir_name,
    write_document_metadata,
    write_memmap,
    write_stats,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export Code Fresh as native olmo-core perplexity artifacts."
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Root directory under which eval-data/perplexity/... will be written.",
    )
    parser.add_argument(
        "--tokenizer",
        default="allenai/dolma2-tokenizer",
        help="HF tokenizer identifier used to tokenize stripped file contents.",
    )
    parser.add_argument(
        "--languages",
        nargs="+",
        default=list(CODE_FRESH_LANGUAGES),
        help="Subset of Code Fresh languages to export.",
    )
    parser.add_argument(
        "--num-procs",
        type=int,
        default=default_num_procs(),
        help="Number of language workers to run in parallel.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing shard directories.",
    )
    parser.add_argument(
        "--max-doc-tokens-before-eos",
        type=int,
        default=8191,
        help="Maximum number of scored tokens before appending terminal EOS.",
    )
    return parser.parse_args()


def _export_language(
    language: str,
    output_root: Path,
    tokenizer_name: str,
    overwrite: bool,
    max_doc_tokens_before_eos: int,
) -> None:
    from datasets import load_dataset
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    export_dir = (
        output_root
        / "eval-data"
        / "perplexity"
        / get_export_dir_name(tokenizer_name)
        / language
        / "val"
    )
    token_path = export_dir / "part-0-00000.npy"
    metadata_path = export_dir / "part-0-00000.csv.gz"
    stats_path = export_dir / "stats.json"

    if export_dir.exists() and not overwrite:
        raise FileExistsError(
            f"output directory '{export_dir}' already exists; pass --overwrite to replace it"
        )
    export_dir.mkdir(parents=True, exist_ok=True)

    rows: Iterable[dict] = load_dataset("allenai/code_fresh_0825_1225", language, split="train")
    docs, stats = build_documents_and_stats(
        rows,
        tokenizer,
        language=language,
        max_doc_tokens_before_eos=max_doc_tokens_before_eos,
    )
    tokens, offsets = flatten_documents(docs)
    write_memmap(token_path, tokens)
    write_document_metadata(metadata_path, offsets)
    write_stats(stats_path, stats)


def main() -> None:
    args = _parse_args()
    languages: Sequence[str] = tuple(args.languages)
    for lang in languages:
        if lang not in CODE_FRESH_LANGUAGES:
            raise ValueError(f"unknown Code Fresh language '{lang}'")

    worker_args = [
        (
            language,
            args.output_root,
            args.tokenizer,
            args.overwrite,
            args.max_doc_tokens_before_eos,
        )
        for language in languages
    ]

    if args.num_procs == 1:
        for worker_arg in worker_args:
            _export_language(*worker_arg)
    else:
        with multiprocessing.Pool(processes=args.num_procs) as pool:
            pool.starmap(_export_language, worker_args)


if __name__ == "__main__":
    main()
