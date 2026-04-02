import gzip
import json
import multiprocessing
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

from ..tokenizer import TokenizerLike

CODE_FRESH_LANGUAGES: Tuple[str, ...] = (
    "blade",
    "c",
    "clojure",
    "common_lisp",
    "cpp",
    "csharp",
    "css",
    "dart",
    "erlang",
    "fortran",
    "go",
    "haskell",
    "html",
    "java",
    "java_server_page",
    "javascript",
    "julia",
    "kotlin",
    "lua",
    "markdown",
    "mathematica",
    "matlab",
    "objective_c",
    "objective_cpp",
    "ocaml",
    "perl",
    "php",
    "powershell",
    "python",
    "restructuredtext",
    "ruby",
    "rust",
    "scala",
    "scheme",
    "swift",
    "systemverilog",
    "tcl",
    "tex",
    "typescript",
    "verilog",
    "vhdl",
    "vue",
)


@dataclass
class ExportStats:
    language: str
    num_docs: int = 0
    total_tokens: int = 0
    max_doc_tokens_before_eos: int = 0
    max_doc_tokens_after_eos: int = 0
    num_skipped_empty: int = 0


def process_code_fresh_file_contents(
    file_contents: str,
    tokenizer: TokenizerLike,
    *,
    max_doc_tokens_before_eos: int = 8191,
) -> Optional[np.ndarray]:
    text = file_contents.strip()
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if not token_ids:
        return None
    if len(token_ids) > max_doc_tokens_before_eos:
        raise ValueError(
            f"document tokenized to {len(token_ids)} tokens before EOS, "
            f"which exceeds the cap of {max_doc_tokens_before_eos}"
        )
    return np.asarray([*token_ids, tokenizer.eos_token_id], dtype=np.uint32)


def build_documents_and_stats(
    rows: Iterable[dict],
    tokenizer: TokenizerLike,
    *,
    language: str,
    max_doc_tokens_before_eos: int = 8191,
) -> Tuple[List[np.ndarray], ExportStats]:
    docs: List[np.ndarray] = []
    stats = ExportStats(language=language)
    for row in rows:
        doc = process_code_fresh_file_contents(
            row["file_contents"],
            tokenizer,
            max_doc_tokens_before_eos=max_doc_tokens_before_eos,
        )
        if doc is None:
            stats.num_skipped_empty += 1
            continue
        doc_len_before_eos = len(doc) - 1
        stats.num_docs += 1
        stats.total_tokens += len(doc)
        stats.max_doc_tokens_before_eos = max(stats.max_doc_tokens_before_eos, doc_len_before_eos)
        stats.max_doc_tokens_after_eos = max(stats.max_doc_tokens_after_eos, len(doc))
        docs.append(doc)
    return docs, stats


def write_memmap(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        mmap = np.memmap(tmp_path, mode="w+", dtype=arr.dtype, shape=arr.shape)
        mmap[:] = arr
        mmap.flush()
        del mmap
        os.replace(tmp_path, path)
    finally:
        tmp_path.unlink(missing_ok=True)


def write_document_metadata(path: Path, offsets: Sequence[Tuple[int, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        with gzip.open(tmp_path, "wt") as f:
            for start, end in offsets:
                f.write(f"{start},{end}\n")
        os.replace(tmp_path, path)
    finally:
        tmp_path.unlink(missing_ok=True)


def write_stats(path: Path, stats: ExportStats) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False, mode="w") as tmp:
        json.dump(asdict(stats), tmp, indent=2, sort_keys=True)
        tmp.flush()
        tmp_path = Path(tmp.name)
    try:
        os.replace(tmp_path, path)
    finally:
        tmp_path.unlink(missing_ok=True)


def flatten_documents(docs: Sequence[np.ndarray]) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    offsets: List[Tuple[int, int]] = []
    start = 0
    for doc in docs:
        end = start + len(doc)
        offsets.append((start, end))
        start = end

    if docs:
        tokens = np.concatenate(docs).astype(np.uint32, copy=False)
    else:
        tokens = np.asarray([], dtype=np.uint32)
    return tokens, offsets


def get_export_dir_name(tokenizer_name: str) -> str:
    return f"code_fresh_0825_1225_{tokenizer_name.rsplit('/', 1)[-1]}"


def default_num_procs() -> int:
    return max(1, multiprocessing.cpu_count() - 1)
