"""
Analysis hook for logging which landmark gates hard top-k retrieval opens at decode time.

Landmark attention divides the sequence into blocks of ``mem_freq + 1`` tokens, each ending in a
landmark ("memory") token whose attention weight gates access to that block's content. At decode
time, the paper's inference procedure (and this code's ``landmark_top_k_blocks`` option) keeps only
the ``top_k`` highest-scoring landmark blocks per head and gives every other past block exactly zero
weight (see ``_apply_topk_landmark_retrieval`` in the landmark mixers). This module records, for
every decode step (= one generated token), every layer, and every head, the set of past landmark
**blocks** whose gate was kept open.

The hook is *off* unless the environment variable ``OLMO_LANDMARK_GATE_LOG`` is set to an output
path; when off, :func:`is_enabled` is the only thing the attention modules ever touch (a cheap
boolean), so the hot path is unaffected. When on, one JSON object per decoded token is appended to
the output file (JSONL — load with ``pandas.read_json(path, lines=True)``)::

    {"dataset": "ruler", "doc_id": 0, "context_len": 4096, "subtask": "niah_single_1",
     "decoded_token_num": 1,
     "layers": {"layer0": {"head0": [3, 17, 42], "head1": [...]}, "layer1": {...}, ...}}

``decoded_token_num`` starts at 1 for the first generated token of an example. The ints under each
head are 0-based landmark-**block** ordinals (block ``b`` is the ``b``-th ``mem_freq + 1``-token
block; its landmark sits at the block's last position).

Per-example metadata is read from the environment (the harness sets these per eval process)::

    OLMO_LANDMARK_GATE_LOG   output path (presence enables logging)
    OLMO_GATE_DATASET        "dataset" field (default "ruler")
    OLMO_GATE_SUBTASK        "subtask" field (default "")
    OLMO_GATE_CONTEXT_LEN    "context_len" field (default: the content prompt length)
    OLMO_GATE_DOC_ID         "doc_id" field -- the eval harness's document id for the example
                             (default: a per-process 0-based counter when unset)

``doc_id`` identifies the example: the eval harness sets ``OLMO_GATE_DOC_ID`` per example (oe-eval
passes each request's ``doc_id``); when unset it falls back to a per-process 0-based counter bumped on
each :func:`start_example`. Under multi-GPU eval the output path is suffixed per worker
(``.rank{RANK}`` under torchrun, else ``.pid{PID}``) so workers never clobber each other's file. The
hook assumes ``batch_size == 1`` (which landmark generation already requires).
"""

import json
import os
import threading
from typing import Dict, List, Optional

import torch

__all__ = [
    "is_enabled",
    "start_example",
    "record_layer",
    "finalize_token",
    "end_example",
    "close",
]

_lock = threading.Lock()
_initialized = False
_enabled = False
_path: Optional[str] = None
_file = None

_dataset = "ruler"
_subtask = ""
_context_len_env: Optional[int] = None
_doc_id_env: Optional[str] = None

_doc_counter = -1
_doc_id: object = 0
_context_len = 0
_decoded_token_num = 0
_current_layers: Dict[str, Dict[str, List[int]]] = {}


def _init() -> None:
    global _initialized, _enabled, _path, _file
    global _dataset, _subtask, _context_len_env
    with _lock:
        if _initialized:
            return
        _initialized = True
        path = os.environ.get("OLMO_LANDMARK_GATE_LOG")
        if not path:
            _enabled = False
            return
        # Per-worker output so parallel eval workers never clobber the same file. Use the torchrun
        # rank when present, else the pid (oe-eval spawns one model process per GPU without setting
        # RANK, so the pid is what guarantees uniqueness there).
        rank = os.environ.get("RANK") or os.environ.get("LOCAL_RANK")
        suffix = f"rank{rank}" if rank else f"pid{os.getpid()}"
        path = f"{path}.{suffix}"
        directory = os.path.dirname(os.path.abspath(path))
        os.makedirs(directory, exist_ok=True)
        _file = open(path, "w", buffering=1)  # line-buffered so a killed job keeps finished rows
        _path = path
        _enabled = True


def is_enabled() -> bool:
    """Whether landmark gate logging is active (``OLMO_LANDMARK_GATE_LOG`` set)."""
    if not _initialized:
        _init()
    return _enabled


def _read_env_metadata() -> None:
    """Refresh the per-example metadata from the environment (cheap; allows per-example updates)."""
    global _dataset, _subtask, _context_len_env, _doc_id_env
    _dataset = os.environ.get("OLMO_GATE_DATASET", "ruler")
    _subtask = os.environ.get("OLMO_GATE_SUBTASK", "")
    cl = os.environ.get("OLMO_GATE_CONTEXT_LEN")
    _context_len_env = int(cl) if cl else None
    did = os.environ.get("OLMO_GATE_DOC_ID")
    _doc_id_env = did if did else None


def start_example(content_prompt_len: int) -> None:
    """Begin a new example: resolve ``doc_id``, reset ``decoded_token_num``, refresh metadata.

    :param content_prompt_len: Length of the content (non-landmark) prompt, used as ``context_len``
        unless ``OLMO_GATE_CONTEXT_LEN`` overrides it.
    """
    global _doc_counter, _doc_id, _decoded_token_num, _context_len, _current_layers
    if not is_enabled():
        return
    _read_env_metadata()
    _doc_counter += 1
    if _doc_id_env is None:
        _doc_id = _doc_counter
    else:
        # The harness doc_id is an int; keep it as one for clean JSON, else fall back to the string.
        _doc_id = int(_doc_id_env) if _doc_id_env.lstrip("-").isdigit() else _doc_id_env
    _decoded_token_num = 0
    _context_len = _context_len_env if _context_len_env is not None else int(content_prompt_len)
    _current_layers = {}


def record_layer(layer_idx: Optional[int], keep: torch.Tensor, block_ids: torch.Tensor) -> None:
    """Record the open landmark gates for one layer at the current decode step.

    :param layer_idx: The model layer index (used as the ``"layer{idx}"`` key).
    :param keep: Boolean tensor of shape ``(B, H, 1, M)`` where ``keep[0, h, 0, m]`` is ``True`` iff
        gate slot ``m`` is kept open for head ``h``. ``B`` must be 1.
    :param block_ids: Long tensor of shape ``(M,)`` mapping each gate slot to its landmark-block
        ordinal (so multiple slots may share a block ordinal, e.g. sparse-landmark chunks).
    """
    if not _enabled:
        return
    keep_c = keep.detach().to("cpu", non_blocking=False).reshape(keep.shape[0], keep.shape[1], -1)
    keep_c = keep_c.bool()
    block_ids_c = block_ids.detach().to("cpu").reshape(-1)
    B, H = keep_c.shape[0], keep_c.shape[1]
    assert B == 1, "landmark gate logging assumes batch_size == 1"
    heads: Dict[str, List[int]] = {}
    for h in range(H):
        sel = block_ids_c[keep_c[0, h]]
        heads[f"head{h}"] = sorted({int(b) for b in sel.tolist()})
    _current_layers[f"layer{layer_idx}"] = heads


def _layer_sort_key(name: str) -> int:
    try:
        return int(name[len("layer") :])
    except ValueError:
        return 1 << 30


def finalize_token() -> None:
    """Flush the accumulated per-layer gates as one record for the just-decoded token.

    A no-op if no layer recorded anything this step (e.g. a prefill forward, which never applies
    top-k retrieval), so ``decoded_token_num`` only advances on real single-token decode steps.
    """
    global _decoded_token_num, _current_layers
    if not _enabled or not _current_layers:
        return
    _decoded_token_num += 1
    record = {
        "dataset": _dataset,
        "doc_id": _doc_id,
        "context_len": _context_len,
        "subtask": _subtask,
        "decoded_token_num": _decoded_token_num,
        "layers": {
            name: _current_layers[name] for name in sorted(_current_layers, key=_layer_sort_key)
        },
    }
    assert _file is not None
    _file.write(json.dumps(record) + "\n")
    _current_layers = {}


def end_example() -> None:
    """Discard any partially-accumulated (unflushed) layer state at the end of an example."""
    global _current_layers
    if not _enabled:
        return
    _current_layers = {}


def close() -> None:
    """Close the output file (flushing). Safe to call when disabled."""
    global _file
    if _file is not None:
        _file.close()
        _file = None
