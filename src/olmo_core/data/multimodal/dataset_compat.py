"""Compatibility loader for HuggingFace datasets saved with deprecated ``List`` features.

PixMo and Tulu datasets on weka were materialized with ``datasets`` 5.x, which uses a
legacy ``List`` feature type that ``datasets`` 3.6 (olmo-core env) cannot parse from
``dataset_info.json``.  Loading the Arrow streams directly avoids the schema mismatch.
"""

from __future__ import annotations

import glob
import json
import os
from pathlib import Path
from typing import Any, Union

_LIST_PATCHED = False


def _patch_list_feature_type() -> None:
    global _LIST_PATCHED
    if _LIST_PATCHED:
        return
    from datasets.features.features import Sequence, _FEATURE_TYPES

    _FEATURE_TYPES["List"] = Sequence
    _LIST_PATCHED = True


def _has_list_feature(path: Path) -> bool:
    info = path / "dataset_info.json"
    if not info.exists():
        return False
    text = info.read_text()
    return '"_type": "List"' in text or '"_type":"List"' in text


def _load_arrow_split(split_dir: Path):
    import pyarrow as pa
    from datasets import Dataset

    _patch_list_feature_type()
    arrow_files = sorted(glob.glob(str(split_dir / "*.arrow")))
    arrow_files = [f for f in arrow_files if not os.path.basename(f).startswith("cache-")]
    if not arrow_files:
        raise FileNotFoundError(f"No arrow files in {split_dir}")
    tables = []
    for f in arrow_files:
        with open(f, "rb") as fh:
            tables.append(pa.ipc.open_stream(fh).read_all())
    table = tables[0] if len(tables) == 1 else pa.concat_tables(tables)
    return Dataset(table)


def load_from_disk_compat(path: Union[str, os.PathLike], **kwargs: Any):
    """Load a HuggingFace dataset from disk, handling legacy ``List`` feature schemas."""
    path = Path(path)
    dict_json = path / "dataset_dict.json"
    if dict_json.exists():
        from datasets import DatasetDict

        meta = json.loads(dict_json.read_text())
        splits = meta.get("splits", [])
        out = {}
        for split in splits:
            split_dir = path / split
            if _has_list_feature(split_dir):
                out[split] = _load_arrow_split(split_dir)
            else:
                from datasets import load_from_disk

                return load_from_disk(str(path), **kwargs)
        return DatasetDict(out)

    if _has_list_feature(path):
        return _load_arrow_split(path)

    _patch_list_feature_type()
    from datasets import load_from_disk

    return load_from_disk(str(path), **kwargs)
