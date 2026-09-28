"""Eval-suite decontamination lists: parsing, and the MMFineReason index filter."""

import json

import numpy as np
from datasets import Dataset

from olmo_core.data.multimodal.mmfinereason import (
    MMFineReasonDataset,
    MMFineReasonDatasetConfig,
)
from olmo_core.data.multimodal.sft_common import load_exclude_ids


def test_load_exclude_ids_accepts_dict_or_list_and_stringifies(tmp_path):
    p = tmp_path / "ex.json"
    p.write_text(json.dumps({"ids": [1, "2", "abc"], "note": "ignored"}))
    assert load_exclude_ids(str(p)) == {"1", "2", "abc"}
    p.write_text(json.dumps([3, 4]))
    assert load_exclude_ids(str(p)) == {"3", "4"}


def test_mmfinereason_index_drops_listed_ids(tmp_path):
    p = tmp_path / "ex.json"
    p.write_text(json.dumps({"ids": ["r1", "r3"]}))
    cfg = MMFineReasonDatasetConfig(exclude_ids_path=str(p))
    ds = MMFineReasonDataset.__new__(MMFineReasonDataset)
    ds.config = cfg
    ds._data = Dataset.from_dict(
        {
            "id": ["r0", "r1", "r2", "r3"],
            "source": ["a", "a", "b", "b"],
            "pass_rate": [0.1, 0.2, 0.3, 0.4],
            "is_consistent": [True, True, False, True],
        }
    )
    index = ds._build_index()
    assert index is not None
    np.testing.assert_array_equal(index, np.array([0, 2]))

    # combines with the existing filters rather than replacing them
    ds.config = MMFineReasonDatasetConfig(exclude_ids_path=str(p), sources=["b"])
    np.testing.assert_array_equal(ds._build_index(), np.array([2]))

    # no filters at all -> direct indexing
    ds.config = MMFineReasonDatasetConfig()
    assert ds._build_index() is None
