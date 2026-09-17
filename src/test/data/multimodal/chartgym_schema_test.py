"""ChartGym's staged corpus must load through `FineVisionDatasetConfig` unchanged.

ChartGym deliberately ships no loader of its own -- it is staged in FineVision's schema so
the existing loader reads it. That only holds if the on-disk layout matches exactly, and two
of the ways it can fail are silent:

* ``Features({"texts": Sequence({...})})`` transposes a list-of-structs into a dict-of-lists,
  which `FineVisionDataset._build` cannot iterate. Staging has to use a list literal.
* ``FineVisionDatasetConfig`` has no ``skip_overlong``, so an over-budget row is
  right-truncated by ``truncate_example`` -- dropping the LAST questions of a 16-question
  figure without any error.

The assertion that matters most here is the amortization claim: N questions must produce
exactly ONE image block, because that is what makes many-questions-per-chart free.
"""

from __future__ import annotations

import io
import zlib
from typing import Dict, List

import numpy as np
import pytest

from olmo_core.data.multimodal import FineVisionDatasetConfig
from olmo_core.nn.vision.molmo2_tokens import IM_PATCH_ID


class _FakeTokenizer:
    """Same stand-in as ``sft_datasets_test``: one id per whitespace word, CRC32-stable."""

    eos_token_id = 151643
    bos_token_id = None

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        assert tokenize is False and add_generation_prompt is True
        return f"<|im_start|>user\n{messages[0]['content']}<|im_end|>\n<|im_start|>assistant\n"

    def encode(self, text, add_special_tokens=True, split_special_tokens=False):
        return [(zlib.crc32(w.encode()) % 10000) + 100 for w in text.split()]


def _png(seed: int = 0) -> bytes:
    from PIL import Image

    rng = np.random.default_rng(seed)
    img = Image.fromarray(rng.integers(0, 255, (64, 96, 3), dtype=np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _stage(tmp_path, n_questions: int, n_rows: int = 2):
    from datasets import Dataset, Features, Image, Value

    rows: List[Dict] = []
    for r in range(n_rows):
        rows.append({
            "images": [{"bytes": _png(r), "path": f"cg-{r}.png"}],
            "texts": [
                {"user": f"How many labelled ticks are on the vertical axis of chart {r}?",
                 "assistant": str(3 + q)}
                for q in range(n_questions)
            ],
            "figure_id": f"cg-{r}", "difficulty": "medium",
            "families": ["cnt.ticks_axis"] * n_questions,
            "capabilities": ["counting"] * n_questions,
        })
    feats = Features({
        "images": [Image(decode=False)],
        "texts": [{"user": Value("string"), "assistant": Value("string")}],
        "figure_id": Value("string"), "difficulty": Value("string"),
        "families": [Value("string")], "capabilities": [Value("string")],
    })
    out = tmp_path / "chartgym-train"
    Dataset.from_list(rows, features=feats).save_to_disk(str(out))
    return out


def _build(path, **kwargs):
    cfg = FineVisionDatasetConfig(
        dataset_path=str(path),
        index_cache_dir="",
        max_crops=kwargs.pop("max_crops", 2),
        max_sequence_length=kwargs.pop("max_sequence_length", 4096),
        **kwargs,
    )
    return cfg.build(_FakeTokenizer())


def test_staged_chartgym_loads_through_finevision(tmp_path):
    ds = _build(_stage(tmp_path, n_questions=16))
    assert len(ds) == 2
    ex = ds[0]
    assert ex["input_ids"].ndim == 1
    assert ex["loss_masks"].sum() > 0, "no supervised tokens"


def test_sixteen_questions_share_exactly_one_image_block(tmp_path):
    """The amortization claim, asserted rather than assumed.

    If each question got its own image block, a 16-question figure would cost 16 image
    encodes and 16x the crops, and packing (which is crop-bound on this tier, ~31% token
    occupancy) would collapse.
    """
    one = _build(_stage(tmp_path / "a", n_questions=1))[0]
    many = _build(_stage(tmp_path / "b", n_questions=16))[0]
    patches_one = int((one["input_ids"] == IM_PATCH_ID).sum())
    patches_many = int((many["input_ids"] == IM_PATCH_ID).sum())
    assert patches_one > 0
    assert patches_many == patches_one, (
        f"16 questions produced {patches_many} image patch tokens vs {patches_one} for one "
        "question; the branches are not sharing the image prefix"
    )
    # ...while the supervised span does grow with the number of questions.
    assert many["loss_masks"].sum() > one["loss_masks"].sum()


def test_branch_weighting_is_sublinear_in_question_count(tmp_path):
    """`root_subsegments_root_tokens` must stop a 16-question figure carrying 16x the
    gradient of a 1-question figure."""
    one = _build(_stage(tmp_path / "a", n_questions=1))[0]
    many = _build(_stage(tmp_path / "b", n_questions=16))[0]
    ratio = float(many["loss_masks"].sum()) / float(one["loss_masks"].sum())
    # Measured on this stack: loss weight scales as ~sqrt(n), so 16 questions carry 5.66x
    # the weight of one, not 16x. Pinning the value catches a weighting-scheme change that
    # would silently make chart-dense rows dominate the gradient.
    assert ratio == pytest.approx(5.657, rel=0.02), f"weight ratio {ratio:.3f}"


def test_truncation_drops_trailing_questions_without_error(tmp_path):
    """Document the exact failure mode `stage_train.py`'s length assertion prevents.

    There is no `skip_overlong` on this config. Two different things happen depending on
    how short the budget is, and only one of them is loud:

    * Too short for the image block -> `truncate_example` raises, `get_example_with_skip`
      skips the row, and after MAX_ROW_SKIP consecutive failures the loader raises. Loud.
    * Long enough for the image but not for every branch -> the trailing questions are
      silently dropped. A 16-question figure quietly becomes a 9-question figure, and
      nothing anywhere reports it.

    The second is why staging trims and *counts* rather than trusting the loader.
    """
    path = _stage(tmp_path, n_questions=16)
    full = _build(path, max_sequence_length=4096)[0]
    full_len = int(len(full["input_ids"]))

    # A budget that comfortably holds the image block but not all 16 branches.
    # 80% of the full length: comfortably past the image block, short of all 16 branches.
    budget = int(full_len * 0.8)
    clipped = _build(path, max_sequence_length=budget)[0]
    assert len(clipped["input_ids"]) <= budget
    assert clipped["loss_masks"].sum() < full["loss_masks"].sum(), (
        "expected trailing branches to be dropped"
    )

    # And the loud case, for contrast: too short for the image block at all.
    with pytest.raises((ValueError, RuntimeError)):
        _build(path, max_sequence_length=16)[0]


@pytest.mark.parametrize("n", [1, 4, 16])
def test_question_count_round_trips(tmp_path, n):
    ds = _build(_stage(tmp_path / f"n{n}", n_questions=n))
    assert len(ds) == 2
    assert ds[0]["loss_masks"].sum() > 0
