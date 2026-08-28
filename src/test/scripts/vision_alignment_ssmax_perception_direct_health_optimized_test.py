"""Exact replay tests for the direct SSMax health producer's shared source bundle."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from olmo_core.data.multimodal import MultimodalCollatorConfig
from olmo_core.eval.vision_alignment_ssmax_perception_direct import (
    SOURCES,
    SSMaxPerceptionDirectEvidenceError,
)
from scripts.eval import vision_alignment_ssmax_perception_direct_health as health

_SEQUENCE_LENGTH = 8
_PATCH_DIMENSION = 14 * 14 * 3


class _Dataset:
    content_fingerprint_version = "optimized-health-test-v1"

    def __init__(self, name: str, tag: int, *, size: int = 64):
        self.name = name
        self.tag = tag
        self.size = size
        self.content_fingerprint = f"{name}-content-v1"

    def __len__(self) -> int:
        return self.size

    def get(self, index: int, epoch: int) -> dict[str, np.ndarray]:
        length = 6
        value = self.tag + index + epoch
        return {
            "input_ids": np.full(length, value, dtype=np.int64),
            "labels": np.full(length, value, dtype=np.int64),
            "loss_masks": np.ones(length, dtype=np.float32),
            "position_ids": np.arange(length, dtype=np.int64),
            "token_type_ids": np.zeros(length, dtype=np.int64),
            "images": np.zeros((0, 729, _PATCH_DIMENSION), dtype=np.float32),
            "pooled_patches_idx": np.full((0, 4), -1, dtype=np.int64),
        }


class _Recipe:
    def __init__(self, names: tuple[str, ...] = SOURCES):
        self.names = names
        self.build_calls = 0

    def _build_mixture_sources(
        self, tokenizer: Any, token_ids: Any, config: Any
    ) -> tuple[list[_Dataset], list[float], list[str]]:
        del tokenizer, token_ids, config
        self.build_calls += 1
        datasets = [_Dataset(name, index * 100) for index, name in enumerate(self.names, 1)]
        weights = [float(index) for index in range(1, len(self.names) + 1)]
        return datasets, weights, list(self.names)


def _config() -> SimpleNamespace:
    return SimpleNamespace(
        collator=MultimodalCollatorConfig(
            pad_token_id=0,
            label_ignore_index=-100,
            pad_sequence_length=_SEQUENCE_LENGTH,
        ),
        global_batch_size=4 * _SEQUENCE_LENGTH,
        data_seed=6198,
    )


def _replay(loader: Any, *, batches: int = 3) -> tuple[list[Any], Any, Any]:
    loader.reshuffle(epoch=1)
    iterator = iter(loader)
    output: list[Any] = []
    try:
        for _ in range(batches):
            output.append(health.paired_runner._jsonable(next(iterator)))
        state = health.paired_runner._jsonable(loader.state_dict())
    finally:
        iterator.close()
    return output, state, health.paired_runner._jsonable(loader.dataset_fingerprints)


def test_shared_sources_build_once_and_are_exactly_legacy_equivalent(tmp_path: Path) -> None:
    recipe = _Recipe()
    config = _config()
    tokenizer = object()
    token_ids = object()
    sources = health._build_immutable_mixture_sources(recipe, config, tokenizer, token_ids)

    assert recipe.build_calls == 1
    assert isinstance(sources.datasets, tuple)
    assert isinstance(sources.weights, tuple)
    assert isinstance(sources.names, tuple)
    with pytest.raises(FrozenInstanceError):
        sources.contract_sha256 = "0" * 64  # type: ignore[misc]

    optimized: list[tuple[list[Any], Any, Any]] = []
    for rank in range(2):
        loader = health._build_rank_loader(
            config,
            sources,
            rank=rank,
            world_size=2,
            work_dir=tmp_path / "optimized",
            prefetch_workers=0,
        )
        assert all(
            observed is expected
            for observed, expected in zip(loader.datasets, sources.datasets, strict=True)
        )
        optimized.append(_replay(loader))
        health._validate_immutable_mixture_sources(sources)
    assert recipe.build_calls == 1

    legacy: list[tuple[list[Any], Any, Any]] = []
    for rank in range(2):
        loader = health.paired_runner._build_loader(
            recipe,
            config,
            tokenizer,
            token_ids,
            rank=rank,
            world_size=2,
            work_dir=tmp_path / "legacy",
            prefetch_workers=0,
        )
        legacy.append(_replay(loader))

    assert recipe.build_calls == 3
    assert optimized == legacy


def test_shared_sources_fail_closed_on_source_order_drift() -> None:
    recipe = _Recipe(tuple(reversed(SOURCES)))

    with pytest.raises(SSMaxPerceptionDirectEvidenceError, match="source order differs"):
        health._build_immutable_mixture_sources(recipe, _config(), object(), object())


@pytest.mark.parametrize("field", ["content_fingerprint", "size"])
def test_shared_sources_fail_closed_on_semantic_mutation(field: str, tmp_path: Path) -> None:
    sources = health._build_immutable_mixture_sources(_Recipe(), _config(), object(), object())
    dataset = sources.datasets[0]
    if field == "content_fingerprint":
        dataset.content_fingerprint = "changed-content"
    else:
        dataset.size += 1

    with pytest.raises(SSMaxPerceptionDirectEvidenceError, match="source contract changed"):
        health._build_rank_loader(
            _config(),
            sources,
            rank=0,
            world_size=2,
            work_dir=tmp_path,
            prefetch_workers=0,
        )


def test_shared_sources_fail_closed_on_misaligned_components() -> None:
    with pytest.raises(SSMaxPerceptionDirectEvidenceError, match="equal non-empty"):
        health._mixture_source_contract([_Dataset("one", 1)], [], ["one"])
