from pathlib import Path

import numpy as np
import pytest

from olmo_core.data.composable import ConcatAndChunkInstanceSourceConfig
from olmo_core.data.composable.mixture_recipe import build_numpy_mixture_from_yaml_spec
from olmo_core.data.tokenizer import TokenizerConfig
from olmo_core.exceptions import OLMoConfigurationError


def _write_mmap(path: Path, num_tokens: int, dtype=np.uint16, eos_token_id: int = 0):
    data = np.ones(num_tokens, dtype=dtype)
    data[-1] = eos_token_id
    mm = np.memmap(str(path), mode="w+", dtype=dtype, shape=(num_tokens,))
    mm[:] = data
    mm.flush()
    return path


@pytest.fixture
def tokenizer():
    return TokenizerConfig(vocab_size=32_000, eos_token_id=0, pad_token_id=-1)


def test_build_numpy_mixture_flat_sources(tmp_path: Path, tokenizer: TokenizerConfig):
    """A flat mix of two sources produces a ConcatAndChunkInstanceSourceConfig with two sampled sources."""
    source_a = _write_mmap(tmp_path / "source_a.npy", 200)
    source_b = _write_mmap(tmp_path / "source_b.npy", 200)

    yaml_path = tmp_path / "mix.yaml"
    yaml_path.write_text(
        f"""\
mix:
- name: source_a
  weight: 0.5
  paths:
  - {source_a}
- name: source_b
  weight: 0.5
  paths:
  - {source_b}
"""
    )

    result = build_numpy_mixture_from_yaml_spec(
        yaml_path,
        tokenizer=tokenizer,
        total_tokens=100,
        sequence_length=4,
        sampling_strategy="contiguous_chunks",
    )

    assert isinstance(result, ConcatAndChunkInstanceSourceConfig)
    assert result.sequence_length == 4
    assert len(result.sources) == 2


def test_build_numpy_mixture_hierarchical_sources(tmp_path: Path, tokenizer: TokenizerConfig):
    """A hierarchical mix (categories within categories) is flattened to individual leaf sources."""
    files = {
        name: _write_mmap(tmp_path / f"{name}.npy", 200)
        for name in ("a_high", "a_low", "b_high", "b_low")
    }

    yaml_path = tmp_path / "mix.yaml"
    yaml_path.write_text(
        f"""\
mix:
- name: category_a
  weight: 0.5
  categories:
  - name: high_quality
    weight: 0.6
    paths:
    - {files["a_high"]}
  - name: low_quality
    weight: 0.4
    paths:
    - {files["a_low"]}
- name: category_b
  weight: 0.5
  categories:
  - name: high_quality
    weight: 0.6
    paths:
    - {files["b_high"]}
  - name: low_quality
    weight: 0.4
    paths:
    - {files["b_low"]}
"""
    )

    result = build_numpy_mixture_from_yaml_spec(
        yaml_path,
        tokenizer=tokenizer,
        total_tokens=100,
        sequence_length=4,
        sampling_strategy="documents",
    )

    assert isinstance(result, ConcatAndChunkInstanceSourceConfig)
    assert result.sequence_length == 4
    # 2 top-level categories × 2 quality tiers = 4 leaf sources
    assert len(result.sources) == 4


def test_build_numpy_mixture_invalid_weights(tmp_path: Path, tokenizer: TokenizerConfig):
    """Weights that don't sum to 1.0 within a group raise OLMoConfigurationError."""
    source_a = _write_mmap(tmp_path / "source_a.npy", 200)
    source_b = _write_mmap(tmp_path / "source_b.npy", 200)

    yaml_path = tmp_path / "mix.yaml"
    yaml_path.write_text(
        f"""\
mix:
- name: source_a
  weight: 0.3
  paths:
  - {source_a}
- name: source_b
  weight: 0.3
  paths:
  - {source_b}
"""
    )

    with pytest.raises(OLMoConfigurationError, match="weights within"):
        build_numpy_mixture_from_yaml_spec(
            yaml_path,
            tokenizer=tokenizer,
            total_tokens=100,
            sequence_length=4,
            sampling_strategy="contiguous_chunks",
        )


def test_build_numpy_mixture_insufficient_tokens(tmp_path: Path, tokenizer: TokenizerConfig):
    """Requesting more tokens than a source contains raises OLMoConfigurationError."""
    source_a = _write_mmap(tmp_path / "source_a.npy", 10)
    source_b = _write_mmap(tmp_path / "source_b.npy", 10)

    yaml_path = tmp_path / "mix.yaml"
    yaml_path.write_text(
        f"""\
mix:
- name: source_a
  weight: 0.5
  paths:
  - {source_a}
- name: source_b
  weight: 0.5
  paths:
  - {source_b}
"""
    )

    with pytest.raises(OLMoConfigurationError, match="doesn't have enough tokens"):
        build_numpy_mixture_from_yaml_spec(
            yaml_path,
            tokenizer=tokenizer,
            total_tokens=10_000,
            sequence_length=4,
            sampling_strategy="contiguous_chunks",
        )
