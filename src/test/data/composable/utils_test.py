import numpy as np

from olmo_core.data.composable.utils import build_global_indices, calculate_sample_sizes


def test_calculate_sample_sizes():
    # Case 1: imbalanced sources but we want a balanced ratio without oversampling, so we
    # under-sample from the larger of the two sources.
    assert calculate_sample_sizes(
        [8, 16],
        [0.5, 0.5],
        [1.0, 1.0],
    ).tolist() == [8, 8]

    # Case 2: a similar situation but with three sources and a different target ratio.
    assert calculate_sample_sizes(
        [8, 8, 16],
        [0.3, 0.2, 0.5],
        [1.0, 1.0, 1.0],
    ).tolist() == [8, 5, 13]

    # Case 3: imbalanced sources but we want a balanced ratio with oversampling allowed, so we
    # "meet in the middle" by oversampling the smaller of the two sources while undersampling the
    # larger one to maintain the same number of tokens that we started with.
    assert calculate_sample_sizes(
        [8, 16],
        [0.5, 0.5],
        [2.0, 1.0],
    ).tolist() == [12, 12]


def test_build_global_indices(seed: int = 42):
    total_tokens = 24
    max_sequence_length = 8

    sequence_length = 4
    indices1 = build_global_indices(
        total_tokens // sequence_length,
        sequence_length=sequence_length,
        max_sequence_length=max_sequence_length,
        seed=seed,
    )
    assert indices1.size == 6
    assert indices1[0:2].tolist() in [[0, 1], [2, 3], [4, 5]]
    assert indices1[2:4].tolist() in [[0, 1], [2, 3], [4, 5]]
    assert indices1[4:6].tolist() in [[0, 1], [2, 3], [4, 5]]

    sequence_length = 8
    indices2 = build_global_indices(
        total_tokens // sequence_length,
        sequence_length=sequence_length,
        max_sequence_length=max_sequence_length,
        seed=seed,
    )
    assert (np.repeat(indices2, 2) == indices1 // 2).all()
