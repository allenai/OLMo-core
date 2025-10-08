from olmo_core.data.composable.utils import calculate_sample_sizes


def test_calculate_sample_sizes():
    assert calculate_sample_sizes(
        [8, 16],
        [0.5, 0.5],
        [1.0, 1.0],
    ).tolist() == [8, 8]

    assert calculate_sample_sizes(
        [8, 8, 16],
        [0.3, 0.2, 0.5],
        [1.0, 1.0, 1.0],
    ).tolist() == [8, 5, 13]

    assert calculate_sample_sizes(
        [8, 16],
        [0.5, 0.5],
        [2.0, 1.0],
    ).tolist() == [12, 12]
