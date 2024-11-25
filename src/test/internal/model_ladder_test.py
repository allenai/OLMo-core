import pytest

from olmo_core.ladder.baseline import BaselineModelLadder


@pytest.mark.parametrize("sequence_length", [2048, 4096])
def test_validate_baseline_model_ladder(tmp_path, sequence_length):
    ladder = BaselineModelLadder(
        name="baseline", project="ladder", root_dir=tmp_path, sequence_length=sequence_length
    )
    ladder.validate()
