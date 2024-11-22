from olmo_core.ladder.baseline import BaselineModelLadder


def test_validate_baseline_model_ladder():
    ladder = BaselineModelLadder()
    ladder.validate()
