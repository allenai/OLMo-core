from olmo_core.model_ladder import ModelSize


def test_model_size_num_params():
    assert ModelSize.size_190M.num_params == 190_000_000
    assert ModelSize.size_7B.num_params == 7_000_000_000
