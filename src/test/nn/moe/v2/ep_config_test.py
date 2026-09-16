import pytest

from olmo_core.config import Config
from olmo_core.nn.moe.v2.ep_config import ExpertParallelConfig, ExpertParallelPath


@pytest.mark.parametrize("path", list(ExpertParallelPath))
def test_ep_config_round_trip_preserves_backend_and_defaults(path):
    config = ExpertParallelConfig(path=path)
    config.validate()
    saved = config.as_config_dict()
    restored = Config.from_dict(saved)
    restored.validate()

    assert isinstance(restored, ExpertParallelConfig)
    assert restored.as_config_dict() == saved


def test_removed_ep_fields_are_ignored() -> None:
    config = Config.from_dict(
        {
            "_CLASS_": "olmo_core.nn.moe.v2.ep_config.ExpertParallelConfig",
            "path": "rowwise_nvshmem",
            "tma_ibgda_symmetric_expert_out": False,
            "wave_use_bf16_persistent_mega_forward": False,
        }
    )

    assert isinstance(config, ExpertParallelConfig)
    assert config.path == ExpertParallelPath.rowwise_nvshmem
