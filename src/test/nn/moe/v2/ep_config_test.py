from olmo_core.config import Config
from olmo_core.nn.moe.v2.ep_config import ExpertParallelConfig, ExpertParallelPath


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
