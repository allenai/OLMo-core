"""
CPU-runnable coverage for the experimental EP backends (DeepEP V2 and rowwise-wave).

The transport forwards themselves are GPU-only (and DeepEP additionally requires the ``deep_ep``
package), so these tests cover the config surface and module entry points that gate them.
"""

import pytest

from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.nn.moe.v2 import ep_deepep_v2, ep_no_sync_rowwise_wave
from olmo_core.nn.moe.v2.ep_config import (
    ExpertParallelConfig,
    ExpertParallelPath,
    ExpertParallelSchedule,
)


@pytest.mark.parametrize(
    "path",
    [ExpertParallelPath.deepep_v2, ExpertParallelPath.rowwise_wave],
)
def test_ep_config_accepts_experimental_paths(path: ExpertParallelPath):
    config = ExpertParallelConfig(path=path)
    config.validate()
    assert config.path == path


def test_ep_config_still_rejects_tbo_schedule():
    config = ExpertParallelConfig(schedule=ExpertParallelSchedule.tbo)
    with pytest.raises(OLMoConfigurationError, match="tbo"):
        config.validate()


def test_ep_config_rowwise_wave_fields_normalize():
    config = ExpertParallelConfig(
        path=ExpertParallelPath.rowwise_wave,
        rowwise_wave_num_waves=4,
        rowwise_wave_mode="EXPERT",
    )
    config.validate()
    assert config.rowwise_wave_num_waves == 4
    # Mode is lower-cased during validation.
    assert config.rowwise_wave_mode == "expert"


def test_ep_config_rowwise_wave_num_waves_requires_wave_path():
    # num_waves != 1 is only valid on the rowwise_wave path.
    config = ExpertParallelConfig(
        path=ExpertParallelPath.rowwise_nvshmem,
        rowwise_wave_num_waves=4,
    )
    with pytest.raises(OLMoConfigurationError, match="rowwise_wave_num_waves"):
        config.validate()


def test_experimental_backend_entry_points_exist():
    # The block dispatches to these lazily-imported entry points.
    assert hasattr(ep_deepep_v2, "combined_forward_ep_deepep_v2")
    assert hasattr(ep_no_sync_rowwise_wave, "combined_forward_ep_no_sync_rowwise_wave")
    # DeepEP availability probe must be import-safe even without the optional package installed.
    assert ep_deepep_v2.is_deepep_available("/nonexistent-deepep-path") is False


def test_ep_config_deprecated_rowwise_nblocks_populates_split_fields():
    # Configs from before the get/put/weighted_put split still deserialize and migrate.
    config = ExpertParallelConfig.from_dict({"rowwise_nblocks": 128})
    with pytest.warns(DeprecationWarning, match="rowwise_nblocks"):
        config.validate()
    assert config.rowwise_get_nblocks == 128
    assert config.rowwise_put_nblocks == 128
    assert config.rowwise_weighted_put_nblocks == 128
    assert config.rowwise_nblocks is None


def test_ep_config_deprecated_rowwise_nblocks_rejects_conflict():
    config = ExpertParallelConfig(rowwise_nblocks=128, rowwise_put_nblocks=64)
    with pytest.raises(OLMoConfigurationError, match="rowwise_nblocks"):
        config.validate()
