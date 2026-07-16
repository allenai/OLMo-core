import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


TECH_REPORT_SCRIPT_DIR = Path(__file__).parents[2] / "scripts" / "train" / "tech_report"
sys.path.insert(0, str(TECH_REPORT_SCRIPT_DIR))

import moe_8l_ddp as benchmark  # noqa: E402


class _Tokenizer:
    @staticmethod
    def padded_vocab_size() -> int:
        return 128


@pytest.fixture
def common():
    return SimpleNamespace(tokenizer=_Tokenizer())


@pytest.fixture(autouse=True)
def benchmark_defaults(monkeypatch):
    monkeypatch.setattr(benchmark, "EP_PATH_NAME", "auto")
    monkeypatch.setattr(benchmark, "PARALLEL_DEGREE", 8)
    monkeypatch.setattr(benchmark, "RANK_MICROBATCH_SEQUENCES", 2)
    monkeypatch.setattr(benchmark, "FORCE_FUSED_ATTENTION", False)
    monkeypatch.setattr(benchmark, "RECOMPUTE_EACH_BLOCK", False)
    monkeypatch.setattr(benchmark, "TWO_BATCH_OVERLAP", False)
    monkeypatch.setattr(benchmark, "UNIFORM_ROUTING", False)


def test_default_schedule_and_random_routing_are_unchanged(common):
    config = benchmark.build_model_config(common)

    assert config.two_batch_overlap is False
    assert config.block.ep.schedule == benchmark.ExpertParallelSchedule.normal
    assert config.block.ep.shared_slots == 1
    assert config.block.routed_experts_router.uniform_expert_assignment is False
    assert config.block.routed_experts_router.random_expert_assignment is True


def test_fused_attention_can_be_forced_without_enabling_mxfp8(common, monkeypatch):
    monkeypatch.setattr(benchmark, "FORCE_FUSED_ATTENTION", True)
    monkeypatch.setattr(benchmark, "MXFP8_ATTN_QKV", False)
    monkeypatch.setattr(benchmark, "MXFP8_ATTN_OUT", False)
    monkeypatch.setattr(benchmark, "MXFP8_ATTN_SAVE_QKV", False)

    config = benchmark.build_model_config(common)

    assert config.block.attention.name == benchmark.AttentionType.fused_v2
    assert config.block.attention.mxfp8_qkv_projection is None
    assert config.block.attention.mxfp8_out_projection is None
    assert config.block.attention.mxfp8_save_qkv_for_backward is False


def test_tbo_selects_rowwise_schedule_and_uniform_routing(common, monkeypatch):
    monkeypatch.setattr(benchmark, "TWO_BATCH_OVERLAP", True)
    monkeypatch.setattr(benchmark, "UNIFORM_ROUTING", True)

    config = benchmark.build_model_config(common)

    assert config.two_batch_overlap is True
    assert config.block.ep.path == benchmark.ExpertParallelPath.rowwise_nvshmem
    assert config.block.ep.schedule == benchmark.ExpertParallelSchedule.tbo
    assert config.block.ep.shared_slots == 2
    assert config.block.routed_experts_router.uniform_expert_assignment is True
    assert config.block.routed_experts_router.random_expert_assignment is False


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        (
            {"RECOMPUTE_EACH_BLOCK": True},
            "cannot be combined with TECH_REPORT_RECOMPUTE_EACH_BLOCK=1",
        ),
        (
            {"EP_PATH_NAME": "sync_1d"},
            "requires TECH_REPORT_EP_PATH=rowwise_nvshmem",
        ),
        (
            {"RANK_MICROBATCH_SEQUENCES": 3},
            "requires an even TECH_REPORT_RANK_MICROBATCH_SEQUENCES",
        ),
    ],
)
def test_tbo_rejects_unsupported_configurations(common, monkeypatch, overrides, message):
    monkeypatch.setattr(benchmark, "TWO_BATCH_OVERLAP", True)
    for name, value in overrides.items():
        monkeypatch.setattr(benchmark, name, value)

    with pytest.raises(ValueError, match=message):
        benchmark.build_model_config(common)
