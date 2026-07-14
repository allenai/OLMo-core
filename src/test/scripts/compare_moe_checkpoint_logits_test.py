import importlib.util
from pathlib import Path

import torch


SCRIPT = Path(__file__).parents[2] / "scripts/compare_moe_checkpoint_logits.py"
SPEC = importlib.util.spec_from_file_location("compare_moe_checkpoint_logits", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_tensor_bitwise_equal_distinguishes_signed_zero() -> None:
    reference = torch.tensor([0.0], dtype=torch.float32)
    candidate = torch.tensor([-0.0], dtype=torch.float32)
    assert torch.equal(reference, candidate)
    assert not MODULE.tensor_bitwise_equal(reference, candidate)


def test_intermediate_comparison_reports_bit_corruption() -> None:
    reference = {"intermediates": {"blocks.0.output": torch.tensor([1.0, 2.0])}}
    candidate = {"intermediates": {"blocks.0.output": torch.tensor([1.0, 3.0])}}
    comparisons = MODULE.compare_intermediates(reference, candidate)
    assert len(comparisons) == 1
    assert comparisons[0]["bitwise_equal"] is False
    assert comparisons[0]["mismatch_count"] == 1
