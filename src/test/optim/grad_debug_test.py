import logging

import pytest
import torch

from olmo_core.optim.grad_debug import debug_nan_inf_grad_norm


@pytest.mark.parametrize("finite,rank", [(True, 0), (False, 1)])
def test_grad_diagnostics_skip_callbacks_for_finite_norms_or_excluded_ranks(finite, rank):
    def unexpected():
        raise AssertionError("Diagnostic callbacks should not run")

    debug_nan_inf_grad_norm(
        torch.tensor(1.0 if finite else float("nan")),
        step=5,
        rank=rank,
        ranks_filter="0",
        component_norms=unexpected,
        iter_local_grads=unexpected,
    )


def test_grad_diagnostics_limit_logs_but_keep_full_report(caplog, tmp_path):
    grads = [
        ("finite", "dp", "Replicate()", torch.tensor([3.0, 4.0])),
        ("nonfinite_a", "ep_dp", "Shard(0)", torch.tensor([float("nan")])),
        ("nonfinite_b", "ep_dp", "Shard(0)", torch.tensor([float("inf")])),
    ]

    with caplog.at_level(logging.ERROR):
        debug_nan_inf_grad_norm(
            torch.tensor(float("nan")),
            step=5,
            rank=0,
            component_norms=lambda: {"dense": torch.tensor(5.0)},
            iter_local_grads=lambda: iter(grads),
            max_log_entries=1,
            dump_dir=str(tmp_path),
        )

    report = torch.load(
        tmp_path / "rank000_step000005_optim_nonfinite_grad_norm.pt", weights_only=True
    )
    assert report["rank"] == 0
    assert report["step"] == 5
    assert report["components"] == {"dense": 5.0}
    assert [entry["name"] for entry in report["bad_entries"]] == ["nonfinite_a", "nonfinite_b"]
    assert len(report["top_entries"]) == 3
    assert report["top_entries"][-1]["local_norm"] == 5.0
    assert caplog.text.count("BAD ") == 1
    assert "01." in caplog.text and "02." not in caplog.text
