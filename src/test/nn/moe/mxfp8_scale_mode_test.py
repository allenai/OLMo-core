from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from olmo_core.kernels import mxfp8_utils
from olmo_core.kernels.mxfp8_utils import (
    dequantize_rows_from_mxfp8,
    quantize_rows_to_mxfp8,
)
from olmo_core.mxfp8_config import get_mxfp8_default_scale_mode
from olmo_core.nn.moe.v2.fp8 import MoERowwiseFP8Config, MoERowwiseFP8ScaleMode
from olmo_core.testing import requires_gpu
from olmo_core.testing.utils import requires_compute_capability


def _one_block(value: float) -> torch.Tensor:
    x = torch.zeros(1, 32, dtype=torch.float32)
    x[0, 0] = value
    return x


def _scale_byte(scales: torch.Tensor) -> int:
    return int(scales.view(torch.uint8)[0, 0].item())


def test_mxfp8_scale_mode_boundary_saturates_only_floor() -> None:
    x = _one_block(500.0)

    q_floor, s_floor = quantize_rows_to_mxfp8(x, block_size=32, scale_mode="floor")
    q_rceil, s_rceil = quantize_rows_to_mxfp8(x, block_size=32, scale_mode="rceil")

    assert _scale_byte(s_floor) == 127  # scale 1
    assert _scale_byte(s_rceil) == 128  # scale 2
    assert q_floor.to(torch.float32)[0, 0].item() == torch.finfo(torch.float8_e4m3fn).max
    assert q_rceil.to(torch.float32)[0, 0].item() < torch.finfo(torch.float8_e4m3fn).max

    dq_floor = dequantize_rows_from_mxfp8(
        q_floor,
        s_floor,
        block_size=32,
        out_dtype=torch.float32,
    )
    dq_rceil = dequantize_rows_from_mxfp8(
        q_rceil,
        s_rceil,
        block_size=32,
        out_dtype=torch.float32,
    )
    assert dq_floor[0, 0].item() == 448.0
    assert dq_rceil[0, 0].item() == 512.0


def test_mxfp8_scale_mode_agrees_at_exact_fp8_boundary() -> None:
    x = _one_block(448.0)

    q_floor, s_floor = quantize_rows_to_mxfp8(x, block_size=32, scale_mode="floor")
    q_rceil, s_rceil = quantize_rows_to_mxfp8(x, block_size=32, scale_mode="rceil")

    assert _scale_byte(s_floor) == 127
    assert _scale_byte(s_rceil) == 127
    torch.testing.assert_close(q_floor.to(torch.float32), q_rceil.to(torch.float32))


def test_rowwise_fp8_config_scale_mode_reflects_import_time_default() -> None:
    default_mode = get_mxfp8_default_scale_mode()

    assert default_mode == "rceil"

    cfg = MoERowwiseFP8Config()
    assert cfg.scale_mode.value == default_mode
    cfg.validate()

    cfg_from_string = MoERowwiseFP8Config(scale_mode=default_mode)  # type: ignore[arg-type]
    cfg_from_string.validate()
    assert cfg_from_string.scale_mode is MoERowwiseFP8ScaleMode(default_mode)

    other_mode = "rceil" if default_mode == "floor" else "floor"
    with pytest.raises(ValueError, match="OLMO_MXFP8_SCALE_MODE"):
        MoERowwiseFP8Config(scale_mode=other_mode).validate()  # type: ignore[arg-type]


def test_mxfp8_scale_mode_env_is_resolved_at_olmo_core_import() -> None:
    src_dir = Path(__file__).parents[3]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(src_dir) + os.pathsep + env.get("PYTHONPATH", "")
    env["OLMO_MXFP8_SCALE_MODE"] = "rceil"

    code = """
import os
import olmo_core
from olmo_core.mxfp8_config import get_mxfp8_default_scale_mode
print(get_mxfp8_default_scale_mode())
os.environ["OLMO_MXFP8_SCALE_MODE"] = "floor"
print(get_mxfp8_default_scale_mode())
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        env=env,
        text=True,
        capture_output=True,
    )
    assert result.stdout.strip().splitlines() == ["rceil", "rceil"]


def test_mxfp8_scale_mode_floor_env_is_resolved_at_olmo_core_import() -> None:
    src_dir = Path(__file__).parents[3]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(src_dir) + os.pathsep + env.get("PYTHONPATH", "")
    env["OLMO_MXFP8_SCALE_MODE"] = "floor"

    code = """
from olmo_core.mxfp8_config import get_mxfp8_default_scale_mode
print(get_mxfp8_default_scale_mode())
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        env=env,
        text=True,
        capture_output=True,
    )
    assert result.stdout.strip() == "floor"


def test_mxfp8_backend_env_defaults_to_olmo(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OLMO_MXFP8_QDQ_BACKEND", raising=False)
    monkeypatch.delenv("OLMO_MXFP8_Q_BACKEND", raising=False)
    monkeypatch.delenv("OLMO_MXFP8_DQ_BACKEND", raising=False)

    assert mxfp8_utils._mxfp8_backend("Q") == "olmo"  # type: ignore[attr-defined]

    monkeypatch.setenv("OLMO_MXFP8_QDQ_BACKEND", "triton")
    assert mxfp8_utils._mxfp8_backend("Q") == "olmo"  # type: ignore[attr-defined]

    monkeypatch.setenv("OLMO_MXFP8_Q_BACKEND", "transformer_engine")
    assert mxfp8_utils._mxfp8_backend("Q") == "te"  # type: ignore[attr-defined]

    monkeypatch.setenv("OLMO_MXFP8_Q_BACKEND", "nope")
    with pytest.raises(ValueError, match="OLMO_MXFP8_Q_BACKEND"):
        mxfp8_utils._mxfp8_backend("Q")  # type: ignore[attr-defined]


def test_mxfp8_te_backend_request_raises_for_cpu_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OLMO_MXFP8_Q_BACKEND", "te")
    x = torch.zeros((1, 32), dtype=torch.float32)

    with pytest.raises(RuntimeError, match="requires a CUDA input tensor"):
        quantize_rows_to_mxfp8(x, block_size=32, scale_mode="rceil")


@requires_gpu
@requires_compute_capability(min_cc=9)
def test_mxfp8_te_preallocated_rceil_matches_olmo(monkeypatch: pytest.MonkeyPatch) -> None:
    if mxfp8_utils._get_te_mxfp8_state() is None:  # type: ignore[attr-defined]
        pytest.skip("TransformerEngine MXFP8 is unavailable")

    monkeypatch.setenv("OLMO_MXFP8_Q_BACKEND", "olmo")
    x = torch.randn((128, 128), device="cuda", dtype=torch.bfloat16)
    q_olmo, s_olmo = quantize_rows_to_mxfp8(x, block_size=32, scale_mode="rceil")

    q_te = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    s_te = torch.empty((128, 4), device=x.device, dtype=torch.float8_e8m0fnu)
    result = mxfp8_utils._quantize_to_mxfp8_te(  # type: ignore[attr-defined]
        x,
        block_size=32,
        scale_mode="rceil",
        out=q_te,
        scales_out=s_te,
    )

    assert result is not None
    assert torch.equal(q_olmo.view(torch.uint8), q_te.view(torch.uint8))
    assert torch.equal(s_olmo.view(torch.uint8), s_te.view(torch.uint8))


@requires_gpu
def test_mxfp8_te_backend_request_raises_when_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OLMO_MXFP8_Q_BACKEND", "te")
    monkeypatch.setattr(mxfp8_utils, "_get_te_mxfp8_state", lambda: None)

    x = torch.randn((128, 128), device="cuda", dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="TransformerEngine MXFP8 Q backend was requested"):
        quantize_rows_to_mxfp8(x, block_size=32, scale_mode="rceil")
