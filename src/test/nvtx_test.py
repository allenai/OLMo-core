import contextlib
import importlib
import sys
import types

import pytest


@contextlib.contextmanager
def _reimported_nvtx(fake: types.ModuleType | None):
    """
    Re-import ``olmo_core._nvtx`` with or without a stand-in for the ``nvtx`` package.

    The module resolves ``nvtx`` once at import time, so the binding can only be exercised by
    reloading it. Any previously imported copy is restored afterwards.
    """
    saved_nvtx = sys.modules.get("nvtx")
    saved_module = sys.modules.pop("olmo_core._nvtx", None)
    if fake is None:
        sys.modules["nvtx"] = None  # type: ignore[assignment]  # forces ImportError
    else:
        sys.modules["nvtx"] = fake
    try:
        yield importlib.import_module("olmo_core._nvtx")
    finally:
        sys.modules.pop("olmo_core._nvtx", None)
        if saved_nvtx is None:
            sys.modules.pop("nvtx", None)
        else:
            sys.modules["nvtx"] = saved_nvtx
        if saved_module is not None:
            sys.modules["olmo_core._nvtx"] = saved_module


def _fake_nvtx() -> tuple[types.ModuleType, list[tuple[str, str | None]]]:
    calls: list[tuple[str, str | None]] = []
    module = types.ModuleType("nvtx")

    @contextlib.contextmanager
    def annotate(label, color=None):
        calls.append((label, color))
        yield

    module.annotate = annotate  # type: ignore[attr-defined]
    return module, calls


def test_exported_nvtx_is_the_real_module_when_installed():
    # This is the whole point of the module: ~20 files annotate hot paths with
    # 'from olmo_core._nvtx import nvtx'. If that name is bound to the no-op unconditionally,
    # installing the 'profiling' extra buys nothing and every range is dead code.
    fake, calls = _fake_nvtx()
    with _reimported_nvtx(fake) as mod:
        assert mod.nvtx is fake
        with mod.nvtx.annotate("permute", color="green"):
            pass
        with mod.maybe_nvtx_annotate("router", "blue"):
            pass
    assert calls == [("permute", "green"), ("router", "blue")]


def test_exported_nvtx_falls_back_to_a_no_op_when_absent():
    with _reimported_nvtx(None) as mod:
        assert type(mod.nvtx).__name__ == "_NoOpNvtx"

        # Usable as a context manager, a decorator, and with or without a color.
        with mod.nvtx.annotate("permute", color="green"):
            pass
        with mod.maybe_nvtx_annotate("router"):
            pass

        @mod.nvtx.annotate("expert", color="purple")
        def annotated():
            return 7

        assert annotated() == 7


def test_no_op_range_does_not_swallow_exceptions():
    with _reimported_nvtx(None) as mod:
        with pytest.raises(ValueError):
            with mod.nvtx.annotate("boom"):
                raise ValueError("boom")
