import pytest
import torch

import olmo_core._nvtx as ranges


def test_opt_in_disabled_decorator_is_identity(monkeypatch):
    monkeypatch.setattr(ranges, "_COMPILE_SAFE_NOOP_RANGES", True)

    def function(x):
        return x + 1

    assert ranges.nvtx.annotate("disabled")(function) is function
    with pytest.raises(ValueError, match="not swallowed"):
        with ranges.nvtx.annotate("disabled"):
            raise ValueError("not swallowed")


def test_disabled_context_allows_compiler_resume(monkeypatch):
    @torch.compiler.disable
    def opaque(x):
        return x.sin()

    def function(x):
        with ranges.nvtx.annotate("disabled"):
            return opaque(x + 1).cos()

    x = torch.arange(8.0)
    expected = function(x)
    monkeypatch.setattr(ranges, "_COMPILE_SAFE_NOOP_RANGES", True)
    torch._dynamo.reset()
    explanation = torch._dynamo.explain(function)(x)
    assert explanation.graph_count == 2
    assert explanation.graph_break_count == 1
    actual = torch.compile(function, backend="eager")(x)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch._dynamo.reset()
