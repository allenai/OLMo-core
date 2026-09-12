from olmo_core._nvtx import nvtx


def test_noop_nvtx_as_decorator():
    @nvtx.annotate("range", color="red")
    def add(a, b):
        return a + b

    assert add(2, 3) == 5


def test_noop_nvtx_as_context_manager():
    with nvtx.annotate("range"):
        result = 21 * 2
    assert result == 42
