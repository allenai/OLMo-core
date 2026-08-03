from olmo_core._nvtx import nvtx


def test_noop_nvtx_as_decorator():
    def add(a, b):
        return a + b

    original = add
    add = nvtx.annotate("range", color="red")(add)

    # A disabled annotation must be a true identity decorator. ContextDecorator-style wrappers
    # introduce unsupported context managers into TorchDynamo-compiled model code.
    assert add is original


def test_noop_nvtx_decorated_function_result():
    @nvtx.annotate("range", color="red")
    def add(a, b):
        return a + b

    assert add(2, 3) == 5


def test_noop_nvtx_as_context_manager():
    with nvtx.annotate("range"):
        result = 21 * 2
    assert result == 42
