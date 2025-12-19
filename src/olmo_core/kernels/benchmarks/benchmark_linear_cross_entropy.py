import gc
import json
import os
from dataclasses import dataclass
from typing import Any, Callable

import pandas as pd
import torch
import triton  # type: ignore
from cut_cross_entropy import linear_cross_entropy as cce_linear_cross_entropy  # type: ignore
from liger_kernel.transformers.fused_linear_cross_entropy import LigerFusedLinearCrossEntropyLoss
from liger_kernel.utils import infer_device
from torch.nn import functional as F

from olmo_core.kernels.helion.linear_cross_entropy import OlmoFusedLinearCrossEntropyFunction

device = infer_device()

QUANTILES = [0.5, 0.2, 0.8]

# Paths for benchmark data and visualizations
BENCHMARK_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "data"))
VISUALIZATIONS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "visualizations"))
DATA_PATH = os.path.join(BENCHMARK_DIR, "all_benchmark_data.csv")


def _dtype_to_str(dtype: torch.dtype) -> str:
    """Convert torch.dtype to a JSON-serializable string."""
    return str(dtype).split(".")[-1]  # e.g., "torch.bfloat16" -> "bfloat16"


def _str_to_dtype(dtype_str: str) -> torch.dtype:
    """Convert string back to torch.dtype."""
    # Map common dtype strings to torch dtypes
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
        "float64": torch.float64,
        "int8": torch.int8,
        "int16": torch.int16,
        "int32": torch.int32,
        "int64": torch.int64,
        "uint8": torch.uint8,
    }
    if dtype_str in dtype_map:
        return dtype_map[dtype_str]
    # Fallback: try to get attribute from torch
    return getattr(torch, dtype_str)


def _serialize_config_for_json(config: dict[str, Any]) -> dict[str, Any]:
    """Convert non-JSON-serializable values in config to JSON-serializable ones."""
    serialized = {}
    for key, value in config.items():
        if isinstance(value, torch.dtype):
            serialized[key] = _dtype_to_str(value)
        else:
            serialized[key] = value
    return serialized


def _deserialize_config_from_json(config: dict[str, Any]) -> dict[str, Any]:
    """Convert JSON-serialized values back to their original types."""
    deserialized = {}
    for key, value in config.items():
        if key == "dtype" and isinstance(value, str):
            deserialized[key] = _str_to_dtype(value)
        else:
            deserialized[key] = value
    return deserialized


@dataclass
class SingleBenchmarkRunInput:
    x: int | float
    kernel_provider: str
    kernel_operation_mode: str | None = ""
    extra_benchmark_config: dict[str, Any] | None = None


@dataclass
class SingleBenchmarkRunOutput:
    y_20: float
    y_50: float
    y_80: float


@dataclass
class BenchmarkData:
    """
    BenchmarkData is a dataclass to store the benchmark data for a completed benchmark
    run on all x-values for a given kernel/kernel operation mode/metric/extra_benchmark_config
    """

    kernel_name: str
    kernel_provider: str
    metric_name: str
    metric_unit: str
    x_name: str
    x_label: str
    x_values: list[float]
    y_values_50: list[float]
    y_values_20: list[float]
    y_values_80: list[float]
    kernel_operation_mode: str | None = None
    extra_benchmark_config_str: str | None = None


def _print_benchmarking_banner(metric_name: str, kernel_name: str) -> None:
    """Print a banner for the benchmarking section."""
    print("\n" + "=" * 80)
    print(f"Benchmarking {kernel_name} - {metric_name}")
    print("=" * 80 + "\n")


def print_benchmark_data(benchmark_data_list: list[BenchmarkData]) -> None:
    """Print benchmark data in a readable format."""
    for data in benchmark_data_list:
        print(
            f"\nKernel: {data.kernel_name} | Provider: {data.kernel_provider} | Mode: {data.kernel_operation_mode}"
        )
        print(f"Metric: {data.metric_name} ({data.metric_unit})")
        print(f"{data.x_label}: {data.x_values}")
        print(f"Values (50th percentile): {[f'{v:.4f}' for v in data.y_values_50]}")


def update_benchmark_data_csv(
    benchmark_data_list: list[BenchmarkData], overwrite: bool = False
) -> None:
    """Update the benchmark data CSV file with new benchmark results."""
    os.makedirs(BENCHMARK_DIR, exist_ok=True)

    # Convert BenchmarkData to DataFrame rows
    rows = []
    for data in benchmark_data_list:
        for i, x_val in enumerate(data.x_values):
            row = {
                "kernel_name": data.kernel_name,
                "kernel_provider": data.kernel_provider,
                "metric_name": data.metric_name,
                "metric_unit": data.metric_unit,
                "x_name": data.x_name,
                "x_label": data.x_label,
                "x_value": x_val,
                "y_value_20": data.y_values_20[i],
                "y_value_50": data.y_values_50[i],
                "y_value_80": data.y_values_80[i],
                "kernel_operation_mode": data.kernel_operation_mode or "",
                "extra_benchmark_config_str": data.extra_benchmark_config_str or "{}",
            }
            rows.append(row)

    new_df = pd.DataFrame(rows)

    # Load existing data if it exists
    if os.path.exists(DATA_PATH) and not overwrite:
        existing_df = pd.read_csv(DATA_PATH)
        # Remove rows that match the new data (same kernel_name, provider, metric, mode, x_value, config)
        for _, row in new_df.iterrows():
            mask = (
                (existing_df["kernel_name"] == row["kernel_name"])
                & (existing_df["kernel_provider"] == row["kernel_provider"])
                & (existing_df["metric_name"] == row["metric_name"])
                & (existing_df["kernel_operation_mode"] == row["kernel_operation_mode"])
                & (existing_df["x_value"] == row["x_value"])
                & (existing_df["extra_benchmark_config_str"] == row["extra_benchmark_config_str"])
            )
            existing_df = existing_df[~mask]
        # Combine existing and new data
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)  # type: ignore
    else:
        combined_df = new_df

    # Save to CSV
    combined_df.to_csv(DATA_PATH, index=False)
    print(f"\nBenchmark data saved to {DATA_PATH}")


def run_benchmarks(
    bench_test_fn: Callable,
    kernel_name: str,
    metric_name: str,
    metric_unit: str,
    x_name: str,
    x_label: str,
    x_values: list[float | int],
    kernel_providers: list[str],
    kernel_operation_modes: list[str] | None = None,
    extra_benchmark_configs: list[dict[str, Any]] | None = None,
    overwrite: bool = False,
) -> None:
    """
    Run benchmarks given a bench_test_fn that takes in a SingleBenchmarkRunInput as input and
    saves data to the CSV file.

    Args:
        bench_test_fn: The benchmark test function to run. This function should take in a
            SingleBenchmarkRunInput as input and return a SingleBenchmarkRunOutput.
        kernel_name: The name of the kernel being benchmarked (e.g. "linear_cross_entropy")
        metric_name: The name of the metric being benchmarked (e.g. "speed" or "memory")
        metric_unit: The unit of the metric being benchmarked (e.g. "ms" or "MB")
        x_name: The name of the x-axis (e.g. "BT" for batch x time)
        x_label: The label of the x-axis (e.g. "B x T")
        x_values: The list of x-values to run the benchmark on (e.g. [2**i for i in range(12, 16)])
        kernel_providers: The list of kernel providers to run the benchmark on (e.g. ["liger", "helion"])
        kernel_operation_modes: The list of kernel operation modes to run the benchmark on (e.g. ["full", "backward"])
        extra_benchmark_configs: The list of extra benchmark configurations to run the benchmark on.
        overwrite: Whether to overwrite the existing benchmark data entry if it already exists.
    """
    if kernel_operation_modes is None:
        kernel_operation_modes = [None]  # type: ignore

    assert kernel_operation_modes is not None and len(kernel_operation_modes) >= 1
    assert len(kernel_providers) >= 1

    if extra_benchmark_configs is None:
        extra_benchmark_configs = [{}]

    _print_benchmarking_banner(metric_name=metric_name, kernel_name=kernel_name)

    benchmark_data_list = []
    for extra_benchmark_config in extra_benchmark_configs:
        for kernel_operation_mode in kernel_operation_modes:  # type: ignore
            for kernel_provider in kernel_providers:
                print(
                    f"Running benchmark: {kernel_name} | {metric_name} | {kernel_operation_mode} | {kernel_provider} | {extra_benchmark_config}"
                )
                y_values_50 = []
                y_values_20 = []
                y_values_80 = []

                for x in x_values:
                    single_benchmark_run_input = SingleBenchmarkRunInput(
                        x=x,
                        kernel_provider=kernel_provider,
                        kernel_operation_mode=kernel_operation_mode,
                        extra_benchmark_config=extra_benchmark_config,
                    )
                    benchmark_result: SingleBenchmarkRunOutput = bench_test_fn(
                        single_benchmark_run_input
                    )
                    y_values_50.append(benchmark_result.y_50)
                    y_values_20.append(benchmark_result.y_20)
                    y_values_80.append(benchmark_result.y_80)

                extra_config_str = (
                    json.dumps(_serialize_config_for_json(extra_benchmark_config), sort_keys=True)
                    if extra_benchmark_config
                    else "{}"
                )

                benchmark_run_data = BenchmarkData(
                    kernel_name=kernel_name,
                    kernel_operation_mode=kernel_operation_mode,
                    kernel_provider=kernel_provider,
                    metric_name=metric_name,
                    metric_unit=metric_unit,
                    x_name=x_name,
                    x_label=x_label,
                    x_values=x_values,
                    y_values_50=y_values_50,
                    y_values_20=y_values_20,
                    y_values_80=y_values_80,
                    extra_benchmark_config_str=extra_config_str,
                )

                benchmark_data_list.append(benchmark_run_data)

                # Clean up after each provider finishes benchmarking all x_values
                _cleanup_memory()

    print_benchmark_data(benchmark_data_list)
    update_benchmark_data_csv(benchmark_data_list=benchmark_data_list, overwrite=overwrite)


def _cleanup_memory():
    """Clean up memory by deleting objects, clearing CUDA cache, and running garbage collection."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def _test_memory(func: Callable, _iter: int = 10) -> float:
    """Test memory consumption of a function."""
    if not torch.cuda.is_available():
        raise RuntimeError("Memory testing requires CUDA")

    total_mem = []
    for _ in range(_iter):
        torch.cuda.reset_peak_memory_stats()
        func()
        mem = torch.cuda.max_memory_allocated() / 2**20  # to MB
        total_mem.append(mem)
    return max(total_mem)


class TorchLMHeadCE(torch.nn.Module):
    """Ground truth implementation of the linear fused with torch based cross entropy loss.

    :param H: hidden size
    :param V: vocab size
    :param ignore_index: index to ignore
    """

    def __init__(self, H: int, V: int, dtype: torch.dtype, ignore_index: int = -100):
        super().__init__()
        self.lin = torch.nn.Linear(in_features=H, out_features=V, bias=False, dtype=dtype)
        self.ce_loss = torch.nn.CrossEntropyLoss(ignore_index=ignore_index, reduction="mean")

    def forward(self, x, y):
        logits = self.lin(x)
        return self.ce_loss(logits.float(), y)


class TorchTuneLinearCrossEntropyLoss(torch.nn.Module):
    """Memory efficient Cross-entropy loss that incrementally computes loss for chunks of tokens
    by masking ignored tokens, calculating logits and then applying cross-entropy loss. Combines
    the linear projection with the cross-entropy calculation for further memory savings.

    Linear cross entropy masks out ignored tokens before the projection layer to save memory.
    You therefore need to skip the final projection layer in your model and pass it to the loss instead.
    You can setup the loss with the model and compile it as shown below.

    >>> model = Transformer(...)
    >>> loss = LinearCrossEntropyLoss(...)
    >>> loss.apply_compile_strategy()
    """

    def __init__(
        self,
        num_output_chunks: int = 8,
        reduction: str = "mean",
        ignore_index: int = -100,
        mask_ignored_tokens: bool = True,
    ):
        super().__init__()
        """
        Args:
            num_output_chunks (int): Number of chunks to split the output tensor into. Default is 8.
            ignore_index (int): Index to ignore in the target tensor. Default is -100.
            mask_ignored_tokens (bool): Whether to mask out ignored tokens during loss computation. Default is True.
        """
        self.linear_projection: torch.nn.Linear | None = None
        self.num_output_chunks = num_output_chunks
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.mask_ignored_tokens = mask_ignored_tokens
        self.apply_compile_strategy()

    def apply_compile_strategy(self, *args, **kwargs):
        """Applies compile only to the compute_cross_entropy function.
        If compiling CE + chunking operation together, memory requirement is higher."""
        self.compute_cross_entropy = torch.compile(self.compute_cross_entropy, *args, **kwargs)
        return self

    def compute_cross_entropy(
        self,
        hidden_chunk: torch.Tensor,
        target_chunk: torch.Tensor,
    ) -> torch.Tensor:
        """Computes cross-entropy by masking tokens, calculating logits and then applying cross-entropy loss.

        Args:
            hidden_chunk (torch.Tensor): [batch_size, chunk_size, embed_dim]
            target_chunk (torch.Tensor): [batch_size, chunk_size]

        Returns:
            torch.Tensor: Sum of cross-entropy loss for non-ignored tokens in the chunk

        Raises:
            AttributeError: if called before update_model
        """
        # [num_valid, embed_dim] @ [embed_dim, vocab_size]
        if self.linear_projection is None:
            raise AttributeError("forward called before update_model")
        logits = self.linear_projection(hidden_chunk)  # [num_valid, vocab_size]

        loss = F.cross_entropy(
            logits.float(),
            target_chunk,
            reduction=self.reduction,
            ignore_index=self.ignore_index,
        )
        return loss

    def mask_inputs(
        self,
        hidden: torch.Tensor,
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        indices = torch.where(target != self.ignore_index)[0]

        hidden = hidden.index_select(0, indices)

        target = target.index_select(0, indices)
        return hidden, target

    def forward(
        self,
        outputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        total_valid_tokens = torch.where(targets != self.ignore_index)[0].numel()
        if total_valid_tokens == 0:
            return torch.tensor(0.0, device=targets.device)

        targets = targets.reshape(-1)
        outputs = outputs.reshape(-1, outputs.shape[-1])

        if self.mask_ignored_tokens:
            outputs, targets = self.mask_inputs(outputs, targets)

        hidden_chunks = outputs.tensor_split(self.num_output_chunks, dim=0)
        target_chunks = targets.tensor_split(self.num_output_chunks, dim=0)

        total_loss = torch.tensor(0.0, device=targets.device)
        for hidden_chunk, target_chunk in zip(hidden_chunks, target_chunks):
            loss = self.compute_cross_entropy(hidden_chunk, target_chunk)
            total_loss += loss

        return total_loss / total_valid_tokens


class TorchTuneLMHeadCE(torch.nn.Module):
    """TorchTune chunked implementation of linear cross entropy loss."""

    def __init__(
        self,
        H: int,
        V: int,
        dtype: torch.dtype,
        ignore_index: int = -100,
        num_output_chunks: int = 8,
    ):
        super().__init__()
        self.lin = torch.nn.Linear(in_features=H, out_features=V, bias=False, dtype=dtype)
        self.ce_loss = TorchTuneLinearCrossEntropyLoss(
            num_output_chunks=num_output_chunks,
            reduction="mean",
            ignore_index=ignore_index,
            mask_ignored_tokens=True,
        )
        # Set the linear projection for the loss
        self.ce_loss.linear_projection = self.lin

    def forward(self, x, y):
        return self.ce_loss(x, y)


class LigerLMHeadCE(torch.nn.Module):
    """Liger kernel implementation of linear cross entropy loss."""

    def __init__(
        self,
        H: int,
        V: int,
        dtype: torch.dtype,
        ignore_index: int = -100,
        accum_dtype=None,
    ):
        super().__init__()
        self.lin = torch.nn.Linear(in_features=H, out_features=V, bias=False, dtype=dtype)
        self.ce_loss = LigerFusedLinearCrossEntropyLoss(
            ignore_index=ignore_index, reduction="mean", accum_dtype=accum_dtype
        )

    def forward(self, x, y):
        return self.ce_loss(self.lin.weight, x, y)


class CCELMHeadCE(torch.nn.Module):
    """CCE (Cut Cross Entropy) kernel implementation of linear cross entropy loss using cce_kahan_full_c mode."""

    def __init__(
        self,
        H: int,
        V: int,
        dtype: torch.dtype,
        ignore_index: int = -100,
        impl: str = "cce_kahan_full_c",
    ):
        super().__init__()
        self.lin = torch.nn.Linear(in_features=H, out_features=V, bias=False, dtype=dtype)
        self.ignore_index = ignore_index
        self.impl = impl

    def forward(self, x, y):
        return cce_linear_cross_entropy(
            x,
            self.lin.weight,
            y,
            ignore_index=self.ignore_index,
            reduction="mean",
            impl=self.impl,
        )


class OlmoLMHeadCE(torch.nn.Module):
    def __init__(
        self, H: int, V: int, dtype: torch.dtype, ignore_index: int = -100, reduction: str = "sum"
    ):
        super().__init__()
        self.lm_head = torch.nn.Linear(in_features=H, out_features=V, bias=False, dtype=dtype)
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, x, target):
        ce_loss, lse = OlmoFusedLinearCrossEntropyFunction.apply(  # pyright: ignore[reportGeneralTypeIssues]
            x, self.lm_head.weight.T, target, self.ignore_index, self.reduction
        )
        return ce_loss


def _create_lm_head_ce(provider: str, H: int, V: int, dtype: torch.dtype) -> Any:
    """Create a language model head with cross entropy loss based on provider."""
    if provider == "liger":
        return LigerLMHeadCE(H=H, V=V, dtype=dtype, accum_dtype=torch.float32).to(device)
    elif provider == "cce":
        return torch.compile(CCELMHeadCE(H=H, V=V, dtype=dtype, impl="cce").to(device))
    elif provider == "cce-kahan-full-c":
        return torch.compile(CCELMHeadCE(H=H, V=V, dtype=dtype, impl="cce_kahan_full_c").to(device))
    elif provider == "helion":
        return OlmoLMHeadCE(H=H, V=V, dtype=dtype).to(device)
    elif provider == "helion-compile":
        return torch.compile(OlmoLMHeadCE(H=H, V=V, dtype=dtype).to(device))
    elif provider == "torch-eager":
        return TorchLMHeadCE(H=H, V=V, dtype=dtype).to(device)
    elif provider == "torch-compile":
        return torch.compile(TorchLMHeadCE(H=H, V=V, dtype=dtype).to(device))  # type: ignore
    elif provider == "torch-tune-8chunks":
        return TorchTuneLMHeadCE(H=H, V=V, dtype=dtype, num_output_chunks=8).to(device)
    elif provider == "torch-tune-16chunks":
        return TorchTuneLMHeadCE(H=H, V=V, dtype=dtype, num_output_chunks=16).to(device)
    else:
        raise ValueError(f"Invalid provider: {provider}")


def bench_memory_linear_cross_entropy(input: SingleBenchmarkRunInput) -> SingleBenchmarkRunOutput:
    """Benchmark memory consumption of linear cross entropy loss."""
    BT = int(input.x)
    if input.extra_benchmark_config is None:
        raise ValueError("extra_benchmark_config is required")
    H = input.extra_benchmark_config["H"]
    V = input.extra_benchmark_config["V"]
    dtype = input.extra_benchmark_config["dtype"]
    provider = input.kernel_provider

    # Clean up before creating new module
    _cleanup_memory()

    lm_head_ce = None
    _input = None
    target = None

    try:
        lm_head_ce = _create_lm_head_ce(provider, H, V, dtype)
        _input = torch.randn(BT, H, requires_grad=True, dtype=dtype, device=device)
        target = torch.randint(V, (BT,), dtype=torch.long, device=device)

        def fwd():
            return lm_head_ce(_input, target)

        def full():
            y = fwd()
            y.backward()

        mem_max = _test_memory(full, _iter=10)
    finally:
        # Clean up after benchmarking
        if lm_head_ce is not None:
            del lm_head_ce
        if _input is not None:
            del _input
        if target is not None:
            del target
        _cleanup_memory()

    # Return as SingleBenchmarkRunOutput for consistency (all values are the same for memory)
    return SingleBenchmarkRunOutput(y_20=mem_max, y_50=mem_max, y_80=mem_max)


def bench_speed_linear_cross_entropy(
    input: SingleBenchmarkRunInput,
) -> SingleBenchmarkRunOutput:
    """Benchmark speed of linear cross entropy loss."""
    BT = int(input.x)
    if input.extra_benchmark_config is None:
        raise ValueError("extra_benchmark_config is required")
    H = input.extra_benchmark_config["H"]
    V = input.extra_benchmark_config["V"]
    dtype = input.extra_benchmark_config["dtype"]
    provider = input.kernel_provider
    mode = input.kernel_operation_mode

    # Clean up before creating new module
    _cleanup_memory()

    lm_head_ce = None
    _input = None
    target = None
    y = None

    try:
        lm_head_ce = _create_lm_head_ce(provider, H, V, dtype)
        _input = torch.randn(BT, H, requires_grad=True, dtype=dtype, device=device)
        target = torch.randint(V, (BT,), dtype=torch.long, device=device)

        def fwd():
            return lm_head_ce(_input, target)

        if mode == "forward":
            ms_50, ms_20, ms_80 = triton.testing.do_bench(
                fwd,
                rep=100,
                quantiles=QUANTILES,
            )
        elif mode == "no-grad-forward":
            with torch.no_grad():
                ms_50, ms_20, ms_80 = triton.testing.do_bench(
                    fwd,
                    rep=100,
                    quantiles=QUANTILES,
                )
        elif mode == "backward":
            y = fwd()

            ms_50, ms_20, ms_80 = triton.testing.do_bench(
                lambda: y.backward(retain_graph=True),
                grad_to_none=[_input],
                rep=100,
                quantiles=QUANTILES,
            )
        elif mode == "full":

            def full():
                y = fwd()
                y.backward()

            ms_50, ms_20, ms_80 = triton.testing.do_bench(
                full,
                rep=100,
                quantiles=QUANTILES,
            )
        else:
            raise ValueError(f"Invalid mode: {mode}")
    finally:
        # Clean up after benchmarking
        if y is not None:
            del y
        if lm_head_ce is not None:
            del lm_head_ce
        if _input is not None:
            del _input
        if target is not None:
            del target
        _cleanup_memory()

    return SingleBenchmarkRunOutput(
        y_20=ms_20,
        y_50=ms_50,
        y_80=ms_80,
    )


if __name__ == "__main__":
    common_configs = {
        "kernel_name": "linear_cross_entropy",
        "x_name": "BT",
        "x_label": "B x T",
        "x_values": [2**i for i in range(13, 16)],
        "kernel_providers": [
            "liger",
            "cce",
            "cce-kahan-full-c",
            "helion",
            "torch-compile",
            "torch-tune-8chunks",  # worse than torch-compile on speed and mem
        ],
        "extra_benchmark_configs": [{"H": 2560, "V": 100352, "dtype": torch.bfloat16}],
        "overwrite": True,
    }

    run_benchmarks(
        bench_test_fn=bench_speed_linear_cross_entropy,
        kernel_operation_modes=["full", "forward"],  # , "no-grad-forward"]
        metric_name="speed",
        metric_unit="ms",
        **common_configs,
    )
    run_benchmarks(
        bench_test_fn=bench_memory_linear_cross_entropy,
        kernel_operation_modes=["full", "forward"],
        metric_name="memory",
        metric_unit="MB",
        **common_configs,
    )
