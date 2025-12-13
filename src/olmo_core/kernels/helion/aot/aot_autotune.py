# Adapted from https://gist.github.com/Chillee/97e530fe23897d4f730dbb0f89a98d1e by @Chillee

import functools
import json
import logging
import os
from collections import defaultdict
from collections.abc import Iterable
from enum import Enum
from pathlib import Path
from typing import Any, Callable, NamedTuple

import helion  # noqa: F401
import torch
from triton.testing import do_bench  # pyright: ignore[reportMissingImports]

log = logging.getLogger(__name__)

AutotuneInputFn = Callable[[], Iterable[tuple[Any, ...]]]


class KernelKey(NamedTuple):
    """
    Key used for kernel config lookup and matching.

    numeric_key: Numeric value used for sorting (e.g., tensor size)
    hash_key: Hashable key for exact matching (e.g., shape, dtype)
    exact_key: Key that must match exactly for config reuse (e.g., specialized dimensions)
    """
    numeric_key: Any
    hash_key: tuple[Any, ...]
    exact_key: tuple[Any, ...]


def get_cuda_device_capability() -> tuple[int, int]:
    if torch.cuda.is_available():
        dev = torch.cuda.current_device()
        cc_major, cc_minor = torch.cuda.get_device_capability(dev)
        return cc_major, cc_minor
    return (0, 0)


class AOTAutotuneMode(Enum):
    NONE = "none"
    RETUNE = "retune"
    CREATE = "create"
    IMPUTE = "impute"
    ROUTE = "route"

    @classmethod
    def from_str(cls, mode: str) -> "AOTAutotuneMode":
        return cls[mode.upper()]


def load_autotune_data_from_json(
    path: Path,
) -> tuple[dict[int, Any], dict[KernelKey, int]]:
    with open(path, "r") as f:
        json_obj = json.load(f)
        useful_configs = {int(k): eval(v) for k, v in json_obj["useful_configs"].items()}
        hash_configs = {}
        for k, v in json_obj["hash_configs"].items():
            key_tuple = tuple(eval(k))
            # Convert tuple to KernelKey
            if len(key_tuple) == 3:
                hash_configs[KernelKey(*key_tuple)] = v
            else:
                # Handle backwards compatibility if needed
                hash_configs[KernelKey(*key_tuple)] = v
    return useful_configs, hash_configs


def bind_and_compile_kernel(kernel: helion.Kernel, args: Any, config: helion.Config) -> Callable:
    has_int64 = any([i.numel() >= 2**31 for i in args if isinstance(i, torch.Tensor)])
    if has_int64:
        kernel.settings.index_dtype = torch.int64
    return kernel.bind(args).compile_config(config)


def aot_autotune(
    config_name: str,
    kernel_key: Callable[..., KernelKey],
    primary_inputs: AutotuneInputFn,
    secondary_inputs: AutotuneInputFn | None = None,
    warn_on_hash_miss: bool = True,
):
    """
    A decorator that automatically tunes and dispatches a Helion kernel based off of the kernel_key and provided inputs.

    Args:
        config_name: Name for the config file (stored in configs/{config_name}_{gpu_arch}.json)
        kernel_key: Function that returns a KernelKey(numeric_key, hash_key, exact_key) for each input.
                    The semantics are:
                    - numeric_key: Used for sorting (e.g., tensor size)
                    - hash_key: Used for exact matching (e.g., shape, dtype)
                    - exact_key: Must match exactly for config reuse (e.g., specialized dimensions)
                    If all 3 match a saved key, we'll use that config. Otherwise, we use a heuristic:
                    - We require exact_key to match the saved config
                    - We prioritize hash_key that match the saved config
                    - We then prioritize the highest numeric_key that's <= the current numeric_key
        primary_inputs: Function returning iterable of input tuples to autotune on
        secondary_inputs: Optional function returning additional inputs to benchmark (but not autotune)
        warn_on_hash_miss: Whether to log a warning when no exact config match is found

    The general flow is this:
    1. We first run helion_kernel.autotune on all primary_inputs. This will give us a list of configs, one per primary_input.
    2. We benchmark every config on every primary_input and secondary_input.
    3. For every primary input/secondary input, we keep the config that is the fastest (NB: We aim to do some deduplication by reusing configs if they're within some threshold of the fastest config).
    4. We'll save the configs and the dispatch choices to a json file.
    5. We create a dispatch function that will lookup to see whether each kernel is one we've tuned for in step 2. If so, we'll use that config. Otherwise, we'll use some heuristic to (deterministically) find a reasonable config for the kernel.

    There are 5 modes (set by env variable HELION_AOT_AUTOTUNE):
    - none: No autotuning will be done. We will skip to step 5. If the config json file doesn't exist, we'll raise an error.
    - retune: We only benchmark the existing useful configs on the primary/secondary inputs. We'll skip to step 2. This finishes much faster than create (albeit won't fully retune for each shape), but requires create to have been run first. For example, for rmsnorm, retune takes maybe one minute for 10 shapes, but create might take 30 minutes.
    - route: Like retune but JIT instead of AOT. For each input, benchmark all existing configs and select the fastest one. No new configs are generated via autotuning. This allows finding the best existing config for new input shapes. Requires existing configs to have been created first.
    - create: The kernel will be fully autotuned, starting from step 1.
    - impute: Like create but doesn't retuneLoad existing configs if available, then only autotune missing inputs. For inputs already in the config, skip autotuning and reuse existing configs. For new inputs, run full autotuning (step 1) and benchmark all configs. This is useful for incrementally adding new shapes without re-tuning existing ones.

    You can also minimize the kernels to be autotuned by setting the env variable HELION_AOT_AUTOTUNE_KERNEL to the name of the kernel. For example, if you want to only autotune rmsnorm, you can set HELION_AOT_AUTOTUNE_KERNEL="rms_norm_fwd".

    Config files are stored in src/olmo_core/kernels/helion/aot/configs/{config_name}_{gpu_arch}.json
    """
    # Threshold for how much faster a config has to be for some shape to be considered "useful"
    threshold = 1.01
    # How many ms to run the kernel when retuning
    retune_rep_ms = 1000

    def inner_autotune(kernel: helion.Kernel):
        configs_dir = Path(__file__).parent / "configs"
        configs_dir.mkdir(exist_ok=True)

        cc_major, cc_minor = get_cuda_device_capability()
        gpu_arch = f"sm_{cc_major}{cc_minor}"

        path = configs_dir / f"{config_name}_{gpu_arch}.json"

        @functools.wraps(kernel_key)
        def wrapped_kernel_key(*inps: Any) -> KernelKey:
            """
            A wrapper that handles dtype specially (since dtype is not
            serializable to json).
            """

            key = kernel_key(*inps)
            assert isinstance(key.hash_key, tuple)
            assert isinstance(key.exact_key, tuple)
            hash_key = tuple(i.itemsize if isinstance(i, torch.dtype) else i for i in key.hash_key)
            if key.exact_key is not None:
                exact_key = tuple(
                    i.itemsize if isinstance(i, torch.dtype) else i for i in key.exact_key
                )
            else:
                exact_key = key.exact_key
            return KernelKey(key.numeric_key, hash_key, exact_key)

        autotune_mode_str = os.environ.get("HELION_AOT_AUTOTUNE", "none")
        autotune_mode = AOTAutotuneMode.from_str(autotune_mode_str)
        autotune_kernel = os.environ.get("HELION_AOT_AUTOTUNE_KERNEL", "all")
        if autotune_kernel != "all" and autotune_kernel != config_name:
            autotune_mode = AOTAutotuneMode.NONE

        @functools.cache
        def get_configs_from_autotuning(autotune_mode: AOTAutotuneMode):
            if autotune_mode == AOTAutotuneMode.NONE:
                if not path.exists():
                    raise RuntimeError(
                        f"Helion kernel not tuned yet. Run with HELION_AOT_AUTOTUNE=create HELION_AOT_AUTOTUNE_KERNEL={config_name} to generate the config at {path}"
                    )
                return load_autotune_data_from_json(path)

            inputs = sorted(list(primary_inputs()), key=lambda x: wrapped_kernel_key(*x).numeric_key)
            if autotune_mode == AOTAutotuneMode.CREATE:
                useful_configs = []
                for idx, input in enumerate(inputs):
                    log.info(f"Autotuning for {config_name} with key: {wrapped_kernel_key(*input)}")
                    config = kernel.autotune(input)
                    useful_configs.append(repr(config))
            elif autotune_mode == AOTAutotuneMode.RETUNE:
                with open(path, "r") as f:
                    json_obj = json.load(f)
                    useful_configs = list(json_obj["useful_configs"].values())
            elif autotune_mode == AOTAutotuneMode.IMPUTE:
                # Load existing configs if available
                existing_useful_configs = {}
                existing_hash_configs = {}
                if path.exists():
                    existing_useful_configs, existing_hash_configs = load_autotune_data_from_json(
                        path
                    )
                    log.info(f"Loaded {len(existing_hash_configs)} existing configs from {path}")

                # Identify which inputs need autotuning
                missing_inputs = []
                for input in inputs:
                    cur_key = wrapped_kernel_key(*input)
                    if cur_key not in existing_hash_configs:
                        missing_inputs.append(input)

                log.info(f"Found {len(missing_inputs)} missing inputs out of {len(inputs)} total")

                # Autotune only missing inputs
                new_configs = []
                for idx, input in enumerate(missing_inputs):
                    log.info(f"Autotuning for {config_name} with key: {wrapped_kernel_key(*input)}")
                    config = kernel.autotune(input)
                    new_configs.append(repr(config))

                # Combine existing and new configs
                useful_configs = list(existing_useful_configs.values()) + new_configs
            elif autotune_mode == AOTAutotuneMode.ROUTE:
                # Load existing configs - they must exist for route mode
                if not path.exists():
                    raise RuntimeError(
                        f"Route mode requires existing configs. Run with HELION_AOT_AUTOTUNE=create first to generate configs at {path}"
                    )
                existing_useful_configs, existing_hash_configs = load_autotune_data_from_json(path)
                log.info(f"Loaded {len(existing_hash_configs)} existing configs from {path}")

                # Use only existing configs - no new autotuning
                useful_configs = list(existing_useful_configs.values())

            log.info("Candidate useful configs: ")
            for idx, config in enumerate(useful_configs):
                log.info(f"{idx}:, Config: {config}")

            input_timings = []
            if secondary_inputs is not None:
                inputs += list(secondary_inputs())
                inputs = sorted(inputs, key=lambda x: kernel_key(*x).numeric_key)

            for input in inputs:
                cur_input_key = wrapped_kernel_key(*input)
                timings = []
                for idx, config in enumerate(useful_configs):
                    try:
                        cur_kernel = bind_and_compile_kernel(kernel, input, eval(config))
                        timings.append(do_bench(lambda: cur_kernel(*input), rep=retune_rep_ms))  # noqa: B023
                    except Exception as e:
                        log.info(f"Error compiling config {config}: {e}")
                        timings.append(float("inf"))
                input_timings.append((cur_input_key, timings))
            for idx, (key, timing) in enumerate(input_timings):
                log.info(f"Key {key} timings: {' '.join([f'{i:.5f}' for i in timing])}")

            hash_configs_timings: dict[tuple[Any, Any], tuple[float, int | None]] = defaultdict(
                lambda: (float("inf"), None)
            )
            for cur_kernel_key, input_timings in input_timings:
                for config_idx, (config, timing) in enumerate(zip(useful_configs, input_timings)):
                    if timing < hash_configs_timings[cur_kernel_key][0] * threshold:
                        hash_configs_timings[cur_kernel_key] = (timing, config_idx)

            kept_configs = {}
            hash_configs = {k: v[1] for k, v in hash_configs_timings.items()}
            for key, config_idx in hash_configs.items():
                assert config_idx is not None
                kept_configs[config_idx] = useful_configs[config_idx]

            json_obj = {
                "useful_configs": kept_configs,
                "hash_configs": {repr(k): v for k, v in hash_configs.items()},
            }

            with open(path, "w") as f:
                json.dump(json_obj, f, indent=2)
                f.write("\n")
            return load_autotune_data_from_json(path)

        cached_kernels = {}

        def wrapped_func(*args: Any):
            nonlocal cached_kernels
            cur_kernel_key = wrapped_kernel_key(*args)
            has_int64 = any([i.numel() >= 2**31 for i in args if isinstance(i, torch.Tensor)])
            key = (cur_kernel_key, has_int64)
            if key in cached_kernels:
                return cached_kernels[key](*args)
            if has_int64:
                kernel.settings.index_dtype = torch.int64
            useful_configs, hash_configs = get_configs_from_autotuning(autotune_mode)
            key_to_config = {k: useful_configs[v] for k, v in hash_configs.items()}
            if key not in cached_kernels and cur_kernel_key in hash_configs:
                cached_kernels[key] = bind_and_compile_kernel(
                    kernel, args, key_to_config[cur_kernel_key]
                )
            else:
                if warn_on_hash_miss:
                    log.warning(
                        f"No config found for key {cur_kernel_key} for kernel={config_name}. Finding best match."
                    )
                used_config = None

                def config_key_sort(config_key: KernelKey) -> tuple[Any, ...]:
                    return (
                        config_key.exact_key == cur_kernel_key.exact_key,
                        config_key.hash_key == cur_kernel_key.hash_key,
                        config_key.numeric_key <= cur_kernel_key.numeric_key,
                        config_key.numeric_key,
                    )

                sorted_keys = sorted(hash_configs.keys(), key=config_key_sort, reverse=True)
                best_match_key = sorted_keys[0]
                used_config = key_to_config[best_match_key]
                # Verify that the exact_key matches - this is required for correctness
                assert best_match_key.exact_key == cur_kernel_key.exact_key, "Exact key not found in configs"
                cached_kernels[key] = bind_and_compile_kernel(kernel, args, used_config)
            return cached_kernels[key](*args)

        return wrapped_func

    return inner_autotune
