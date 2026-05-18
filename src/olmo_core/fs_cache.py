import functools as ft
import hashlib
import json
import logging
import os
import pickle
import tempfile
import typing
from pathlib import Path
from typing import Callable, TypeVar

from dataclass_extensions import encode
from filelock import FileLock

log = logging.getLogger(__name__)

CACHE_DIR_ENV_VAR = "OLMO_CORE_FS_CACHE_DIR"


F = TypeVar("F", bound=Callable[..., object])


def maybe_cache(*, condition: Callable[..., bool] | None = None) -> Callable[[F], F]:
    f"""
    Similar ``functools.cache``, but uses a persistent cache on the filesystem when the env var
    '{CACHE_DIR_ENV_VAR}' is set, otherwise caching is disabled.

    Arguments must be JSON-serializable. The result must be pickle-able.
    """

    def decorator(user_function: F) -> F:
        @ft.wraps(user_function)
        def wrapper(*args, **kwargs):
            cache_dir: Path | None = None
            if (condition is None or condition(*args, **kwargs)) and (
                _cache_dir := os.environ.get(CACHE_DIR_ENV_VAR)
            ) is not None:
                cache_dir = Path(_cache_dir)

                # Hash the arguments to create a unique cache key.
                key = f"{user_function.__qualname__}-{_deterministic_hash((args, kwargs))}"
                cache_dir.mkdir(parents=True, exist_ok=True)
                lock_path = cache_dir / f"{key}.lock"
                cache_path = cache_dir / f"{key}.pkl"
                with FileLock(lock_path):
                    if cache_path.exists():
                        log.debug(
                            "Loading result for %s() from filesystem cache...",
                            user_function.__qualname__,
                        )
                        # Cache hit. Load and return the cached result.
                        with cache_path.open("rb") as f:
                            return pickle.load(f)
                    else:
                        # Cache miss. Call the user function and cache the result.
                        log.debug(
                            "Computing result for %s() and storying to filesystem cache...",
                            user_function.__qualname__,
                        )
                        result = user_function(*args, **kwargs)
                        tmp_file = tempfile.NamedTemporaryFile(
                            mode="wb", dir=cache_dir, prefix=key, suffix=".tmp", delete=False
                        )
                        tmp_path = Path(tmp_file.name)
                        try:
                            pickle.dump(result, tmp_file)

                            # Ensure all data is written to disk.
                            tmp_file.flush()
                            if hasattr(os, "fdatasync"):  # only available on linux
                                os.fdatasync(tmp_file)  # type: ignore
                            tmp_file.close()

                            # Copy to final destination.
                            tmp_path.replace(cache_path)
                        finally:
                            tmp_file.close()
                            tmp_path.unlink(missing_ok=True)
                        return result
            else:
                return user_function(*args, **kwargs)

        return typing.cast(F, wrapper)

    return decorator


def _deterministic_hash(obj) -> str:
    # For a simple, consistent hash, we can use a JSON representation
    # JSON standardizes order for dict keys, which helps with determinism.
    # The 'sort_keys=True' argument is important.
    # The output of json.dumps() must be encoded to bytes before hashing.
    encoded_object = json.dumps(encode(obj), sort_keys=True).encode("utf-8")

    # Use SHA256 for a robust, standard cryptographic hash
    return hashlib.sha256(encoded_object).hexdigest()
