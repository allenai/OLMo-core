import io
import logging
import os
import pickle
import re
import shutil
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Generator, Optional, Tuple, Type, Union

try:
    from functools import cache
except ImportError:
    from functools import lru_cache as cache

import requests
import torch
from cached_path import cached_path
from cached_path.schemes import S3Client, SchemeClient, add_scheme_client
from rich.progress import track

from .aliases import PathOrStr
from .exceptions import (
    OLMoEnvironmentError,
    OLMoInvalidRangeRequestError,
    OLMoNetworkError,
    OLMoUploadError,
)

log = logging.getLogger(__name__)

############################################
## Unified API for local and remote files ##
############################################


def normalize_path(path: PathOrStr) -> str:
    """
    Normalize a path/URL.

    :param path: The path/URL to normalize.
    """
    return str(path).rstrip("/").replace("file://", "")


def join_path(path1: PathOrStr, path2: PathOrStr) -> PathOrStr:
    """
    Join two paths.

    :param path1: The first path.
    :param path2: The second path.

    :returns: The joined result.
    """
    if is_url(path1):
        return f"{normalize_path(path1)}/{normalize_path(path2)}"
    else:
        return Path(path1) / path2


def resource_path(folder: PathOrStr, fname: str, local_cache: Optional[PathOrStr] = None) -> Path:
    """
    Returns an actual path for local or remote file, potentially downloading it if a copy doesn't
    exist locally yet.
    """
    folder = normalize_path(folder)
    if local_cache is not None and (local_path := Path(local_cache) / fname).is_file():
        log.info(f"Found local cache of {fname} at {local_path}")
        return local_path
    else:
        return cached_path(f"{folder}/{fname}", quiet=True)


def is_url(path: PathOrStr) -> bool:
    """
    Check if a path is a URL.

    :param path: Path-like object to check.
    """
    path = normalize_path(path)
    return re.match(r"[a-z0-9]+://.*", str(path)) is not None


def get_file_size(path: PathOrStr) -> int:
    """
    Get the size of a local or remote file in bytes.

    :param path: Path/URL to the file.
    """
    path = normalize_path(path)

    if is_url(path):
        from urllib.parse import urlparse

        parsed = urlparse(str(path))
        if parsed.scheme == "gs":
            return _gcs_file_size(parsed.netloc, parsed.path.strip("/"))
        elif parsed.scheme in ("s3", "r2", "weka"):
            return _s3_file_size(parsed.scheme, parsed.netloc, parsed.path.strip("/"))
        elif parsed.scheme in ("http", "https"):
            return _http_file_size(str(path))
        elif parsed.scheme == "file":
            return get_file_size(str(path).replace("file://", "", 1))
        else:
            raise NotImplementedError(f"file size not implemented for '{parsed.scheme}' files")
    else:
        return os.stat(path).st_size


def get_bytes_range(path: PathOrStr, bytes_start: int, num_bytes: int) -> bytes:
    """
    Get a range of bytes from a local or remote file.

    :param source: Path/URL to the file.
    :param bytes_start: Byte offset to start at.
    :param num_bytes: Number of bytes to get.
    """
    path = normalize_path(path)

    if is_url(path):
        from urllib.parse import urlparse

        parsed = urlparse(str(path))
        if parsed.scheme == "gs":
            return _gcs_get_bytes_range(
                parsed.netloc, parsed.path.strip("/"), bytes_start, num_bytes
            )
        elif parsed.scheme in ("s3", "r2", "weka"):
            return _s3_get_bytes_range(
                parsed.scheme, parsed.netloc, parsed.path.strip("/"), bytes_start, num_bytes
            )
        elif parsed.scheme in ("http", "https"):
            return _http_get_bytes_range(str(path), bytes_start, num_bytes)
        elif parsed.scheme == "file":
            return get_bytes_range(str(path).replace("file://", "", 1), bytes_start, num_bytes)
        else:
            raise NotImplementedError(f"file size not implemented for '{parsed.scheme}' files")
    else:
        with open(path, "rb") as f:
            f.seek(bytes_start)
            return f.read(num_bytes)


def upload(source: PathOrStr, target: str, save_overwrite: bool = False, quiet: bool = False):
    """
    Upload source file to a target location on GCS or S3.

    :param source: Path to the file to upload.
    :param target: Target URL to upload to.
    :param save_overwrite: Overwrite any existing file.
    """
    from urllib.parse import urlparse

    source = Path(normalize_path(source))
    assert source.is_file()
    num_bytes = get_file_size(source)
    if not quiet:
        log.info(f"Uploading {_format_bytes(num_bytes)} from '{source}' to '{target}'...")
    parsed = urlparse(target)

    if parsed.scheme == "gs":
        _gcs_upload(source, parsed.netloc, parsed.path.strip("/"), save_overwrite=save_overwrite)
    elif parsed.scheme in ("s3", "r2", "weka"):
        _s3_upload(
            source,
            parsed.scheme,
            parsed.netloc,
            parsed.path.strip("/"),
            save_overwrite=save_overwrite,
        )
    else:
        raise NotImplementedError(f"Upload not implemented for '{parsed.scheme}' scheme")

    if not quiet:
        log.info(f"Uploaded {_format_bytes(num_bytes)} to '{target}'")


def copy_file(
    source: PathOrStr, target: PathOrStr, save_overwrite: bool = False, quiet: bool = False
):
    """
    Copy a file from ``source`` to ``target``.

    :param source: The path/URL to the source file.
    :param target: The path/URL to the target location.
    :param save_overwrite: Overwrite any existing file.

    :raises FileNotFoundError: If the ``source`` file doesn't exist.
    :raises FileExistsError: If the ``target`` already exists and ``save_overwrite=False``.
    """
    source = normalize_path(source)
    target = normalize_path(target)
    local_source = cached_path(source, quiet=True)
    if is_url(target):
        upload(local_source, target, save_overwrite=save_overwrite, quiet=quiet)
    else:
        target = Path(target)
        if file_exists(target):
            if not save_overwrite:
                raise FileExistsError(target)
            else:
                target.unlink(missing_ok=True)
        else:
            target.parent.mkdir(exist_ok=True, parents=True)
        try:
            os.link(local_source, target, follow_symlinks=True)
        except OSError:
            # Can't hard link across devices, so copy instead.
            shutil.copyfile(local_source, target, follow_symlinks=True)


def copy_dir(
    source: PathOrStr,
    target: PathOrStr,
    save_overwrite: bool = False,
    num_threads: Optional[int] = None,
    quiet: bool = False,
):
    """
    Copy a directory from ``source`` to ``target``.

    :param source: The path/URL to the source directory.
    :param target: The path/URL to the target location.
    :param save_overwrite: Overwrite any existing files.
    :param num_threads: The number of threads to use.

    :raises FileNotFoundError: If the ``source`` dir doesn't exist.
    :raises FileExistsError: If any source files already exist in the ``target`` and ``save_overwrite=False``.
    """
    source = normalize_path(source)
    target = normalize_path(target)

    if num_threads is None:
        from .utils import get_default_thread_count

        num_threads = get_default_thread_count()

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        if not quiet:
            log.info(f"Collecting source files from '{source}' to copy to '{target}'...")

        futures = []
        for source_path in list_directory(source, recurse=True, include_dirs=False):
            assert source_path.startswith(source)
            relative_source_path = source_path.replace(source, "", 1).lstrip("/")
            target_path = join_path(target, relative_source_path)
            futures.append(
                executor.submit(
                    copy_file, source_path, target_path, save_overwrite=save_overwrite, quiet=True
                )
            )

        if not quiet:
            log.info(f"Collected {len(futures)} source files to copy")

        deque(
            track(
                (f.result() for f in as_completed(futures)),
                description="Copying source files...",
                disable=quiet,
                total=len(futures),
            ),
            maxlen=0,
        )


def dir_is_empty(dir: PathOrStr) -> bool:
    """
    Check if a local or remote directory is empty.
    This also returns true if the directory does not exist.

    :param dir: Path/URL to the directory.
    """
    try:
        next(list_directory(dir))
        return False
    except (StopIteration, FileNotFoundError):
        return True


def file_exists(path: PathOrStr) -> bool:
    """
    Check if a local or remote file exists.

    :param path: Path/URL to a file.
    """
    path = normalize_path(path)

    if is_url(path):
        from urllib.parse import urlparse

        parsed = urlparse(str(path))
        if parsed.scheme == "gs":
            try:
                _gcs_file_size(parsed.netloc, parsed.path.strip("/"))
            except FileNotFoundError:
                return False
            else:
                return True
        elif parsed.scheme in ("s3", "r2", "weka"):
            try:
                _s3_file_size(parsed.scheme, parsed.netloc, parsed.path.strip("/"))
            except FileNotFoundError:
                return False
            else:
                return True
        elif parsed.scheme in ("http", "https"):
            return _http_file_exists(str(path))
        else:
            raise NotImplementedError(f"file_exists not implemented for '{parsed.scheme}' files")
    else:
        return Path(path).exists()


def remove_file(path: PathOrStr):
    """
    Remove a local or remote file.

    :param path: The path or URL to the file.

    :raises FileNotFoundError: If the file doesn't exist.
    """
    path = normalize_path(path)

    if is_url(path):
        from urllib.parse import urlparse

        parsed = urlparse(str(path))
        if parsed.scheme == "gs":
            return _gcs_remove_file(parsed.netloc, parsed.path.strip("/"))
        elif parsed.scheme in ("s3", "r2", "weka"):
            return _s3_remove_file(parsed.scheme, parsed.netloc, parsed.path.strip("/"))
        else:
            raise NotImplementedError(f"remove_file not implemented for '{parsed.scheme}' files")
    else:
        Path(path).unlink()


def clear_directory(dir: PathOrStr, force: bool = False):
    """
    Clear out the contents of a local or remote directory.

    .. warning::
        This function is potentially very destructive!

        By default, for safety, this raise a :class:`ValueError` if you attempt to clear a remote
        directory too close to the root of the bucket. Set ``force=True`` to override.

    :param dir: Path/URL to the directory.
    :param force: See note about safety.
    """
    dir = normalize_path(dir)

    if is_url(dir):
        from urllib.parse import urlparse

        parsed = urlparse(str(dir))
        if parsed.scheme in ("s3", "r2", "weka", "gs"):
            prefix = parsed.path.strip("/")
            # For safety (so people don't accidentally delete a whole bunch of important data),
            # ensure prefix is at least 2 folders deep.
            if not force and prefix.count("/") < 2:
                raise ValueError(
                    "For safety, clearing a remote directory this close to the root of a bucket is "
                    "not allowed by default. To override this behavior set ``force=True``."
                )

        if parsed.scheme == "gs":
            prefix = parsed.path.strip("/")
            return _gcs_clear_directory(parsed.netloc, prefix)
        elif parsed.scheme in ("s3", "r2", "weka"):
            prefix = parsed.path.strip("/")
            return _s3_clear_directory(parsed.scheme, parsed.netloc, prefix)
        else:
            raise NotImplementedError(
                f"clear_directory not implemented for '{parsed.scheme}' folders"
            )
    elif Path(dir).is_dir():
        shutil.rmtree(dir, ignore_errors=True)


def list_directory(
    dir: PathOrStr, recurse: bool = False, include_files: bool = True, include_dirs: bool = True
) -> Generator[str, None, None]:
    """
    List the contents of a local or remote directory.
    If ``recurse=False``, only the immediate children of the directory are returned, otherwise
    every sub-folder is recursed into.

    :param dir: Path/URL to the directory.
    :param recurse: Whether to recurse into sub-folders.
    :param include_files: Include regular files in the results.
    :param include_dirs: Include directories in the results.

    :returns: A generator over paths in the directory. If the ``dir`` is a URL, the results will be
        full URLs. If the ``dir`` is a local path, the results will be of the form ``join_path(dir, p)``.

    :raises FileNotFoundError: If the ``source`` file doesn't exist.
    """
    dir = normalize_path(dir)

    if not is_url(dir):
        for p in Path(dir).iterdir():
            if (p.is_file() and include_files) or (p.is_dir() and include_dirs):
                yield str(p)
            if recurse and p.is_dir():
                yield from list_directory(
                    p, recurse=True, include_files=include_files, include_dirs=include_dirs
                )
    else:
        from urllib.parse import urlparse

        parsed = urlparse(dir)
        if parsed.scheme == "gs":
            yield from _gcs_list_directory(
                parsed.netloc,
                parsed.path.strip("/"),
                recurse=recurse,
                include_files=include_files,
                include_dirs=include_dirs,
            )
        elif parsed.scheme in ("s3", "r2", "weka"):
            yield from _s3_list_directory(
                parsed.scheme,
                parsed.netloc,
                parsed.path.strip("/"),
                recurse=recurse,
                include_files=include_files,
                include_dirs=include_dirs,
            )
        else:
            raise NotImplementedError(
                f"list_directory size not implemented for '{parsed.scheme}' URLs"
            )


def glob_directory(pattern: str) -> Generator[str, None, None]:
    """
    Similar to ``glob.glob()`` from the standard library, but works with remote directories as well.

    .. warning::
        Only a subset of glob patterns are supported. Specifically, ``*`` and ``**`` wildcards,
        which the follow the semantics defined here https://docs.python.org/3/library/pathlib.html#pattern-language.
    """
    # Pull out base directory from pattern by finding the first part before any wildcard.
    # Split by '/' and take path components until we hit one with a wildcard.
    parts = pattern.split("/")
    base_parts = []
    for part in parts:
        if "*" in part:
            break
        base_parts.append(part)
    dir = "/".join(base_parts) if base_parts else "."

    # Translate the glob pattern into a regex.
    # For example, "src/examples/**/*.py" --> "^src/examples/.*[^/]*\\.py$".
    pattern_regex = re.compile(
        "^"
        + re.escape(pattern).replace(r"\*\*/", ".*").replace(r"\*\*", ".*").replace(r"\*", "[^/]*")
        + "$"
    )

    for path in list_directory(dir, recurse="**" in pattern):
        if pattern_regex.match(path):
            yield path


def init_client(remote_path: str):
    """
    Initialize the right client for the given remote resource. This is helpful to avoid threading issues
    with boto3.
    """
    if remote_path.startswith("s3://"):
        _get_s3_client("s3")
    elif remote_path.startswith("r2://"):
        _get_s3_client("r2")
    elif remote_path.startswith("weka://"):
        _get_s3_client("weka")


###################################
## Serialization/deserialization ##
###################################


def serialize_to_tensor(x: Any) -> torch.Tensor:
    """
    Serialize an object to a byte tensor using pickle.

    :param x: The pickeable object to serialize.
    """
    serialized_bytes = pickle.dumps(x)
    return torch.frombuffer(bytearray(serialized_bytes), dtype=torch.uint8)


def deserialize_from_tensor(data: torch.Tensor) -> Any:
    """
    Deserialize an object from a byte tensor using pickle.

    :param data: The byte tensor to deserialize.
    """
    assert data.dtype == torch.uint8
    return pickle.loads(bytearray([int(x.item()) for x in data.flatten()]))


######################
## Internal helpers ##
######################


def _wait_before_retry(attempt: int):
    time.sleep(min(0.5 * 2**attempt, 3.0))


def _format_bytes(num: Union[int, float], suffix="B") -> str:
    for unit in ("", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"):
        if abs(num) < 1024.0:
            return f"{num:3.1f}{unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f}Yi{suffix}"


def retriable(
    max_attempts: int = 3,
    retriable_errors: Tuple[Type[Exception], ...] = (
        requests.exceptions.ConnectionError,
        requests.exceptions.Timeout,
        requests.exceptions.ChunkedEncodingError,
    ),
    retry_condition: Optional[Callable[[Exception], bool]] = None,
):
    def decorator(func):
        @wraps(func)
        def new_func(*args, **kwargs):
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as exc:
                    if isinstance(exc, retriable_errors) or (
                        retry_condition is not None and retry_condition(exc)
                    ):
                        if attempt >= max_attempts:
                            # When torch's DataLoader intercepts exceptions, it may try to re-raise them
                            # by recalling their constructor with a single message arg. Torch has some
                            # logic to deal with the absence of a single-parameter constructor, but it
                            # doesn't gracefully handle other possible failures in calling such a constructor
                            # This can cause an irrelevant exception (e.g. KeyError: 'error'), resulting
                            # in us losing the true exception info. To avoid this, we change the exception
                            # to a type that has a single-parameter constructor.
                            raise OLMoNetworkError(
                                f"'{func.__name__}' failed {max_attempts} attempts with: {exc}"
                            ) from exc
                        else:
                            log.warning(
                                f"'{func.__name__}' failed attempt {attempt} with retriable error: {exc}"
                            )
                            _wait_before_retry(attempt)
                    else:
                        raise

        return new_func

    return decorator


######################
## HTTPS IO helpers ##
######################


@retriable()
def _http_file_size(url: str) -> int:
    response = requests.head(url, allow_redirects=True)
    content_length = response.headers.get("content-length")
    assert (
        content_length is not None
    ), f"No content-length header found for {url}. Headers: {dict(response.headers)}"
    return int(content_length)


@retriable(
    retry_condition=lambda exc: (
        isinstance(exc, requests.exceptions.HTTPError)
        and exc.response is not None
        and exc.response.status_code == 502
    ),
)
def _http_get_bytes_range(url: str, bytes_start: int, num_bytes: int) -> bytes:
    response = requests.get(
        url, headers={"Range": f"bytes={bytes_start}-{bytes_start + num_bytes - 1}"}
    )

    if response.status_code == 404:
        raise FileNotFoundError(url)

    response.raise_for_status()

    result = response.content
    # Some web servers silently ignore range requests and send everything
    assert len(result) == num_bytes, f"expected {num_bytes} bytes, got {len(result)}"

    return result


@retriable()
def _http_file_exists(url: str) -> bool:
    response = requests.head(url)
    if response.status_code == 404:
        return False

    response.raise_for_status()
    return True


####################
## GCS IO helpers ##
####################


@cache
def _get_gcs_client():
    from google.cloud import storage as gcs

    return gcs.Client()


def _gcs_is_retriable(exc: Exception) -> bool:
    from google.api_core.exceptions import BadRequest, GatewayTimeout
    from google.api_core.retry import if_transient_error
    from google.auth.exceptions import RefreshError

    return if_transient_error(exc) or isinstance(
        exc,
        (
            requests.exceptions.Timeout,
            BadRequest,  # Weird choice, but Google throws this transiently
            GatewayTimeout,
            RefreshError,
        ),
    )


def _get_gcs_retry():
    from google.api_core.retry import Retry

    return Retry(
        predicate=_gcs_is_retriable,  # NOTE: it appears google might ignore this
        initial=1.0,
        maximum=10.0,
        multiplier=2.0,
        timeout=600.0,
    )


@retriable(retry_condition=_gcs_is_retriable)
def _gcs_file_size(bucket_name: str, key: str) -> int:
    from google.api_core.exceptions import NotFound

    storage_client = _get_gcs_client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(key)
    try:
        blob.reload(retry=_get_gcs_retry())
    except NotFound:
        raise FileNotFoundError(f"gs://{bucket_name}/{key}")
    assert blob.size is not None
    return blob.size


@retriable(retry_condition=_gcs_is_retriable)
def _gcs_remove_file(bucket_name: str, key: str):
    from google.api_core.exceptions import NotFound

    storage_client = _get_gcs_client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(key)
    try:
        blob.reload(retry=_get_gcs_retry())
        bucket.delete_blob(blob.name)
    except NotFound:
        raise FileNotFoundError(f"gs://{bucket_name}/{key}")


@retriable(retry_condition=_gcs_is_retriable)
def _gcs_get_bytes_range(bucket_name: str, key: str, bytes_start: int, num_bytes: int) -> bytes:
    from google.api_core.exceptions import NotFound

    storage_client = _get_gcs_client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(key)
    try:
        blob.reload(retry=_get_gcs_retry())
    except NotFound:
        raise FileNotFoundError(f"gs://{bucket_name}/{key}")
    return blob.download_as_bytes(
        start=bytes_start,
        end=bytes_start + num_bytes - 1,
        retry=_get_gcs_retry(),
        checksum=None,  # type: ignore
    )


@retriable(retry_condition=_gcs_is_retriable)
def _gcs_upload(source: Path, bucket_name: str, key: str, save_overwrite: bool = False):
    storage_client = _get_gcs_client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(key)

    if blob.exists(retry=_get_gcs_retry()) and not save_overwrite:
        raise FileExistsError(
            f"gs://{bucket_name}/{key} already exists. Use save_overwrite to overwrite it."
        )

    try:
        blob.upload_from_filename(
            source,
            # NOTE: mypy and language servers may complain about the type here, but it does
            # not in fact need to be a ConditionalRetry, a plain old Retry is fine.
            retry=_get_gcs_retry(),  # type: ignore
        )
    except Exception as e:
        raise OLMoUploadError(
            f"Failed to upload '{source}' to '{key}' in GCS bucket '{bucket_name}'"
        ) from e


@retriable(retry_condition=_gcs_is_retriable)
def _gcs_clear_directory(bucket_name: str, prefix: str):
    from google.api_core.exceptions import NotFound

    storage_client = _get_gcs_client()

    prefix = prefix.strip("/")
    if prefix:
        prefix += "/"

    try:
        bucket = storage_client.bucket(bucket_name)
        blobs = bucket.list_blobs(prefix=prefix, retry=_get_gcs_retry())
        for blob in blobs:
            bucket.delete_blob(blob.name)
    except NotFound:
        return


def _gcs_list_directory(
    bucket_name: str,
    prefix: str,
    recurse: bool = False,
    include_files: bool = True,
    include_dirs: bool = True,
) -> Generator[str, None, None]:
    from google.api_core.exceptions import NotFound

    storage_client = _get_gcs_client()

    prefix = prefix.strip("/")
    if prefix:
        prefix += "/"

    for match_glob in (
        prefix + "*",  # only immediate files
        prefix + "**/",  # only immediate sub-folders
    ):
        try:
            bucket = storage_client.bucket(bucket_name)
            blobs = bucket.list_blobs(
                prefix=prefix,
                delimiter="/",
                match_glob=match_glob,
                retry=_get_gcs_retry(),
            )
        except NotFound:
            raise FileNotFoundError(f"gs://{bucket_name}/{prefix}")

        # NOTE: need to iterate over these blobs even if not yielding files, otherwise 'blobs.prefixes'
        # won't be populated.
        for blob in blobs:
            if include_files:
                yield f"gs://{bucket_name}/{blob.name}"

        for folder in blobs.prefixes:
            if include_dirs:
                yield f"gs://{bucket_name}/{folder.strip('/')}"
            if recurse:
                yield from _gcs_list_directory(
                    bucket_name,
                    folder,
                    recurse=True,
                    include_files=include_files,
                    include_dirs=include_dirs,
                )


###################
## S3 IO helpers ##
###################


@cache
def _get_s3_client(scheme: str):
    import boto3
    from botocore import UNSIGNED
    from botocore.config import Config

    session = boto3.Session(profile_name=_get_s3_profile_name(scheme))

    credentials = session.get_credentials()
    config = Config(retries={"max_attempts": 10, "mode": "standard"})
    if credentials is None:
        config = config.merge(Config(signature_version=UNSIGNED))

    return session.client(
        "s3",
        endpoint_url=_get_s3_endpoint_url(scheme),
        config=config,
        use_ssl=not int(os.environ.get("OLMO_NO_SSL", "0")),
    )


def _get_s3_profile_name(scheme: str) -> Optional[str]:
    if scheme == "s3":
        # For backwards compatibility, we assume S3 uses the default profile if S3_PROFILE is not set.
        return os.environ.get("S3_PROFILE")
    if scheme == "r2":
        profile_name = os.environ.get("R2_PROFILE")
        if profile_name is None:
            raise OLMoEnvironmentError(
                "R2 profile name is not set. Did you forget to set the 'R2_PROFILE' env var?"
            )

        return profile_name
    if scheme == "weka":
        profile_name = os.environ.get("WEKA_PROFILE")
        if profile_name is None:
            raise OLMoEnvironmentError(
                "WEKA profile name is not set. Did you forget to set the 'WEKA_PROFILE' env var?"
            )

        return profile_name

    raise NotImplementedError(f"Cannot get profile name for scheme {scheme}")


def _get_s3_endpoint_url(scheme: str) -> Optional[str]:
    if scheme == "s3":
        return None
    if scheme == "r2":
        r2_endpoint_url = os.environ.get("R2_ENDPOINT_URL")
        if r2_endpoint_url is None:
            raise OLMoEnvironmentError(
                "R2 endpoint url is not set. Did you forget to set the 'R2_ENDPOINT_URL' env var?"
            )

        return r2_endpoint_url
    if scheme == "weka":
        weka_endpoint_url = os.environ.get("WEKA_ENDPOINT_URL")
        if weka_endpoint_url is None:
            raise OLMoEnvironmentError(
                "WEKA endpoint url is not set. Did you forget to set the 'WEKA_ENDPOINT_URL' env var?"
            )

        return weka_endpoint_url

    raise NotImplementedError(f"Cannot get endpoint url for scheme {scheme}")


def _s3_retry_condition(err: Exception) -> bool:
    import botocore.exceptions as boto_errors

    return isinstance(
        err, (boto_errors.ClientError, boto_errors.HTTPClientError, boto_errors.ConnectionError)
    )


@retriable(retry_condition=_s3_retry_condition)
def _s3_file_size(scheme: str, bucket_name: str, key: str) -> int:
    from botocore.exceptions import ClientError

    try:
        return _get_s3_client(scheme).head_object(Bucket=bucket_name, Key=key)["ContentLength"]
    except ClientError as e:
        if e.response["ResponseMetadata"]["HTTPStatusCode"] == 404:
            raise FileNotFoundError(f"{scheme}://{bucket_name}/{key}") from e
        else:
            raise


@retriable(retry_condition=_s3_retry_condition)
def _s3_remove_file(scheme: str, bucket_name: str, key: str):
    from botocore.exceptions import ClientError

    try:
        return _get_s3_client(scheme).delete_object(Bucket=bucket_name, Key=key)
    except ClientError as e:
        if e.response["ResponseMetadata"]["HTTPStatusCode"] == 404:
            raise FileNotFoundError(f"{scheme}://{bucket_name}/{key}") from e
        else:
            raise


@retriable(retry_condition=_s3_retry_condition)
def _s3_get_bytes_range(
    scheme: str, bucket_name: str, key: str, bytes_start: int, num_bytes: int
) -> bytes:
    from botocore.exceptions import ClientError

    try:
        return (
            _get_s3_client(scheme)
            .get_object(
                Bucket=bucket_name,
                Key=key,
                Range=f"bytes={bytes_start}-{bytes_start + num_bytes - 1}",
            )["Body"]
            .read()
        )
    except ClientError as e:
        if e.response["ResponseMetadata"]["HTTPStatusCode"] == 404:
            raise FileNotFoundError(f"{scheme}://{bucket_name}/{key}") from e
        elif e.response["Error"]["Code"] == "InvalidRange":
            raise OLMoInvalidRangeRequestError(
                f"Invalid range request to '{scheme}://{bucket_name}/{key}' ({bytes_start=}, {num_bytes=})"
            )
        else:
            raise


@retriable(retry_condition=_s3_retry_condition)
def _s3_upload(
    source: Path,
    scheme: str,
    bucket_name: str,
    key: str,
    save_overwrite: bool = False,
):
    from botocore.exceptions import ClientError

    if not save_overwrite:
        try:
            _get_s3_client(scheme).head_object(Bucket=bucket_name, Key=key)
            raise FileExistsError(
                f"s3://{bucket_name}/{key} already exists. Use save_overwrite to overwrite it."
            )
        except ClientError as e:
            if e.response["ResponseMetadata"]["HTTPStatusCode"] == 404:
                pass
            else:
                raise

    _get_s3_client(scheme).upload_file(source, bucket_name, key)


@retriable(retry_condition=_s3_retry_condition)
def _s3_clear_directory(scheme: str, bucket_name: str, prefix: str):
    from botocore.exceptions import ClientError

    if not prefix.endswith("/"):
        prefix = prefix + "/"

    try:
        for o in _get_s3_client(scheme).list_objects_v2(Bucket=bucket_name, Prefix=prefix)[
            "Contents"
        ]:
            _get_s3_client(scheme).delete_object(Bucket=bucket_name, Key=o["Key"])
        return
    except ClientError as e:
        if e.response["ResponseMetadata"]["HTTPStatusCode"] == 404:
            return
    except KeyError:
        return


def _s3_list_directory(
    scheme: str,
    bucket_name: str,
    prefix: str,
    recurse: bool = False,
    include_files: bool = True,
    include_dirs: bool = True,
) -> Generator[str, None, None]:
    client = _get_s3_client(scheme)
    paginator = client.get_paginator("list_objects_v2")
    if not prefix.endswith("/"):
        prefix = prefix + "/"
    for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix, MaxKeys=50, Delimiter="/"):
        if include_files:
            for file_item in page.get("Contents", []):
                yield f"{scheme}://{bucket_name}/{file_item['Key']}"
        for dir_item in page.get("CommonPrefixes", []):
            if include_dirs:
                yield f"{scheme}://{bucket_name}/{dir_item['Prefix'].strip('/')}"
            if recurse:
                yield from _s3_list_directory(
                    scheme,
                    bucket_name,
                    dir_item["Prefix"],
                    recurse=True,
                    include_files=include_files,
                    include_dirs=include_dirs,
                )


#############################################
## Custom cached path client for 'weka://' ##
#############################################


def add_cached_path_clients():
    """
    Add additional cached-path clients.
    """
    add_scheme_client(_WekaClient)


class _WekaClient(SchemeClient):
    recoverable_errors = S3Client.recoverable_errors

    scheme = "weka"

    def __init__(self, resource: str) -> None:
        super().__init__(resource)
        self.bucket_name, self.path = _WekaClient._split_cloud_path(resource, "weka")
        self.s3 = _get_s3_client("weka")
        self.object_info = None

    @staticmethod
    def _split_cloud_path(url: str, provider: str) -> Tuple[str, str]:
        """Split a full s3 path into the bucket name and path."""
        from urllib.parse import urlparse

        parsed = urlparse(url)
        if not parsed.netloc or not parsed.path:
            raise ValueError("bad {} path {}".format(provider, url))
        bucket_name = parsed.netloc
        provider_path = parsed.path
        # Remove '/' at beginning of path.
        if provider_path.startswith("/"):
            provider_path = provider_path[1:]
        return bucket_name, provider_path

    def _ensure_object_info(self):
        import botocore.exceptions as boto_exceptions

        if self.object_info is None:
            try:
                self.object_info = self.s3.head_object(Bucket=self.bucket_name, Key=self.path)
            except boto_exceptions.ClientError as e:
                if e.response["ResponseMetadata"]["HTTPStatusCode"] == 404:
                    raise FileNotFoundError(f"weka://{self.bucket_name}/{self.path}") from e
                raise e

    def get_etag(self) -> Optional[str]:
        self._ensure_object_info()
        assert self.object_info is not None
        return self.object_info.get("ETag")

    def get_size(self) -> Optional[int]:
        self._ensure_object_info()
        assert self.object_info is not None
        return self.object_info.get("ContentLength")

    def get_resource(self, temp_file: io.BufferedWriter) -> None:
        self.s3.download_fileobj(Fileobj=temp_file, Bucket=self.bucket_name, Key=self.path)

    def get_bytes_range(self, index: int, length: int) -> bytes:
        response = self.s3.get_object(
            Bucket=self.bucket_name, Key=self.path, Range=f"bytes={index}-{index + length - 1}"
        )
        return response["Body"].read()
