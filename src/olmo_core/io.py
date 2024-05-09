import logging
import os
import pickle
import re
import shutil
import time
from os import PathLike
from pathlib import Path
from typing import Any, Optional, Union

try:
    from functools import cache
except ImportError:
    from functools import lru_cache as cache

import torch

from .exceptions import OLMoEnvironmentError, OLMoNetworkError

log = logging.getLogger(__name__)

PathOrStr = Union[Path, PathLike, str]

############################################
## Unified API for local and remote files ##
############################################


def is_url(path: PathOrStr) -> bool:
    """
    Check if a path is a URL.

    :param path: Path-like object to check.
    """
    return re.match(r"[a-z0-9]+://.*", str(path)) is not None


def file_size(path: PathOrStr) -> int:
    """
    Get the size of a local or remote file in bytes.

    :param path: Path/URL to the file.
    """
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
            return file_size(str(path).replace("file://", "", 1))
        else:
            raise NotImplementedError(f"file size not implemented for '{parsed.scheme}' files")
    else:
        return os.stat(path).st_size


def get_bytes_range(path: PathOrStr, bytes_start: int, num_bytes: int) -> bytes:
    """
    Get a range of bytes from a file.

    :param source: Path/URL to the file.
    :param bytes_start: Byte offset to start at.
    :param num_bytes: Number of bytes to get.
    """
    if is_url(path):
        from urllib.parse import urlparse

        parsed = urlparse(str(path))
        if parsed.scheme == "gs":
            return _gcs_get_bytes_range(parsed.netloc, parsed.path.strip("/"), bytes_start, num_bytes)
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


def upload(source: PathOrStr, target: str, save_overwrite: bool = False):
    """
    Upload source file to a target location on GCS or S3.

    :param source: Path to the file to upload.
    :param target: Target URL to upload to.
    :param save_overwrite: Overwrite any existing file.
    """
    from urllib.parse import urlparse

    source = Path(source)
    assert source.is_file()
    num_bytes = file_size(source)
    log.info(f"Uploading {_format_bytes(num_bytes)} from '{source}' to '{target}'...")
    parsed = urlparse(target)
    if parsed.scheme == "gs":
        _gcs_upload(source, parsed.netloc, parsed.path.strip("/"), save_overwrite=save_overwrite)
    elif parsed.scheme in ("s3", "r2", "weka"):
        _s3_upload(source, parsed.scheme, parsed.netloc, parsed.path.strip("/"), save_overwrite=save_overwrite)
    else:
        raise NotImplementedError(f"Upload not implemented for '{parsed.scheme}' scheme")


def dir_is_empty(dir: PathOrStr) -> bool:
    """
    Check if a local directory is empty. This also returns true if the directory does not exist.

    :param dir: Path to the local directory.
    """
    dir = Path(dir)
    if not dir.is_dir():
        return True
    try:
        next(dir.glob("*"))
        return False
    except StopIteration:
        return True


def file_exists(path: PathOrStr) -> bool:
    """
    Check if a file exists.

    :param path: Path/URL to a file.
    """
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
        elif parsed.scheme == "file":
            return file_exists(str(path).replace("file://", "", 1))
        else:
            raise NotImplementedError(f"file_exists not implemented for '{parsed.scheme}' files")
    else:
        return Path(path).exists()


def clear_directory(dir: PathOrStr):
    """
    Clear out the contents of a local or remote directory. GCS (``gs://``) and S3 (``s3://``) URLs are supported.

    :param dir: Path/URL to the directory.
    """
    if is_url(dir):
        from urllib.parse import urlparse

        parsed = urlparse(str(dir))
        if parsed.scheme in ("s3", "r2", "weka"):
            return _s3_clear_directory(parsed.scheme, parsed.netloc, parsed.path.strip("/"))
        elif parsed.scheme == "file":
            return clear_directory(str(dir).replace("file://", "", 1))
        else:
            raise NotImplementedError(f"clear_directory not implemented for '{parsed.scheme}' folders")
    elif Path(dir).is_dir():
        shutil.rmtree(dir, ignore_errors=True)


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


######################
## HTTPS IO helpers ##
######################


def _http_file_size(url: str) -> int:
    import requests

    response = requests.head(url, allow_redirects=True)
    content_length = response.headers.get("content-length")
    assert content_length
    return int(content_length)


def _http_get_bytes_range(url: str, bytes_start: int, num_bytes: int) -> bytes:
    import requests

    response = requests.get(url, headers={"Range": f"bytes={bytes_start}-{bytes_start+num_bytes-1}"})
    if response.status_code == 404:
        raise FileNotFoundError(url)

    response.raise_for_status()

    result = response.content
    # Some web servers silently ignore range requests and send everything
    assert len(result) == num_bytes, f"expected {num_bytes} bytes, got {len(result)}"

    return result


def _http_file_exists(url: str) -> bool:
    import requests

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


def _gcs_file_size(bucket_name: str, key: str) -> int:
    from google.api_core.exceptions import NotFound

    storage_client = _get_gcs_client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(key)
    try:
        blob.reload()
    except NotFound:
        raise FileNotFoundError(f"gs://{bucket_name}/{key}")
    assert blob.size is not None
    return blob.size


def _gcs_get_bytes_range(bucket_name: str, key: str, bytes_start: int, num_bytes: int) -> bytes:
    from google.api_core.exceptions import NotFound

    storage_client = _get_gcs_client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(key)
    try:
        blob.reload()
    except NotFound:
        raise FileNotFoundError(f"gs://{bucket_name}/{key}")
    return blob.download_as_bytes(start=bytes_start, end=bytes_start + num_bytes - 1)


def _gcs_upload(source: Path, bucket_name: str, key: str, save_overwrite: bool = False):
    storage_client = _get_gcs_client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(key)
    if not save_overwrite and blob.exists():
        raise FileExistsError(f"gs://{bucket_name}/{key} already exists. Use save_overwrite to overwrite it.")
    blob.upload_from_filename(source)


###################
## S3 IO helpers ##
###################


@cache
def _get_s3_client(scheme: str):
    import boto3
    from botocore.config import Config

    session = boto3.Session(profile_name=_get_s3_profile_name(scheme))
    return session.client(
        "s3",
        endpoint_url=_get_s3_endpoint_url(scheme),
        config=Config(retries={"max_attempts": 10, "mode": "standard"}),
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


def _s3_file_size(scheme: str, bucket_name: str, key: str, max_attempts: int = 3) -> int:
    from botocore.exceptions import ClientError

    err: Optional[Exception] = None
    for attempt in range(1, max_attempts + 1):
        try:
            return _get_s3_client(scheme).head_object(Bucket=bucket_name, Key=key)["ContentLength"]
        except ClientError as e:
            if e.response["ResponseMetadata"]["HTTPStatusCode"] == 404:
                raise FileNotFoundError(f"s3://{bucket_name}/{key}") from e
            err = e

        if attempt < max_attempts:
            log.warning("%s failed attempt %d with retriable error: %s", _s3_file_size.__name__, attempt, err)
            _wait_before_retry(attempt)

    raise OLMoNetworkError("Failed to get s3 file size") from err


def _s3_get_bytes_range(
    scheme: str, bucket_name: str, key: str, bytes_start: int, num_bytes: int, max_attempts: int = 3
) -> bytes:
    from botocore.exceptions import ClientError, ConnectionError, HTTPClientError

    err: Optional[Exception] = None
    for attempt in range(1, max_attempts + 1):
        try:
            return (
                _get_s3_client(scheme)
                .get_object(
                    Bucket=bucket_name, Key=key, Range=f"bytes={bytes_start}-{bytes_start + num_bytes - 1}"
                )["Body"]
                .read()
            )
        except ClientError as e:
            if e.response["ResponseMetadata"]["HTTPStatusCode"] == 404:
                raise FileNotFoundError(f"s3://{bucket_name}/{key}") from e
            err = e
        except (HTTPClientError, ConnectionError) as e:
            # ResponseStreamingError (subclass of HTTPClientError) can happen as
            # a result of a failed read from the stream (http.client.IncompleteRead).
            # Retrying can help in this case.
            err = e

        if attempt < max_attempts:
            log.warning(
                "%s failed attempt %d with retriable error: %s", _s3_get_bytes_range.__name__, attempt, err
            )
            _wait_before_retry(attempt)

    # When torch's DataLoader intercepts exceptions, it may try to re-raise them
    # by recalling their constructor with a single message arg. Torch has some
    # logic to deal with the absence of a single-parameter constructor, but it
    # doesn't gracefully handle other possible failures in calling such a constructor
    # This can cause an irrelevant exception (e.g. KeyError: 'error'), resulting
    # in us losing the true exception info. To avoid this, we change the exception
    # to a type that has a single-parameter constructor.
    raise OLMoNetworkError("Failed to get bytes range from s3") from err


def _s3_upload(
    source: Path, scheme: str, bucket_name: str, key: str, save_overwrite: bool = False, max_attempts: int = 3
):
    from botocore.exceptions import ClientError

    err: Optional[Exception] = None
    if not save_overwrite:
        for attempt in range(1, max_attempts + 1):
            try:
                _get_s3_client(scheme).head_object(Bucket=bucket_name, Key=key)
                raise FileExistsError(
                    f"s3://{bucket_name}/{key} already exists. Use save_overwrite to overwrite it."
                )
            except ClientError as e:
                if e.response["ResponseMetadata"]["HTTPStatusCode"] == 404:
                    err = None
                    break
                err = e

            if attempt < max_attempts:
                log.warning("%s failed attempt %d with retriable error: %s", _s3_upload.__name__, attempt, err)
                _wait_before_retry(attempt)

        if err is not None:
            raise OLMoNetworkError("Failed to check object existence during s3 upload") from err

    try:
        _get_s3_client(scheme).upload_file(source, bucket_name, key)
    except ClientError as e:
        raise OLMoNetworkError("Failed to upload to s3") from e


def _s3_clear_directory(scheme: str, bucket_name: str, prefix: str, max_attempts: int = 3):
    from botocore.exceptions import ClientError

    if not prefix.endswith("/"):
        prefix = prefix + "/"

    err: Optional[Exception] = None
    for attempt in range(1, max_attempts + 1):
        try:
            for o in _get_s3_client(scheme).list_objects_v2(Bucket=bucket_name, Prefix=prefix)["Contents"]:
                _get_s3_client(scheme).delete_object(Bucket=bucket_name, Key=o["Key"])
            return
        except ClientError as e:
            if e.response["ResponseMetadata"]["HTTPStatusCode"] == 404:
                return
            err = e

        if attempt < max_attempts:
            log.warning("%s failed attempt %d with retriable error: %s", _s3_upload.__name__, attempt, err)
            _wait_before_retry(attempt)

    raise OLMoNetworkError("Failed to remove S3 directory") from err
