import uuid
from typing import Generator

import pytest

from olmo_core.io import clear_directory


@pytest.fixture
def bucket_name() -> str:
    return "ai2-olmo-testing"


@pytest.fixture
def unique_name() -> str:
    return uuid.uuid4().hex


@pytest.fixture
def s3_checkpoint_dir(bucket_name, unique_name) -> Generator[str, None, None]:
    folder = f"s3://{bucket_name}/checkpoints/{unique_name}"
    yield folder
    clear_directory(folder)
