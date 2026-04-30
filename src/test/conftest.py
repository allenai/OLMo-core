import os
import uuid
from functools import partial
from typing import Generator

import pytest
import torch
import torch.nn as nn

from olmo_core.io import clear_directory


@pytest.fixture
def bucket_name() -> str:
    return "ai2-olmo-testing"


@pytest.fixture
def gcs_bucket_name() -> str:
    return "ai2-olmo-testing"


@pytest.fixture
def unique_name() -> str:
    return uuid.uuid4().hex


def _has_aws_credentials() -> bool:
    return bool(os.environ.get("AWS_ACCESS_KEY_ID")) and bool(
        os.environ.get("AWS_SECRET_ACCESS_KEY")
    )


@pytest.fixture
def s3_checkpoint_dir(bucket_name, unique_name) -> Generator[str, None, None]:
    from botocore.exceptions import NoCredentialsError

    if not _has_aws_credentials():
        pytest.skip("Requires AWS credentials")

    folder = f"s3://{bucket_name}/checkpoints/{unique_name}"
    yield folder

    try:
        clear_directory(folder, force=True)
    except NoCredentialsError:
        pass


@pytest.fixture
def gcs_checkpoint_dir(gcs_bucket_name, unique_name) -> Generator[str, None, None]:
    from google.auth.exceptions import DefaultCredentialsError

    folder = f"gs://{gcs_bucket_name}/checkpoints/{unique_name}"
    yield folder

    try:
        clear_directory(folder, force=True)
    except DefaultCredentialsError:
        pass


class TinyModel(nn.Module):
    def __init__(self, dim: int = 8):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.ReLU(),
            nn.Linear(dim * 2, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, x):
        return self.fc(x)


@pytest.fixture
def tiny_model_factory():
    return TinyModel


@pytest.fixture
def tiny_model(tiny_model_factory) -> TinyModel:
    return tiny_model_factory()


@pytest.fixture
def tiny_model_data_factory():
    return partial(torch.rand, 2, 8)


@pytest.fixture
def tiny_model_data(tiny_model_data_factory) -> torch.Tensor:
    return tiny_model_data_factory()
