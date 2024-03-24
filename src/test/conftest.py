import uuid
from typing import Generator

import pytest
import torch
import torch.nn as nn

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


class TinyModel(nn.Module):
    def __init__(self, dim: int = 8):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(dim, dim * 2), nn.ReLU(), nn.Linear(dim * 2, dim), nn.ReLU(), nn.Linear(dim, dim)
        )

    def forward(self, x):
        return self.fc(x)


@pytest.fixture
def tiny_model() -> TinyModel:
    return TinyModel(8)


@pytest.fixture
def tiny_model_data() -> torch.Tensor:
    return torch.rand(2, 8)
