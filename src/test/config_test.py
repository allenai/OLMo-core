from dataclasses import dataclass
from typing import Optional

from olmo_core.config import Config


@dataclass
class MockConfig(Config):
    name: str = "default"
    x: Optional[int] = None


def test_simple_config_as_dict():
    c = MockConfig()
    assert c.as_dict() == dict(name="default", x=None)
    assert c.as_dict(exclude_none=True) == dict(name="default")
