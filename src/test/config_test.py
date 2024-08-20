from dataclasses import dataclass
from typing import Optional

from olmo_core.config import Config


def test_simple_config_as_dict():
    @dataclass
    class MockConfig(Config):
        name: str = "default"
        x: Optional[int] = None

    c = MockConfig()
    assert c.as_dict() == dict(name="default", x=None)
    assert c.as_dict(exclude_none=True) == dict(name="default")


def test_nested_configs():
    @dataclass
    class Bar:
        x: int
        y: int

    @dataclass
    class Foo(Config):
        bar: Bar
        z: str

    foo = Foo(bar=Bar(x=1, y=2), z="z")
    data = foo.as_dict()
    assert isinstance(data["bar"], dict)

    foo1 = Foo.from_dict(data)
    assert foo1 == foo
    assert isinstance(foo1.bar, Bar)