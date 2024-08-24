from dataclasses import dataclass
from typing import List, Optional, Set, Tuple

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
        _z: int = 0

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

    foo2 = Foo.from_dict(data, overrides=["bar.x=0"])
    assert foo2.bar.x == 0
    foo3 = foo2.merge(["bar.x=-1"])
    assert foo3.bar.x == -1

    assert foo.as_dict(recurse=False) == {"z": "z", "bar": foo.bar}

    assert foo.as_config_dict() == {
        Config.CLASS_NAME_FIELD: "Foo",
        "z": "z",
        "bar": {
            Config.CLASS_NAME_FIELD: "Bar",
            "x": 1,
            "y": 2,
        },
    }


def test_json_safe_dump():
    @dataclass
    class Foo(Config):
        x_list: List[int]
        x_tuple: Tuple[int, ...]
        x_set: Set[str]

    foo = Foo(x_list=[0, 1], x_tuple=(0, 1), x_set={"a"})
    assert foo.as_config_dict() == {
        Config.CLASS_NAME_FIELD: "Foo",
        "x_list": [0, 1],
        "x_tuple": [0, 1],
        "x_set": ["a"],
    }
