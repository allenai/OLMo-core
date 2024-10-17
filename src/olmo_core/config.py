from dataclasses import dataclass, fields, is_dataclass, replace
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
    cast,
)

import torch
from omegaconf import OmegaConf as om
from omegaconf.errors import OmegaConfBaseException
from typing_extensions import Self

from .exceptions import OLMoConfigurationError


class StrEnum(str, Enum):
    """
    This is equivalent to Python's :class:`enum.StrEnum` since version 3.11.
    We include this here for compatibility with older version of Python.
    """

    def __str__(self) -> str:
        return self.value

    def __repr__(self) -> str:
        return f"'{str(self)}'"


C = TypeVar("C", bound="Config")


@dataclass
class Config:
    """
    A base class for configuration dataclasses.

    .. important::
        When you subclass this you should still decorate your subclasses with
        :func:`@dataclass <dataclasses.dataclass>`. For example::

            @dataclass
            class MyConfig(Config):
                ...

    .. important::
        Config classes need to be serializable, so you should only use simple types for your fields.
        Though you can use nested configs.
    """

    CLASS_NAME_FIELD = "_CLASS_"
    """
    The name of the class name field inject into the dictionary from :meth:`as_dict()` or
    :meth:`as_config_dict()`.
    """

    def as_dict(
        self,
        *,
        exclude_none: bool = False,
        exclude_private_fields: bool = False,
        include_class_name: bool = False,
        json_safe: bool = False,
        recurse: bool = True,
    ) -> Dict[str, Any]:
        """
        Convert into a regular Python dictionary.

        :param exclude_none: Don't include values that are ``None``.
        :param exclude_private_fields: Don't include private fields.
        :param include_class_name: Include a field for the name of the class.
        :param json_safe: Output only JSON-safe types.
        :param recurse: Recurse into fields that are also configs/dataclasses.
        """

        def iter_fields(d) -> Generator[Tuple[str, Any], None, None]:
            for field in fields(d):
                value = getattr(d, field.name)
                if exclude_none and value is None:
                    continue
                elif exclude_private_fields and field.name.startswith("_"):
                    continue
                else:
                    yield (field.name, value)

        def as_dict(d: Any, recurse: bool = True) -> Any:
            if is_dataclass(d):
                if recurse:
                    out = {k: as_dict(v) for k, v in iter_fields(d)}
                else:
                    out = {k: v for k, v in iter_fields(d)}
                if include_class_name:
                    out[self.CLASS_NAME_FIELD] = d.__class__.__name__
                return out
            elif isinstance(d, dict):
                return {k: as_dict(v) for k, v in d.items()}
            elif isinstance(d, (list, tuple, set)):
                if json_safe:
                    return [as_dict(x) for x in d]
                else:
                    return d.__class__((as_dict(x) for x in d))
            elif d is None or isinstance(d, (float, int, bool, str)):
                return d
            elif json_safe:
                return str(d)
            else:
                return d

        return as_dict(self, recurse=recurse)

    def as_config_dict(self) -> Dict[str, Any]:
        """
        A convenience wrapper around :meth:`as_dict()` for creating JSON-safe dictionaries suitable
        for recording the config.
        """
        return self.as_dict(
            exclude_none=True,
            exclude_private_fields=True,
            include_class_name=True,
            json_safe=True,
            recurse=True,
        )

    def apply(self, func: Callable[["Config"], None]):
        """
        Recursively apply a function to every config instance field, including ``self``.

        :param func: The function to apply.
        """

        def apply(d):
            if isinstance(d, Config):
                func(d)

            if is_dataclass(d):
                for field in fields(d):
                    value = getattr(d, field.name)
                    apply(value)
            elif isinstance(d, dict):
                for value in d.values():
                    apply(value)
            elif isinstance(d, (list, tuple, set)):
                for x in d:
                    apply(x)

        apply(self)

    def validate(self):
        """
        Validate fields in ``self``. This may modify ``self`` in-place.
        """
        pass

    def merge(self, dotlist: List[str]) -> Self:
        """
        Merge self with fields from a "dotlist", creating a new object.

        :param dotlist: A list of field attributes with dot notation, e.g. ``foo.bar=1``.
        """
        try:
            merge_fields = om.from_dotlist(_clean_opts(dotlist))
            merged = om.merge(self, merge_fields)
            out = cast(Self, om.to_object(merged))
            out.apply(lambda c: c.validate())
            return out
        except OmegaConfBaseException as e:
            raise OLMoConfigurationError(str(e))

    def replace(self, **changes) -> Self:
        """
        Creates a new object of the same type, replacing fields with values from ``changes``.
        """
        return replace(self, **changes)

    @classmethod
    def from_dict(cls: Type[C], data: Dict[str, Any], overrides: Optional[List[str]] = None) -> C:
        """
        Initialize from a regular Python dictionary.

        :param data: A Python dictionary.
        :param overrides: A list of field overrides with dot notation, e.g. ``foo.bar=1``.
        """
        try:
            schema = om.structured(cls)
            conf = om.merge(schema, data)
            if overrides:
                conf = om.merge(conf, om.from_dotlist(_clean_opts(overrides)))
            return cast(C, om.to_object(conf))
        except OmegaConfBaseException as e:
            raise OLMoConfigurationError(str(e))


def _clean_opts(opts: List[str]) -> List[str]:
    return [_clean_opt(s) for s in opts]


def _clean_opt(arg: str) -> str:
    if "=" not in arg:
        arg = f"{arg}=True"
    name, val = arg.split("=", 1)
    name = name.strip("-").replace("-", "_")
    return f"{name}={val}"


class DType(StrEnum):
    """
    An enumeration of supported PyTorch data types.
    """

    float32 = "float32"
    bfloat16 = "bfloat16"

    @classmethod
    def from_pt(cls, dtype: torch.dtype) -> "DType":
        if dtype == torch.float32:
            return DType.float32
        elif dtype == torch.bfloat16:
            return DType.bfloat16
        else:
            raise NotImplementedError(dtype)

    def as_pt(self) -> torch.dtype:
        return getattr(torch, self)
