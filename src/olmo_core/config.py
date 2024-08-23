from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from typing import Any, Dict, Type, TypeVar, cast

from omegaconf import OmegaConf as om
from omegaconf.errors import OmegaConfBaseException

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
        When you subclass this you should still decorate your subclasses with ``@dataclass``.
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
        :param json_safe: Output only JSON-safe types if possible.
        :param recurse: Recurse into fields that are also configs/dataclasses.
        """

        def as_dict(d: Any, recurse: bool = True) -> Any:
            if is_dataclass(d):
                if recurse:
                    out = {field.name: as_dict(getattr(d, field.name)) for field in fields(d)}
                else:
                    out = {field.name: getattr(d, field.name) for field in fields(d)}
                for k in list(out.keys()):
                    v = out[k]
                    if (exclude_none and v is None) or (
                        exclude_private_fields and k.startswith("_")
                    ):
                        del out[k]
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
            elif isinstance(d, (float, int, bool, str)):
                return d
            elif json_safe:
                raise TypeError(f"Cannot convert type '{type(d)}' to a JSON-safe representation")
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

    @classmethod
    def from_dict(cls: Type[C], data: Dict[str, Any]) -> C:
        """
        Initialize from a raw Python dictionary.
        """
        schema = om.structured(cls)
        try:
            return cast(C, om.to_object(om.merge(schema, data)))
        except OmegaConfBaseException as e:
            raise OLMoConfigurationError(str(e))
