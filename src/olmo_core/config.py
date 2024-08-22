from dataclasses import asdict, dataclass, fields
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
    def as_dict(self, exclude_none: bool = False, recurse: bool = True) -> Dict[str, Any]:
        """
        Convert into a regular Python dictionary.
        """
        if recurse:
            out = asdict(self)  # type: ignore
        else:
            out = {field.name: getattr(self, field.name) for field in fields(self)}
        if exclude_none:
            for k in list(out.keys()):
                v = out[k]
                if v is None:
                    del out[k]
        return out

    @classmethod
    def from_dict(cls: Type[C], data: Dict[str, Any]) -> C:
        schema = om.structured(cls)
        try:
            return cast(C, om.to_object(om.merge(schema, data)))
        except OmegaConfBaseException as e:
            raise OLMoConfigurationError(str(e))
