from dataclasses import asdict, dataclass, is_dataclass
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

    def as_dict(
        self,
        *,
        exclude_none: bool = False,
        exclude_private_fields: bool = False,
        recurse: bool = True,
    ) -> Dict[str, Any]:
        """
        Convert into a regular Python dictionary.

        :param exclude_none: Don't include values that are ``None``.
        :param exclude_private_fields: Don't include private fields.
        :param recurse: Recurse into fields that are also configs/dataclasses.
        """

        def dict_factory(d) -> Dict[str, Any]:
            out = dict(d)
            for k in list(out.keys()):
                v = out[k]
                if (exclude_none and v is None) or (exclude_private_fields and k.startswith("_")):
                    del out[k]
            return out

        if recurse:
            return asdict(self, dict_factory=dict_factory)
        else:
            return dict_factory(self)

    def as_config_dict(self) -> Dict[str, Any]:
        """
        A convenience wrapper around :meth:`as_dict()` for creating dictionaries suitable
        for recording the config.
        """
        out = self.as_dict(exclude_none=True, exclude_private_fields=True, recurse=True)
        out["CLASS"] = self.__class__.__name__
        return out

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
