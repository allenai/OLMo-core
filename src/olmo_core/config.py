from dataclasses import Field, asdict, dataclass, fields, is_dataclass
from enum import Enum
from typing import Any, Dict, Type, TypeVar


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
        def init_field_from_data(field: Field, d: Any) -> Any:
            if is_dataclass(field.type) and isinstance(d, dict):
                return field.type(
                    **{
                        field.name: init_field_from_data(field, d.get(field.name))
                        for field in fields(field.type)
                    }
                )
            else:
                return d

        return cls(
            **{
                field.name: init_field_from_data(field, data.get(field.name))
                for field in fields(cls)
            }
        )
