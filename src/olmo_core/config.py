from dataclasses import asdict, fields
from enum import Enum
from typing import Any, Dict


class StrEnum(str, Enum):
    """
    This is equivalent to Python's :class:`enum.StrEnum` since version 3.11.
    We include this here for compatibility with older version of Python.
    """

    def __str__(self) -> str:
        return self.value

    def __repr__(self) -> str:
        return f"'{str(self)}'"


class Config:
    def as_dict(self, exclude_none: bool = False, recurse: bool = True) -> Dict[str, Any]:
        """
        Convert into a regular Python dictionary.
        """
        if recurse:
            out = asdict(self)  # type: ignore
        else:
            out = {field.name: getattr(self, field.name) for field in fields(self)}  # type: ignore
        if exclude_none:
            for k in list(out.keys()):
                v = out[k]
                if v is None:
                    del out[k]
        return out
