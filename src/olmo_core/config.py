import copy
import dataclasses
import json
from dataclasses import dataclass
from enum import Enum
from typing import (
    Any,
    Callable,
    ClassVar,
    Collection,
    Dict,
    Generator,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
)

import torch
import yaml
from cached_path import cached_path
from dataclass_extensions import Registrable, decode
from typing_extensions import Self

from .aliases import PathOrStr
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

    _IGNORE_FIELDS: ClassVar[Tuple[str, ...]] = ()
    """
    Fields to ignore when loading from config (for backwards compatibility).
    """

    def as_dict(
        self,
        *,
        exclude_none: bool = False,
        exclude_private_fields: bool = False,
        exclude: Optional[Collection[str]] = None,
        include_class_name: bool = False,
        json_safe: bool = False,
        recurse: bool = True,
    ) -> Dict[str, Any]:
        """
        Convert into a regular Python dictionary.

        :param exclude_none: Don't include values that are ``None``.
        :param exclude_private_fields: Don't include private fields.
        :param exclude: A list of field names to exclude.
        :param include_class_name: Include a field for the name of the class.
        :param json_safe: Output only JSON-safe types.
        :param recurse: Recurse into fields that are also configs/dataclasses.
        """

        exclude_set = set(exclude) if exclude is not None else set()

        def iter_fields(d) -> Generator[Tuple[str, Any], None, None]:
            for field in dataclasses.fields(d):
                if field.name in exclude_set:
                    continue
                value = getattr(d, field.name)
                if exclude_none and value is None:
                    continue
                elif exclude_private_fields and field.name.startswith("_"):
                    continue
                else:
                    yield (field.name, value)

        def as_dict(d: Any, recurse: bool = True) -> Any:
            if dataclasses.is_dataclass(d):
                if recurse:
                    out = {k: as_dict(v) for k, v in iter_fields(d)}
                else:
                    out = {k: v for k, v in iter_fields(d)}
                if include_class_name:
                    out[self.CLASS_NAME_FIELD] = f"{d.__class__.__module__}.{d.__class__.__name__}"
                if isinstance(d, Registrable):
                    try:
                        registered_name = d.get_registered_name()
                        out["type"] = registered_name
                    except ValueError:
                        pass
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

            if dataclasses.is_dataclass(d):
                for field in dataclasses.fields(d):
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

    def merge(self, dotlist: List[str], prefix: Optional[str] = None, strict: bool = True) -> Self:
        """
        Merge self with fields from a "dotlist", creating a new object.

        :param dotlist: A list of field attributes with dot notation, e.g. ``foo.bar=1``.
        :param prefix: Only use override items in the dotlist that start with a given prefix name,
            and strip that prefix (including the subsequent ".") before applying the overrides.
        :param strict: Parse the dotlist strictly.
        """
        overrides = _clean_opts(dotlist)
        if prefix is not None:
            overrides = [
                (k.replace(f"{prefix}.", "", 1), v)
                for k, v in overrides
                if k.startswith(f"{prefix}.")
            ]

        if not strict:
            field_names = set(f.name for f in dataclasses.fields(self))
            overrides = [
                (k, v)
                for k, v in overrides
                if any([k == name or k.startswith(f"{name}.") for name in field_names])
            ]

        merged_data = self.as_dict(include_class_name=True)
        for key, value in overrides:
            _set_nested(merged_data, key, value)
        return self.from_dict(merged_data)

    def replace(self, **changes) -> Self:
        """
        Creates a new object of the same type, replacing fields with values from ``changes``.
        """
        return dataclasses.replace(self, **changes)

    def copy(self, deep: bool = True) -> Self:
        """
        Creates a new object of the same type, with the same values.
        """
        return copy.deepcopy(self) if deep else copy.copy(self)

    @classmethod
    def from_dict(cls: Type[C], data: Dict[str, Any], overrides: Optional[List[str]] = None) -> C:
        """
        Initialize from a regular Python dictionary.

        :param data: A Python dictionary.
        :param overrides: A list of field overrides with dot notation, e.g. ``foo.bar=1``.
        """
        from importlib import import_module

        def resolve_cls(cls_name: str) -> Optional[Any]:
            if "." in cls_name:
                *modules, cls_name = cls_name.split(".")
                module_name = ".".join(modules)
                module = import_module(module_name)
                return getattr(module, cls_name, None)
            else:
                return None

        def decode_data(d: Any, prefix: str) -> Any:
            if isinstance(d, dict):
                # HACK: Try to convert string keys to int if they look like integers. Handles cases
                # where integer keys were serialized as strings (eg "block_overrides")
                d = {(int(k) if isinstance(k, str) and k.isdigit() else k): v for k, v in d.items()}

                new_dict = {
                    str(k): decode_data(v, f"{prefix}.{k}" if prefix else str(k))
                    for k, v in d.items()
                    if k != cls.CLASS_NAME_FIELD
                }
                if (cls_name := d.get(cls.CLASS_NAME_FIELD)) is not None and (
                    cls_o := resolve_cls(cls_name)
                ) is not None:
                    # Remove ignored fields if the class defines any
                    if (ignore_fields := getattr(cls_o, "_IGNORE_FIELDS", None)) is not None:
                        new_dict = {k: v for k, v in new_dict.items() if k not in ignore_fields}

                    try:
                        return decode(cls_o, new_dict)
                    except Exception as e:
                        if prefix:
                            msg = f"Failed to construct '{prefix}' in config: {e}"
                        else:
                            msg = f"Error building config: {e}"
                        raise OLMoConfigurationError(msg) from e
                return new_dict
            elif isinstance(d, (list, tuple, set)):
                return d.__class__(
                    (decode_data(x, f"{prefix}.{i}" if prefix else str(i)) for i, x in enumerate(d))
                )
            else:
                return d

        if overrides:
            for key, value in _clean_opts(overrides):
                _set_nested(data, key, value)

        decoded = decode_data(data, "")
        if isinstance(decoded, cls):
            return decoded
        else:
            return decode(cls, decoded)

    @classmethod
    def from_file(cls: Type[C], path: PathOrStr, overrides: Optional[List[str]] = None) -> C:
        path_str = str(path)
        if path_str.endswith((".yml", ".yaml")):
            return cls.from_yaml(path, overrides=overrides)
        elif path_str.endswith(".json"):
            return cls.from_json(path, overrides=overrides)
        else:
            raise OLMoConfigurationError(f"Unsupported config file type: {path}")

    @classmethod
    def from_json(cls: Type[C], path: PathOrStr, overrides: Optional[List[str]] = None) -> C:
        with cached_path(path).open() as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict, overrides=overrides)

    @classmethod
    def from_yaml(cls: Type[C], path: PathOrStr, overrides: Optional[List[str]] = None) -> C:
        with cached_path(path).open() as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict, overrides=overrides)


def _set_nested(data: Any, key: str, value: Any):
    if "." in key:
        key, child_keys = key.split(".", 1)
        if isinstance(data, dict):
            _set_nested(data[key], child_keys, value)
        elif isinstance(data, list):
            _set_nested(data[int(key)], child_keys, value)
        else:
            raise ValueError(data)
    else:
        if isinstance(data, dict):
            data[key] = value
        elif isinstance(data, list):
            data[int(key)] = value
        else:
            raise ValueError(f"Can't set value '{value}' at key '{key}' for object {data}")


def _clean_opts(opts: Sequence[str]) -> list[tuple[str, Any]]:
    return [_clean_opt(s) for s in opts]


def _clean_opt(arg: str) -> tuple[str, Any]:
    if "=" not in arg:
        name, val = arg, "true"
    else:
        name, val = arg.split("=", 1)
    name = name.strip(" -").replace("-", "_")
    if not val or val.isspace():
        val = ""
    else:
        val = yaml.safe_load(val)
    return (name, val)


class DType(StrEnum):
    """
    An enumeration of supported PyTorch data types.
    """

    float32 = "float32"
    bfloat16 = "bfloat16"
    float16 = "float16"
    float8_e4m3fn = "float8_e4m3fn"  # note: other e4m3 variants are supported in torch
    float8_e5m2 = "float8_e5m2"

    @classmethod
    def from_pt(cls, dtype: torch.dtype) -> "DType":
        if dtype == torch.float32:
            return DType.float32
        elif dtype == torch.bfloat16:
            return DType.bfloat16
        elif dtype == torch.float16:
            return DType.float16
        elif dtype == torch.float8_e4m3fn:
            return DType.float8_e4m3fn
        elif dtype == torch.float8_e5m2:
            return DType.float8_e5m2
        else:
            raise NotImplementedError(dtype)

    def as_pt(self) -> torch.dtype:
        return getattr(torch, self)
