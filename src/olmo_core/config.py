from dataclasses import asdict
from typing import Any, Dict


class Config:
    def as_dict(self, exclude_none: bool = False) -> Dict[str, Any]:
        """
        Convert into a regular Python dictionary.
        """
        # TODO: make this work for nested configs.
        out = asdict(self)  # type: ignore
        if exclude_none:
            for k in list(out.keys()):
                v = out[k]
                if v is None:
                    del out[k]
        return out
