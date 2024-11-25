from typing import Optional

from beaker import Beaker

_BEAKER_USERNAME: Optional[str] = None


def get_beaker_username() -> str:
    global _BEAKER_USERNAME

    if _BEAKER_USERNAME is None:
        _BEAKER_USERNAME = Beaker.from_env().account.whoami().name

    return _BEAKER_USERNAME
