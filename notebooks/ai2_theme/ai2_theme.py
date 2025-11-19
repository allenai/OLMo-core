"""Matplotlib theming helpers for the Ai2 visual identity."""

import dataclasses as dt
import re
from pathlib import Path
from typing import ClassVar, TypeVar

import matplotlib as mpl
from matplotlib.font_manager import FontProperties, fontManager

MANROPE_BASE_PATH = (Path(__file__).parent / "manrope").absolute()
for path in MANROPE_BASE_PATH.iterdir():
    if path.suffix == ".ttf":
        fontManager.addfont(str(path))


AI2_COLORS = {
    "pink": "#f0529c",
    "teal": "#105257",
    "purple": "#b11be8",
    "green": "#0fcb8c",
    "lime": "#bef576",
    "sky": "#12cce5",
    "orange": "#f65834",
    "yellow": "#fff500",
    "error": "#fd4645",
    "warning": "#ffa31c",
    "confirmation": "#549c35",
    "information": "#2a88ef",
    "off_white": "#faf2e9",
    "dark_teal": "#0a3235",
}


T = TypeVar("T", bound="BaseShade")


@dt.dataclass
class BaseShade:
    name: str = ""
    color: str = ""
    lighter: list[str] = dt.field(default_factory=list)
    darker: list[str] = dt.field(default_factory=list)

    _registry: ClassVar[dict[str, type["BaseShade"]]] = {}

    def __post_init__(self):
        cls_name = type(self).__name__

        if not self.name:
            raise ValueError(f"{cls_name} must have a name assigned")

        if not self.color:
            raise ValueError(f"{cls_name} must have a color assigned")

    def __len__(self):
        return len(self.lighter) + len(self.darker) + 1

    @staticmethod
    def shade(*colors: str) -> list[str]:
        return dt.field(default_factory=lambda: list(colors))

    @staticmethod
    def validate_color(color: str) -> bool:
        if not re.match(r"^#[0-9a-fA-F]{6}$", color):
            return False
        return True

    @staticmethod
    def sort_colors(colors: list[str]) -> list[str]:
        return sorted(colors, key=lambda x: -int(x.lstrip("#"), 16))

    @classmethod
    def register_shade(cls, shade: type[T]) -> type[T]:
        cls._registry[shade.name] = shade
        return shade

    @classmethod
    def get_shade(cls, name: str) -> "BaseShade":
        if name not in cls._registry:
            raise ValueError(f"Shade '{name}' not found")
        return cls._registry[name]()

    def get_colors(self, n: int | None = None, prefer_light: bool = True) -> list[str]:
        n = len(self) if n is None else n
        assert n > 0, "Cannot get 0 shades"

        if (n - 1) <= len(self.lighter) and prefer_light:
            return self.sort_colors([self.color] + self.lighter[: n - 1])
        elif (n - 1) <= len(self.darker) and not prefer_light:
            return self.sort_colors([self.color] + self.darker[: n - 1])
        elif (n - 1) <= (len(self.lighter) + len(self.darker)):
            if prefer_light:
                n -= len(self.lighter) + 1
                return self.sort_colors([self.color] + self.lighter + self.darker[:n])
            else:
                n -= len(self.darker)
                return self.sort_colors([self.color] + self.darker + self.lighter[:n])

        raise ValueError(f"Cannot get {n} shades for {self.name}")


@dt.dataclass
@BaseShade.register_shade
class Ai2PinkShade(BaseShade):
    name: str = "pink"
    color: str = "#f0529c"
    lighter: list[str] = BaseShade.shade("#f586ba", "#f9bad7", "#fcdceb")
    darker: list[str] = BaseShade.shade("#a8396d", "#78294e", "#48192f")


@dt.dataclass
@BaseShade.register_shade
class Ai2GreenShade(BaseShade):
    name: str = "green"
    color: str = "#0ca270"
    lighter: list[str] = BaseShade.shade("#0fcb8c", "#6fe0ba", "#b7efdd")
    darker: list[str] = BaseShade.shade("#097a54", "#065138", "#03291c")


@dt.dataclass
@BaseShade.register_shade
class Ai2TealShade(BaseShade):
    name: str = "teal"
    color: str = "#105257"
    lighter: list[str] = BaseShade.shade("#88a9ab", "#b7cbcd", "#e7eeee")
    darker: list[str] = BaseShade.shade("#286368", "#0d4246", "#062123")


@dt.dataclass
@BaseShade.register_shade
class Ai2PurpleShade(BaseShade):
    name: str = "purple"
    color: str = "#b017e8"
    lighter: list[str] = BaseShade.shade("#c65aee", "#dd9cf5", "#f4defc")
    darker: list[str] = BaseShade.shade("#8912b4", "#620d81", "#3b084d")


@dt.dataclass
@BaseShade.register_shade
class Ai2GrayShade(BaseShade):
    name: str = "gray"
    color: str = "#969696"
    lighter: list[str] = BaseShade.shade("#ADADAD", "#C4C4C4", "#DBDBDB")
    darker: list[str] = BaseShade.shade("#808080", "#696969", "#525252")


@dt.dataclass
@BaseShade.register_shade
class Ai2RainbowShade(BaseShade):
    name: str = "rainbow"
    color: str = Ai2PinkShade.color
    lighter: list[str] = BaseShade.shade(Ai2GreenShade.color, Ai2PurpleShade.color)
    darker: list[str] = BaseShade.shade(Ai2TealShade.color, Ai2GrayShade.color)


AI2_FONT_WEIGHTS = {
    "light": 300,
    "regular": 400,
    "medium": 500,
    "bold": 700,
}


def get_shade(name: str, n: int | None = None, light: bool = True) -> list[str]:
    """Get a shade by name."""
    return BaseShade.get_shade(name).get_colors(n=n, prefer_light=light)


def apply_ai2_theme(shade: str = "rainbow") -> None:
    """Apply the AI2 matplotlib theme to the current session."""

    # Core palette and typography.
    mpl.rcParams.update(
        {
            "font.family": "Manrope",
            "text.color": AI2_COLORS["dark_teal"],
            "axes.labelweight": AI2_FONT_WEIGHTS["medium"],
            "axes.titleweight": AI2_FONT_WEIGHTS["bold"],
            "axes.labelcolor": AI2_COLORS["dark_teal"],
            "axes.titlesize": "x-large",
            "axes.titlelocation": "left",
            "axes.edgecolor": AI2_COLORS["teal"],
            "axes.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.labelsize": "large",
            "axes.prop_cycle": mpl.cycler(color=get_shade(shade)),
            "figure.titlesize": "x-large",
            "figure.titleweight": AI2_FONT_WEIGHTS["bold"],
            "figure.autolayout": False,
            "xtick.color": AI2_COLORS["teal"],
            "ytick.color": AI2_COLORS["teal"],
            "xtick.labelsize": "medium",
            "ytick.labelsize": "medium",
            "xtick.major.size": 0,
            "ytick.major.size": 0,
            "grid.color": AI2_COLORS["teal"],
            "grid.alpha": 0.15,
            "grid.linewidth": 0.8,
            "grid.linestyle": "-",
            "axes.grid": True,
            "axes.axisbelow": True,
            "lines.linewidth": 2.2,
            "lines.solid_capstyle": "round",
            "lines.dash_capstyle": "round",
            "lines.solid_joinstyle": "round",
            "patch.edgecolor": "none",
            "patch.force_edgecolor": False,
            "legend.frameon": True,
            "legend.facecolor": "#ffffff",
            "legend.edgecolor": AI2_COLORS["teal"],
            "legend.fancybox": True,
            "legend.fontsize": "large",
            "legend.title_fontsize": "large",
        }
    )


__all__ = ["apply_ai2_theme", "get_shade"]
