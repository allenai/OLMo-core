# /// script
# requires-python = ">=3.11,<3.12"
# dependencies = [
#     "matplotlib",
#     "numpy"
# ]
# ///

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ai2_theme import get_shade, apply_ai2_theme

species = ("Adelie", "Chinstrap", "Gentoo")
penguin_means = {
    "Bill Depth": (18.35, 18.43, 14.98),
    "Bill Length": (38.79, 48.83, 47.50),
    "Flipper Depth": (30.39, 32.11, 28.92),
    "Flipper Length": (189.95, 195.82, 217.19),
}

x = np.arange(len(species))  # the label locations

apply_ai2_theme()


all_shades = [
    ("teal", True),
    ("teal", False),
    ("pink", True),
    ("pink", False),
    ("green", True),
    ("green", False),
    ("purple", True),
    ("purple", False),
]
c = 2
w = 6
fig, subplots = plt.subplots(s := len(all_shades) // c, c, figsize=(w * s, w * c), layout="constrained")


for ax, (shade_name, direction) in zip(subplots.flatten(), all_shades):
    width = 0.2  # the width of the bars
    multiplier = 0
    colors = get_shade(shade_name, len(penguin_means), direction)
    for (attribute, measurement), color in zip(penguin_means.items(), colors):
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width=width, color=color, label=attribute)
        ax.bar_label(rects, padding=4)
        multiplier += 1

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel("Length (mm)")
    ax.set_title("Penguin attributes")
    ax.set_xticks(x + width, species)
    ax.legend(loc="upper left", ncols=4, title="Attributes")
    ax.set_ylim(0, 400)

# Path("tmp").mkdir(parents=True, exist_ok=True)
# fig.savefig("tmp/ai2_theme_example.png", dpi=150, bbox_inches="tight")
# plt.close(fig)

plt.show()
