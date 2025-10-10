import argparse
import json
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import mplcursors
import pandas as pd
from cached_path import cached_path

from olmo_core.io import add_cached_path_clients


def run_coord_check(data_files: List[str], widths: List[str], x_axis_label: str, title: str):
    if len(data_files) != len(widths):
        raise ValueError(
            f"Number of data files {len(data_files)} does not match number of widths {len(widths)}"
        )

    data: List[Dict] = []
    for data_file, width in zip(data_files, widths):
        local_data_file = cached_path(data_file)
        coord_data: List[Tuple[str, tuple, float]] = json.loads(local_data_file.read_text())
        for entry in coord_data:
            data.append(
                {
                    "width": width,
                    "param": entry[0],
                    "shape": tuple(entry[1]),
                    "mean_magnitude": entry[2],
                }
            )

    df = pd.DataFrame(data)

    params = df["param"].unique()

    for param in params:
        param_df = df[df["param"] == param]
        mean_magnitude_data = param_df.groupby(["width"], sort=False)["mean_magnitude"].mean()
        plt.plot(mean_magnitude_data, label=param)

    plt.xlabel(x_axis_label)
    plt.ylabel("Activation coord mean magnitude")
    plt.yscale("log", base=2)
    plt.title(title)
    # plt.legend()

    # Function to update annotation text on hover
    def hover_annotation(sel):
        label = sel.artist.get_label()
        sel.annotation.set_text(f"{label}: {sel.target[1]:.2e}")

    # Enable hover functionality
    cursor1 = mplcursors.cursor(hover=True)
    cursor1.connect("add", hover_annotation)

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_files",
        nargs="+",
        type=str,
        required=True,
        help="Local or remote paths of mup coord data",
    )
    parser.add_argument(
        "--widths",
        nargs="+",
        type=str,
        required=True,
        help="Width of model corresponding to each data file",
    )
    parser.add_argument(
        "--x_axis_label",
        type=str,
        default="width",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="muP coord check",
    )
    args = parser.parse_args()

    add_cached_path_clients()

    run_coord_check(
        data_files=args.data_files,
        widths=args.widths,
        x_axis_label=args.x_axis_label,
        title=args.title,
    )
