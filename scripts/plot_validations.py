#!/usr/bin/env python
# coding: utf-8
import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

matplotlib.use("Agg")


def read_vfiles(vfiles: List[Path]) -> Dict:
    """
    Parse validation report files
    :param vfiles: list of files
    :return:
    """
    models = {}
    for vfile in vfiles:
        assert vfile.is_file(), f"{vfile} not found."
        model_name = vfile.parent.stem

        steps = {}
        for line in vfile.read_text(encoding="utf-8").splitlines():
            entries = line.strip().split()
            key = int(entries[1])
            steps[key] = {}
            for i in range(2, len(entries) - 1, 2):
                name = entries[i].strip(":")
                value = float(entries[i + 1])
                steps[key][name] = value
        models[model_name] = steps
    return models


def plot_models(models: Dict, plot_values: List, output_path: str) -> None:
    """
    Plot the learning curves for several models
    :param models:
    :param plot_values:
    :param output_path:
    :return:
    """
    # models is a dict: name -> ckpt values
    f, axes = plt.subplots(
        len(plot_values),
        len(models),
        sharex="col",
        sharey="row",
        figsize=(3 * len(models), 3 * len(plot_values)),
    )
    axes = np.array(axes).reshape((len(plot_values), len(models)))

    for col, model_name in enumerate(models):
        values = {}
        # get arrays for plotting
        for step in sorted(models[model_name]):
            logged_values = models[model_name][step]
            for plot_value in plot_values:
                if plot_value not in logged_values:  # pylint: disable=no-else-continue
                    continue
                elif plot_value not in values:
                    values[plot_value] = [[], []]
                values[plot_value][1].append(logged_values[plot_value])
                values[plot_value][0].append(step)

        for row, plot_value in enumerate(plot_values):
            axes[row][col].plot(values[plot_value][0], values[plot_value][1])
            axes[row][0].set_ylabel(plot_value)
            axes[0][col].set_title(model_name)
        axes[-1][col].set_xlabel("steps")

    plt.tight_layout()
    if output_path.endswith(".pdf"):
        pp = PdfPages(output_path)
        pp.savefig(f)
        pp.close()
    else:
        if not output_path.endswith(".png"):
            output_path += ".png"
        plt.savefig(output_path)

    plt.close()


def main(args):  # pylint: disable=redefined-outer-name
    models = read_vfiles([Path(m) / "validations.txt" for m in args.model_dirs])
    plot_models(models, args.plot_values, args.output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("JoeyNMT Validation plotting.")
    parser.add_argument("model_dirs", type=str, nargs="+", help="Model directories.")
    parser.add_argument(
        "--plot_values",
        type=str,
        nargs="+",
        default=["bleu"],
        help="Value(s) to plot. Default: bleu",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="plot.pdf",
        help="Plot will be stored in this location.",
    )
    args = parser.parse_args()

    main(args)
