#!/usr/bin/env python

from typing import List, Optional
import numpy as np

# pylint: disable=wrong-import-position
import matplotlib
matplotlib.use('Agg')

from matplotlib import rcParams
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def plot_heatmap(scores: np.array, column_labels: List[str],
                 row_labels: List[str], output_path: Optional[str] = None,
                 dpi: int = 300) -> Figure:

    """
    Plotting function that can be used to visualize (self-)attention.
    Plots are saved if `output_path` is specified, in format that this file
    ends with ('pdf' or 'png').

    :param scores: attention scores
    :param column_labels:  labels for columns (e.g. target tokens)
    :param row_labels: labels for rows (e.g. source tokens)
    :param output_path: path to save to
    :param dpi: set resolution for matplotlib
    :return: pyplot figure
    """

    if output_path is not None:
        assert output_path.endswith(".png") or output_path.endswith(".pdf"), \
        "output path must have .png or .pdf extension"

    x_sent_len = len(column_labels)
    y_sent_len = len(row_labels)
    scores = scores[:y_sent_len, :x_sent_len]
    # check that cut off part didn't have any attention
    assert np.sum(scores[y_sent_len:, :x_sent_len]) == 0

    # automatic label size
    labelsize = 25 * (10 / max(x_sent_len, y_sent_len))

    # font config
    rcParams['xtick.labelsize'] = labelsize
    rcParams['ytick.labelsize'] = labelsize
    #rcParams['font.family'] = "sans-serif"
    #rcParams['font.sans-serif'] = ["Fira Sans"]
    #rcParams['font.weight'] = "regular"

    fig, ax = plt.subplots(figsize=(10, 10), dpi=dpi)
    plt.imshow(scores, cmap='viridis', aspect='equal',
               origin='upper', vmin=0., vmax=1.)
    ax.xaxis.tick_top()
    ax.set_xticks(np.arange(scores.shape[1]) + 0, minor=False)
    ax.set_yticks(np.arange(scores.shape[0]) + 0, minor=False)

    ax.set_xticklabels(column_labels, minor=False, rotation="vertical")
    ax.set_yticklabels(row_labels, minor=False)

    plt.tight_layout()

    if output_path is not None:
        if output_path.endswith(".pdf"):
            pp = PdfPages(output_path)
            pp.savefig(fig)
            pp.close()
        else:
            if not output_path.endswith(".png"):
                output_path += ".png"
            plt.savefig(output_path)

    plt.close()

    return fig
