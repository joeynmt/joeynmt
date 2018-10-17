#!/usr/bin/env python

import logging
import matplotlib
matplotlib.use('Agg')

import numpy as np
from matplotlib import rcParams
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# font config
rcParams['font.family'] = "sans-serif"
rcParams['font.sans-serif'] = ["Fira Sans"]
rcParams['font.weight'] = "regular"


def plot_heatmap(scores=None, column_labels=None, row_labels=None,
                 output_path="plot.png"):

    """
    Plotting function that can be used to visualize (self-)attention

    :param scores: attention scores
    :param column_labels:  labels for columns (e.g. target tokens)
    :param row_labels: labels for rows (e.g. source tokens)
    :param output_path: path to save to
    :return:
    """

    assert output_path.endswith(".png") or output_path.endswith(".pdf"), \
        "output path must have .png or .pdf extension"

    x_sent_len = len(column_labels)
    y_sent_len = len(row_labels)
    scores = scores[:y_sent_len, :x_sent_len]
    # check that cut off part didn't have any attention
    assert np.sum(scores[y_sent_len:, :x_sent_len]) == 0

    # automatic label size
    labelsize = 25 * (10 / max(x_sent_len, y_sent_len))

    matplotlib.rcParams['xtick.labelsize'] = labelsize
    matplotlib.rcParams['ytick.labelsize'] = labelsize

    fig, ax = plt.subplots(figsize=(10, 10), dpi=300)
    heatmap = plt.imshow(scores, cmap='viridis', aspect='equal',
                         origin='upper', vmin=0., vmax=1.)

    ax.set_xticklabels(column_labels, minor=False, rotation="vertical")
    ax.set_yticklabels(row_labels, minor=False)

    ax.xaxis.tick_top()
    ax.set_xticks(np.arange(scores.shape[1]) + 0, minor=False)
    ax.set_yticks(np.arange(scores.shape[0]) + 0, minor=False)
    plt.tight_layout()

    if output_path.endswith(".pdf"):
        pp = PdfPages(output_path)
        pp.savefig(fig)
        pp.close()
    else:
        if not output_path.endswith(".png"):
            output_path = output_path + ".png"
        plt.savefig(output_path)

    plt.close()
