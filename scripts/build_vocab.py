#!/usr/bin/env python3

import argparse
from collections import OrderedDict
import numpy as np


def build_vocab(train_paths, output_path):
    """
    Builds the vocabulary.
    Compatible with Nematus build_dict function, but does not
    output frequencies and special symbols.
    :param train_paths:
    :param output_path:
    :return:
    """

    counter = OrderedDict()

    # iterate over input paths
    for path in train_paths:
        with open(path, encoding="utf-8", mode="r") as f:
            for line in f:
                for token in line.strip('\r\n ').split(' '):
                    if token:
                        if token not in counter:
                            counter[token] = 0
                        counter[token] += 1

    words = list(counter.keys())
    freqs = list(counter.values())

    sorted_idx = np.argsort(freqs)
    sorted_words = [words[ii] for ii in sorted_idx[::-1]]

    with open(output_path, mode='w', encoding='utf-8') as f:
        for word in sorted_words:
            f.write(word + "\n")


if __name__ == "__main__":

    ap = argparse.ArgumentParser(
        description="Builds a vocabulary from training file(s)."
                    ""
                    "Can be used to build a joint vocabulary for weight tying."
                    "To do so, first apply BPE to both source and target "
                    "training files, and then build a vocabulary using"
                    "this script from their concatenation."
                    ""
                    "If you provide multiple files then this program "
                    "will merge them before building a joint vocabulary."
                    "")

    ap.add_argument("train_paths", type=str,
                    help="One or more input (training) file(s)", nargs="+")
    ap.add_argument("--output_path", type=str,
                    help="Output path for the built vocabulary",
                    default="vocab.txt")
    args = ap.parse_args()

    build_vocab(args.train_paths, args.output_path)
