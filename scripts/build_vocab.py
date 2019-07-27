#!/usr/bin/env python3

import argparse
from collections import Counter


def build_vocab(train_paths, output_path):
    """
    Builds the vocabulary.
    Compatible with subword-nmt's get_vocab function, but does not
    output frequencies.
    :param train_paths:
    :param output_path:
    :return:
    """

    counter = Counter()

    with open(output_path, encoding="utf-8", mode="w") as f_out:

        # iterate over input paths
        for path in train_paths:
            with open(path, encoding="utf-8", mode="r") as f:
                for line in f:
                    for token in line.strip('\r\n ').split(' '):
                        if token:
                            counter[token] += 1

        # write the vocabulary to file
        for key, f in sorted(counter.items(), key=lambda x: x[1], reverse=True):
            f_out.write(key + "\n")


if __name__ == "__main__":

    ap = argparse.ArgumentParser(
        description="Builds a vocabulary from training file(s)."
                    ""
                    "Can be used to build a joint vocabulary for weight tying."
                    "To do so, first apply BPE to both source and target "
                    "training files, and then build a vocabulary using"
                    "this script from their concatenation.")

    ap.add_argument("train_paths", type=str,
                    help="One or more input (training) file(s)", nargs="+")
    ap.add_argument("--output_path", type=str,
                    help="Output path for the built vocabulary",
                    default="vocab.txt")
    args = ap.parse_args()

    build_vocab(args.train_paths, args.output_path)
