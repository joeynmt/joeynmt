#!/usr/bin/env python

import argparse
from pathlib import Path

import numpy as np


def generate_samples(n=10000, low=0, high=10, maxlen=10):
    samples = []
    for _ in range(n):
        size = np.random.randint(1, maxlen + 1)
        sample = np.random.randint(low, high, size)
        samples.append(sample)
    return samples


def sample_to_str(sample):
    return " ".join(map(str, sample))


def save_samples(samples, filename, reverse=False):
    with open(filename, mode="w", encoding="utf-8") as f:
        for sample in samples:
            sample = sample[::-1] if reverse else sample
            f.write(sample_to_str(sample) + "\n")


def generate_task(args):
    # pylint: disable=redefined-outer-name
    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)

    # train
    samples = generate_samples(
        n=args.train_size, low=args.low, high=args.high, maxlen=args.maxlen
    )
    filename = out_dir / "train"
    save_samples(samples, filename.with_suffix('.src'), reverse=False)
    save_samples(samples, filename.with_suffix('.trg'), reverse=True)

    # dev
    samples = generate_samples(
        n=args.dev_size, low=args.low, high=args.high, maxlen=args.maxlen + 5
    )
    filename = out_dir / "dev"
    save_samples(samples, filename.with_suffix('.src'), reverse=False)
    save_samples(samples, filename.with_suffix('.trg'), reverse=True)

    # test
    samples = generate_samples(
        n=args.test_size, low=args.low, high=args.high, maxlen=args.maxlen + 5
    )
    filename = out_dir / "test"
    save_samples(samples, filename.with_suffix('.src'), reverse=False)
    save_samples(samples, filename.with_suffix('.trg'), reverse=True)


if __name__ == "__main__":
    root_path = Path(__file__).parent.resolve()

    ap = argparse.ArgumentParser("Generate data for the reverse task")
    ap.add_argument(
        "--out_dir",
        type=str,
        default=(root_path / "../test/data/reverse").as_posix(),
        help="path to output dir. default: ../test/data/reverse",
    )
    ap.add_argument("--train_size", type=int, default=50000, help="train set size")
    ap.add_argument("--dev_size", type=int, default=1000, help="dev set size")
    ap.add_argument("--test_size", type=int, default=1000, help="test set size")
    ap.add_argument("--low", type=int, default=0, help="min value")
    ap.add_argument("--high", type=int, default=50, help="max value")
    ap.add_argument("--maxlen", type=int, default=25, help="max sequence length")
    ap.add_argument("--seed", type=int, default=42, help="random seed")
    args = ap.parse_args()

    np.random.seed(args.seed)
    generate_task(args)
