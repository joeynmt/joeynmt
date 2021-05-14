#!/usr/bin/env python

import numpy as np
import os

np.random.seed(42)


def generate_samples(n=10000, low=0, high=10, maxlen=10):
    samples = []
    for i in range(n):
        size = np.random.randint(1, maxlen + 1)
        sample = np.random.randint(low, high, size)
        sample = [chr(ord('a') + x) for x in sample]
        samples.append(sample)
    return samples


def sample_to_str(sample):
    return " ".join(map(str, sample))


def save_samples(samples,
                 output_dir="copy_task",
                 prefix="train", ext="src", reverse=False):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(os.path.join(output_dir, prefix + "." + ext), mode="w", encoding="utf-8") as f:
        for sample in samples:
            sample = sample[::-1] if reverse else sample
            f.write(sample_to_str(sample) + "\n")


def generate_task(train="train", dev="dev", test="test", src="src", trg="trg"):

    # train
    samples = generate_samples(10000, high=26, maxlen=20)
    save_samples(samples, prefix=train, ext=src, reverse=False)
    save_samples(samples, prefix=train, ext=trg, reverse=False)

    # dev
    samples = generate_samples(500, high=26,  maxlen=20)
    save_samples(samples, prefix=dev, ext=src, reverse=False)
    save_samples(samples, prefix=dev, ext=trg, reverse=False)

    # test
    samples = generate_samples(500, high=26, maxlen=20)
    save_samples(samples, prefix=test, ext=src, reverse=False)
    save_samples(samples, prefix=test, ext=trg, reverse=False)


if __name__ == "__main__":
    generate_task()
