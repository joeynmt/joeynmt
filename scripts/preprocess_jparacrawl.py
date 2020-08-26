# coding: utf-8
"""
Preprocess JParaCrawl
"""

import os
import argparse
import pandas as pd
import numpy as np
import unicodedata
from collections import OrderedDict


def prepare(data_dir, size, seed=None):
    dtype = OrderedDict({'source': str, 'probability': float, 'en': str, 'ja': str})
    df = pd.read_csv(os.path.join(data_dir, 'en-ja', 'en-ja.bicleaner05.txt'), header=None, names=dtype.keys(),
                     sep='\t', encoding='utf8', quoting=3, keep_default_na=False, na_values='', dtype=dtype)
    df = df.drop_duplicates(subset=['en', 'ja'])
    df = df[~df['en'].str.contains('�') & ~df['ja'].str.contains('�')]
    df = df[['en', 'ja']].applymap(lambda x: unicodedata.normalize('NFKC', x))
    df = df.dropna(how='any')

    if seed is not None:
        np.random.seed(seed)
    test_index = np.random.choice(df.index, size=size, replace=False)
    train_index = np.setdiff1d(df.index, test_index)
    for lang in ['en', 'ja']:
        for data_set, drop_index in zip(['train', 'dev'], [test_index, train_index]):
            df[lang].drop(index=drop_index, inplace=False).to_csv(os.path.join(data_dir, data_set+'.'+lang),
                          header=False, index=False, sep='\t', encoding='utf8', quoting=3)


def main():
    PATH = os.path.dirname(os.path.abspath('__file__'))

    ap = argparse.ArgumentParser("Preprocess JParaCrawl")
    ap.add_argument("--data_dir", type=str, default=os.path.join(PATH, "../test/data/jparacrawl"),
                    help="path to data dir. default: ../test/data/jparacrawl")
    ap.add_argument("--dev_size", type=int, default=5000, help="development set size")
    ap.add_argument("--seed", type=int, default=12345, help="random seed for train-dev-split")
    args = ap.parse_args()

    prepare(args.data_dir, args.dev_size, args.seed)


if __name__ == "__main__":
    main()

