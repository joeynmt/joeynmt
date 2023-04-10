import unittest

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from joeynmt.data import load_data
from joeynmt.datasets import (
    RandomSubsetSampler,
    SentenceBatchSampler,
    TokenBatchSampler,
)


class TestDataSampler(unittest.TestCase):

    def setUp(self):

        # minimal data config
        data_cfg = {
            "train": "test/data/toy/train",
            "dev": "test/data/toy/dev",
            "test": "test/data/toy/test",
            "src": {
                "lang": "de",
                "level": "word",
                "lowercase": False,
                "max_length": 10,
            },
            "trg": {
                "lang": "en",
                "level": "word",
                "lowercase": False,
                "max_length": 10,
            },
            "sample_train_subset": -1,
            "sample_dev_subset": 10,
            "dataset_type": "plain",
        }

        # load toy data
        _, self.trg_vocab, self.train_data, self.dev_data, self.test_data = load_data(
            data_cfg, datasets=["train", "dev", "test"])

        # random seed
        self.seed = 42

    def testSentenceBatchSampler(self):
        batch_size = 10  # 10 sentences

        # load all sents here, filtering happends in iterator
        self.assertEqual(len(self.train_data), 1000)

        # make batches by number of sentences
        train_loader = self.train_data.make_iter(
            batch_size=batch_size,
            batch_type="sentence",
            shuffle=True,
            seed=self.seed,
            eos_index=self.trg_vocab.eos_index,
            pad_index=self.trg_vocab.pad_index,
            device=torch.device("cpu"),
            num_workers=0,
        )
        self.assertTrue(isinstance(train_loader, DataLoader))
        self.assertTrue(isinstance(train_loader.batch_sampler, SentenceBatchSampler))
        self.assertTrue(isinstance(train_loader.batch_sampler.sampler,
                                   RandomSampler))  # shuffle=True
        initial_seed = train_loader.batch_sampler.sampler.generator.initial_seed()
        self.assertEqual(initial_seed, self.seed)
        self.assertEqual(len(train_loader), 100)  # num_samples // batch_size

        # reset seed (set epoch)
        train_iter = iter(train_loader.batch_sampler.sampler)
        train_loader.batch_sampler.set_epoch(0)
        batch_idx = next(train_loader.batch_sampler.sampler.__iter__())
        self.assertEqual(batch_idx, 542)

        _ = next(train_loader.batch_sampler.sampler.__iter__())
        _ = next(train_loader.batch_sampler.sampler.__iter__())
        train_loader.batch_sampler.set_epoch(0)
        batch_idx = next(train_loader.batch_sampler.sampler.__iter__())
        self.assertEqual(batch_idx, 542)

        # data stats
        train_loader.batch_sampler.set_epoch(10)
        for b in train_loader:
            pass
        d = train_loader.batch_sampler.sampler.data_source.stats
        self.assertEqual(d.epoch, 10)
        self.assertEqual(d.num_valid, 382)
        self.assertEqual(d.num_invalid, 618)
        self.assertEqual(d.max_length, 20)
        self.assertEqual(d.min_length, 2)

        # subsampling
        train_loader.batch_sampler.sampler.data_source.random_subset = 500
        self.assertEqual(len(train_loader), 50)  # num_samples // batch_size

        train_loader.batch_sampler.sampler.data_source.random_subset = 2000
        with self.assertRaises(AssertionError) as e:
            train_loader.batch_sampler.num_samples
            self.assertEqual(
                "Can only subsample from train or dev set larger than 2000.",
                str(e.exception))

        # make batches by number of sentences
        dev_loader = self.dev_data.make_iter(
            batch_size=batch_size,
            batch_type="sentence",
            shuffle=False,
            seed=self.seed,
            eos_index=self.trg_vocab.eos_index,
            pad_index=self.trg_vocab.pad_index,
            device=torch.device("cpu"),
            num_workers=0,
        )
        self.assertTrue(isinstance(dev_loader, DataLoader))
        self.assertTrue(isinstance(dev_loader.batch_sampler, SentenceBatchSampler))
        self.assertTrue(isinstance(dev_loader.batch_sampler.sampler,
                                   RandomSubsetSampler))  # shuffle=False

        # reset seed (set epoch)
        self.assertEqual(dev_loader.batch_sampler.sampler.data_source.indices,
                         [0, 1, 2, 4, 6, 10, 11, 14, 15, 18])

        dev_loader.batch_sampler.set_epoch(10)
        self.assertEqual(dev_loader.batch_sampler.sampler.data_source.indices,
                         [1, 2, 6, 8, 9, 11, 12, 13, 16, 17])

        dev_loader.batch_sampler.set_epoch(20)
        self.assertEqual(dev_loader.batch_sampler.sampler.data_source.indices,
                         [1, 2, 3, 5, 7, 9, 13, 14, 18, 19])

        # data stats
        dev_loader.batch_sampler.set_epoch(10)
        for b in dev_loader:
            pass
        d = dev_loader.batch_sampler.sampler.data_source.stats
        self.assertEqual(d.epoch, 10)
        self.assertEqual(d.num_valid, 10)
        self.assertEqual(d.num_invalid, 0)
        self.assertEqual(d.max_length, 92)
        self.assertEqual(d.min_length, 9)

    def testTokenBatchSampler(self):
        batch_size = 50  # 50 tokens

        self.assertEqual(len(self.test_data), 20)

        # make batches by number of tokens
        test_loader = self.test_data.make_iter(
            batch_size=batch_size,
            batch_type="token",
            shuffle=False,
            seed=self.seed,
            eos_index=self.trg_vocab.eos_index,
            pad_index=self.trg_vocab.pad_index,
            device=torch.device("cpu"),
            num_workers=0,
        )
        self.assertTrue(isinstance(test_loader, DataLoader))
        self.assertEqual(test_loader.batch_sampler.batch_size, batch_size)
        self.assertTrue(isinstance(test_loader.batch_sampler, TokenBatchSampler))
        self.assertTrue(isinstance(test_loader.batch_sampler.sampler,
                                   SequentialSampler))  # shuffle=False

        with self.assertRaises(NotImplementedError) as e:
            len(test_loader)
            self.assertEqual("NotImplementedError", str(e.exception))

        # data stats
        test_loader.batch_sampler.set_epoch(10)
        for b in test_loader:
            pass
        d = test_loader.batch_sampler.sampler.data_source.stats
        self.assertEqual(d.epoch, 10)
        self.assertEqual(d.num_valid, 20)
        self.assertEqual(d.num_invalid, 0)
        self.assertEqual(d.max_length, 20)
        self.assertEqual(d.min_length, 2)

        # subsampling
        test_loader.batch_sampler.sampler.data_source.random_subset = 10
        with self.assertRaises(AssertionError) as e:
            test_loader.batch_sampler.num_samples
            self.assertEqual(
                "Can only subsample from train or dev set larger than 10.",
                str(e.exception))
