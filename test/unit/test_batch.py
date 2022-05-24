from test.unit.test_helpers import TensorTestCase

import torch
from torch.utils.data import BatchSampler, DataLoader, RandomSampler, SequentialSampler

from joeynmt.batch import Batch
from joeynmt.data import TokenBatchSampler, load_data, make_data_iter


class TestBatch(TensorTestCase):

    def setUp(self):
        # minimal data config
        data_cfg = {
            "train": "test/data/toy/train",
            "dev": "test/data/toy/dev",
            "src": {
                "lang": "de",
                "level": "char",
                "lowercase": True,
                "max_length": 20,
            },
            "trg": {
                "lang": "en",
                "level": "char",
                "lowercase": True,
                "max_length": 20,
            },
            "dataset_type": "plain",
        }

        # load the data
        (
            self.src_vocab,
            self.trg_vocab,
            self.train_data,
            self.dev_data,
            _,
        ) = load_data(data_cfg, datasets=["train", "dev"])
        self.pad_index = self.trg_vocab.pad_index
        # random seeds
        self.seed = 42

    def testBatchTrainIterator(self):

        batch_size = 4
        # load  all sents, filtering happens during batch construction
        self.assertEqual(len(self.train_data), 1000)

        # make data iterator
        train_iter = make_data_iter(
            dataset=self.train_data,
            batch_size=batch_size,
            batch_type="sentence",
            shuffle=True,
            seed=self.seed,
            pad_index=self.pad_index,
            device=torch.device("cpu"),
        )
        self.assertTrue(isinstance(train_iter, DataLoader))
        self.assertEqual(train_iter.batch_sampler.batch_size, batch_size)
        self.assertTrue(isinstance(train_iter.batch_sampler, BatchSampler))
        self.assertTrue(isinstance(train_iter.batch_sampler.sampler,
                                   RandomSampler))  # shuffle=True
        initial_seed = train_iter.batch_sampler.sampler.generator.initial_seed()
        self.assertEqual(initial_seed, self.seed)

        expected_src0 = torch.LongTensor(
            [[27, 7, 5, 14, 5, 4, 27, 5, 9, 30, 6, 12, 5, 9, 15, 6, 17, 5, 6, 24, 3],
             [19, 25, 11, 37, 24, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
             [28, 23, 5, 7, 23, 5, 6, 4, 12, 11, 6, 25, 5, 4, 8, 16, 13, 31, 6, 24, 3],
             [12, 11, 8, 4, 7, 8, 10, 4, 28, 11, 8, 8, 7, 5, 9, 10, 24, 3, 1, 1, 1]])
        expected_src0_len = torch.LongTensor([21, 6, 21, 18])
        expected_trg0 = torch.LongTensor(
            [[7, 4, 14, 8, 6, 4, 8, 23, 4, 17, 13, 7, 10, 21, 5, 24, 3],
             [8, 28, 7, 18, 24, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
             [6, 13, 7, 10, 28, 4, 18, 8, 16, 24, 3, 1, 1, 1, 1, 1, 1],
             [9, 6, 4, 15, 9, 15, 24, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
        expected_trg0_len = torch.LongTensor([17, 6, 11, 8])

        total_samples = 0
        for b in train_iter:
            self.assertTrue(isinstance(b, Batch))
            if total_samples == 0:
                self.assertTensorEqual(b.src, expected_src0)
                self.assertTensorEqual(b.src_length, expected_src0_len)
                self.assertTensorEqual(b.trg, expected_trg0)
                self.assertTensorEqual(b.trg_length, expected_trg0_len)
            total_samples += b.nseqs
            self.assertLessEqual(b.nseqs, batch_size)
        self.assertEqual(total_samples, 27)

    def testTokenBatchTrainIterator(self):

        batch_size = 50  # num of tokens in one batch
        # load all sents here, filtering happends during batch construction
        self.assertEqual(len(self.train_data), 1000)

        # make data iterator
        train_iter = make_data_iter(
            dataset=self.train_data,
            batch_size=batch_size,
            batch_type="token",
            shuffle=True,
            seed=self.seed,
            pad_index=self.pad_index,
            device=torch.device("cpu"),
        )
        self.assertTrue(isinstance(train_iter, DataLoader))
        self.assertEqual(train_iter.batch_sampler.batch_size, batch_size)
        self.assertTrue(isinstance(train_iter.batch_sampler, TokenBatchSampler))
        self.assertTrue(isinstance(train_iter.batch_sampler.sampler,
                                   RandomSampler))  # shuffle=True
        initial_seed = train_iter.batch_sampler.sampler.generator.initial_seed()
        self.assertEqual(initial_seed, self.seed)

        expected_src0 = torch.LongTensor(
            [[27, 7, 5, 14, 5, 4, 27, 5, 9, 30, 6, 12, 5, 9, 15, 6, 17, 5, 6, 24, 3],
             [19, 25, 11, 37, 24, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
             [28, 23, 5, 7, 23, 5, 6, 4, 12, 11, 6, 25, 5, 4, 8, 16, 13, 31, 6, 24, 3]])
        expected_src0_len = torch.LongTensor([21, 6, 21])
        expected_trg0 = torch.LongTensor(
            [[7, 4, 14, 8, 6, 4, 8, 23, 4, 17, 13, 7, 10, 21, 5, 24, 3],
             [8, 28, 7, 18, 24, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
             [6, 13, 7, 10, 28, 4, 18, 8, 16, 24, 3, 1, 1, 1, 1, 1, 1]])
        expected_trg0_len = torch.LongTensor([17, 6, 11])

        total_tokens = 0
        for b in train_iter:
            self.assertTrue(isinstance(b, Batch))
            if total_tokens == 0:
                self.assertTensorEqual(b.src, expected_src0)
                self.assertTensorEqual(b.src_length, expected_src0_len)
                self.assertTensorEqual(b.trg, expected_trg0)
                self.assertTensorEqual(b.trg_length, expected_trg0_len)
            total_tokens += b.ntokens
        self.assertEqual(total_tokens, 387)

    def testBatchDevIterator(self):

        batch_size = 3
        self.assertEqual(len(self.dev_data), 20)

        # make data iterator
        dev_iter = make_data_iter(
            dataset=self.dev_data,
            batch_size=batch_size,
            batch_type="sentence",
            shuffle=False,
            pad_index=self.pad_index,
            device=torch.device("cpu"),
        )
        self.assertTrue(isinstance(dev_iter, DataLoader))
        self.assertEqual(dev_iter.batch_sampler.batch_size, batch_size)
        self.assertTrue(isinstance(dev_iter.batch_sampler, BatchSampler))
        self.assertTrue(isinstance(dev_iter.batch_sampler.sampler,
                                   SequentialSampler))  # shuffle=False

        expected_src0 = torch.LongTensor([[
            32, 11, 4, 22, 4, 11, 14, 8, 19, 4, 22, 4, 21, 11, 8, 4, 8, 19, 14, 14, 4,
            20, 7, 19, 13, 11, 16, 25, 7, 6, 17, 4, 8, 5, 7, 6, 4, 38, 3
        ],
                                          [
                                              7, 16, 13, 4, 23, 9, 5, 15, 5, 4, 18, 7,
                                              16, 13, 4, 22, 4, 12, 11, 8, 8, 4, 7, 16,
                                              13, 4, 12, 11, 4, 20, 7, 6, 4, 24, 3, 1,
                                              1, 1, 1
                                          ],
                                          [
                                              32, 11, 4, 22, 4, 17, 15, 10, 5, 6, 4, 10,
                                              11, 17, 4, 24, 3, 1, 1, 1, 1, 1, 1, 1, 1,
                                              1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
                                          ]])
        expected_src0_len = torch.LongTensor([39, 35, 17])
        expected_trg0 = torch.LongTensor([[
            18, 5, 11, 4, 26, 4, 11, 8, 4, 26, 4, 19, 13, 7, 6, 4, 9, 11, 4, 25, 9, 8,
            13, 7, 17, 28, 9, 10, 21, 4, 34, 3
        ],
                                          [
                                              9, 0, 20, 4, 13, 7, 22, 22, 18, 4, 6, 8,
                                              4, 25, 5, 4, 13, 5, 12, 5, 4, 24, 3, 1, 1,
                                              1, 1, 1, 1, 1, 1, 1
                                          ],
                                          [
                                              18, 5, 11, 4, 26, 4, 13, 5, 14, 14, 8, 4,
                                              24, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                              1, 1, 1, 1, 1, 1
                                          ]])
        expected_trg0_len = torch.LongTensor([32, 23, 14])

        total_samples = 0
        for b in dev_iter:
            self.assertTrue(isinstance(b, Batch))

            # test the sorting by src length
            before_sort = b.src_length
            b.sort_by_src_length()
            after_sort = b.src_length
            self.assertTensorEqual(
                torch.sort(before_sort, descending=True)[0], after_sort)
            if total_samples == 0:
                self.assertTensorEqual(b.src, expected_src0)
                self.assertTensorEqual(b.src_length, expected_src0_len)
                self.assertTensorEqual(b.trg, expected_trg0)
                self.assertTensorEqual(b.trg_length, expected_trg0_len)
            total_samples += b.nseqs
            self.assertLessEqual(b.nseqs, batch_size)
        self.assertEqual(total_samples, len(self.dev_data))
