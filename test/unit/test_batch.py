import torch
import random

from torchtext.data.batch import Batch as TorchTBatch

from joeynmt.batch import Batch
from joeynmt.data import load_data, make_data_iter
from joeynmt.constants import PAD_TOKEN
from .test_helpers import TensorTestCase


class TestData(TensorTestCase):

    def setUp(self):
        self.train_path = "test/data/toy/train"
        self.dev_path = "test/data/toy/dev"
        self.test_path = "test/data/toy/test"
        self.levels = ["char", "word"]  # bpe is equivalently processed to word
        self.max_sent_length = 20

        # minimal data config
        self.data_cfg = {"src": "de", "trg": "en", "train": self.train_path,
                         "dev": self.dev_path, "level": "char",
                         "lowercase": True,
                         "max_sent_length": self.max_sent_length}

        # load the data
        self.train_data, self.dev_data, self.test_data, src_vocab, trg_vocab = \
            load_data(self.data_cfg)
        self.pad_index = trg_vocab.stoi[PAD_TOKEN]
        # random seeds
        seed = 42
        torch.manual_seed(seed)
        random.seed(42)

    def testBatchTrainIterator(self):

        batch_size = 4
        self.assertEqual(len(self.train_data), 27)

        # make data iterator
        train_iter = make_data_iter(self.train_data, train=True, shuffle=True,
                                    batch_size=batch_size)
        self.assertEqual(train_iter.batch_size, batch_size)
        self.assertTrue(train_iter.shuffle)
        self.assertTrue(train_iter.train)
        self.assertEqual(train_iter.epoch, 0)
        self.assertEqual(train_iter.iterations, 0)

        expected_src0 = torch.Tensor(
            [[21, 10, 4, 16, 4, 5, 21, 4, 12, 33, 6, 14, 4, 12, 23, 6, 18, 4,
              6, 9, 3],
             [20, 28, 4, 10, 28, 4, 6, 5, 14, 8, 6, 15, 4, 5, 7, 17, 11, 27,
              6, 9, 3],
             [24, 8, 7, 5, 24, 10, 12, 14, 5, 18, 4, 7, 17, 11, 4, 11, 4, 6,
              25, 3, 1]]).long()
        expected_src0_len = torch.Tensor([21, 21, 20]).long()
        expected_trg0 = torch.Tensor(
            [[6,  4, 27,  5,  8,  4,  5, 31,  4, 26,  7,  6, 10, 20, 11,
               9,  3],
             [8,  7,  6, 10, 17,  4, 13,  5, 15,  9,  3,  1,  1,  1,  1,
              1,  1],
             [12,  5,  4, 25,  7,  6,  8,  4,  7,  6, 18, 18, 11, 10, 12,
              23,  3]]).long()
        expected_trg0_len = torch.Tensor([18, 12, 18]).long()

        total_samples = 0
        for b in iter(train_iter):
            b = Batch(torch_batch=b, pad_index=self.pad_index)
            if total_samples == 0:
                self.assertTensorEqual(b.src, expected_src0)
                self.assertTensorEqual(b.src_length, expected_src0_len)
                self.assertTensorEqual(b.trg, expected_trg0)
                self.assertTensorEqual(b.trg_length, expected_trg0_len)
            total_samples += b.nseqs
            self.assertLessEqual(b.nseqs, batch_size)
        self.assertEqual(total_samples, len(self.train_data))

    def testBatchDevIterator(self):

        batch_size = 3
        self.assertEqual(len(self.dev_data), 20)

        # make data iterator
        dev_iter = make_data_iter(self.dev_data, train=False, shuffle=False,
                                  batch_size=batch_size)
        self.assertEqual(dev_iter.batch_size, batch_size)
        self.assertFalse(dev_iter.shuffle)
        self.assertFalse(dev_iter.train)
        self.assertEqual(dev_iter.epoch, 0)
        self.assertEqual(dev_iter.iterations, 0)

        expected_src0 = torch.Tensor(
            [[29, 8, 5, 22, 5, 8, 16, 7, 19, 5, 22, 5, 24, 8, 7, 5, 7, 19,
              16, 16, 5, 31, 10, 19, 11, 8, 17, 15, 10, 6, 18, 5, 7, 4, 10, 6,
              5, 25, 3],
             [10, 17, 11, 5, 28, 12, 4, 23, 4, 5, 0, 10, 17, 11, 5, 22, 5, 14,
              8, 7, 7, 5, 10, 17, 11, 5, 14, 8, 5, 31, 10, 6, 5, 9, 3, 1,
              1, 1, 1],
             [29, 8, 5, 22, 5, 18, 23, 13, 4, 6, 5, 13, 8, 18, 5, 9, 3, 1,
              1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
              1, 1, 1]]).long()
        expected_src0_len = torch.Tensor([39, 35, 17]).long()
        expected_trg0 = torch.Tensor(
            [[13, 11, 12, 4, 22, 4, 12, 5, 4, 22, 4, 25, 7, 6, 8, 4, 14, 12,
              4, 24, 14, 5, 7, 6, 26, 17, 14, 10, 20, 4, 23, 3],
             [14, 0, 28, 4, 7, 6, 18, 18, 13, 4, 8, 5, 4, 24, 11, 4, 7, 11,
              16, 11, 4, 9, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1],
             [13, 11, 12, 4, 22, 4, 7, 11, 27, 27, 5, 4, 9, 3, 1, 1, 1, 1,
              1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]).long()
        expected_trg0_len = torch.Tensor([33, 24, 15]).long()

        total_samples = 0
        for b in iter(dev_iter):
            self.assertEqual(type(b), TorchTBatch)
            b = Batch(b, pad_index=self.pad_index)

            # test the sorting by src length
            self.assertEqual(type(b), Batch)
            before_sort = b.src_length
            b.sort_by_src_length()
            after_sort = b.src_length
            self.assertTensorEqual(torch.sort(before_sort, descending=True)[0],
                                   after_sort)
            self.assertEqual(type(b), Batch)

            if total_samples == 0:
                self.assertTensorEqual(b.src, expected_src0)
                self.assertTensorEqual(b.src_length, expected_src0_len)
                self.assertTensorEqual(b.trg, expected_trg0)
                self.assertTensorEqual(b.trg_length, expected_trg0_len)
            total_samples += b.nseqs
            self.assertLessEqual(b.nseqs, batch_size)
        self.assertEqual(total_samples, len(self.dev_data))


