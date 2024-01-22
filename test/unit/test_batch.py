import unittest
from types import SimpleNamespace

import torch

from joeynmt.batch import Batch
from joeynmt.data import load_data


class TestBatch(unittest.TestCase):

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
            "special_symbols": SimpleNamespace(
                **{
                    "unk_token": "<unk>",
                    "pad_token": "<pad>",
                    "bos_token": "<s>",
                    "eos_token": "</s>",
                    "sep_token": "<sep>",
                    "unk_id": 0,
                    "pad_id": 1,
                    "bos_id": 2,
                    "eos_id": 3,
                    "sep_id": 4,
                    "lang_tags": ["<de>", "<en>"],
                }
            ),
        }

        # load the data
        _, trg_vocab, self.train_data, self.dev_data, _ = load_data(
            data_cfg, datasets=["train", "dev"]
        )
        self.eos_index = trg_vocab.eos_index
        self.pad_index = trg_vocab.pad_index
        # random seed
        self.seed = 42

    def testBatchTrainIterator(self):

        batch_size = 4
        # load  all sents, filtering happens during batch construction
        self.assertEqual(len(self.train_data), 1000)

        # make data iterator
        train_iter = self.train_data.make_iter(
            batch_size=batch_size,
            batch_type="sentence",
            shuffle=True,
            seed=self.seed,
            pad_index=self.pad_index,
            device=torch.device("cpu"),
        )

        expected_src0 = torch.LongTensor([
            [30, 10, 8, 17, 8, 7, 30, 8, 12, 33, 9, 15, 8, 12, 18, 9, 20, 8, 9, 27, 3],
            [22, 28, 14, 40, 27, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [31, 26, 8, 10, 26, 8, 9, 7, 15, 14, 9, 28, 8, 7, 11, 19, 16, 34, 9, 27, 3],
            [15, 14, 11, 7, 10, 11, 13, 7, 31, 14, 11, 11, 10, 8, 12, 13, 27, 3, 1,
             1, 1],
        ])  # yapf: disable
        expected_src0_len = torch.LongTensor([21, 6, 21, 18])
        expected_trg0 = torch.LongTensor([
            [10, 7, 17, 11, 9, 7, 11, 26, 7, 20, 16, 10, 13, 24, 8, 27, 3],
            [11, 31, 10, 21, 27, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [9, 16, 10, 13, 31, 7, 21, 11, 19, 27, 3, 1, 1, 1, 1, 1, 1],
            [12, 9, 7, 18, 12, 18, 27, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        ])
        expected_indices = torch.LongTensor([531, 873, 807, 245])

        total_samples = 0
        for b in train_iter:
            self.assertTrue(isinstance(b, Batch))
            if total_samples == 0:
                torch.testing.assert_close(b.src, expected_src0)
                torch.testing.assert_close(b.src_length, expected_src0_len)
                torch.testing.assert_close(b.trg, expected_trg0)
                torch.testing.assert_close(b.indices, expected_indices)
            total_samples += b.nseqs
            self.assertLessEqual(b.nseqs, batch_size)
        self.assertEqual(total_samples, 27)

    def testTokenBatchTrainIterator(self):

        batch_size = 50  # num of tokens in one batch

        # make data iterator
        train_iter = self.train_data.make_iter(
            batch_size=batch_size,
            batch_type="token",
            shuffle=True,
            seed=self.seed,
            pad_index=self.pad_index,
            device=torch.device("cpu"),
        )

        expected_src0 = torch.LongTensor([
            [30, 10, 8, 17, 8, 7, 30, 8, 12, 33, 9, 15, 8, 12, 18, 9, 20, 8, 9, 27, 3],
            [22, 28, 14, 40, 27, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [31, 26, 8, 10, 26, 8, 9, 7, 15, 14, 9, 28, 8, 7, 11, 19, 16, 34, 9, 27, 3]
        ])
        expected_src0_len = torch.LongTensor([21, 6, 21])
        expected_trg0 = torch.LongTensor([
            [10, 7, 17, 11, 9, 7, 11, 26, 7, 20, 16, 10, 13, 24, 8, 27, 3],
            [11, 31, 10, 21, 27, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [9, 16, 10, 13, 31, 7, 21, 11, 19, 27, 3, 1, 1, 1, 1, 1, 1],
        ])
        expected_indices = torch.LongTensor([531, 873, 807])

        total_tokens = 0
        for b in train_iter:
            self.assertTrue(isinstance(b, Batch))
            if total_tokens == 0:
                torch.testing.assert_close(b.src, expected_src0)
                torch.testing.assert_close(b.src_length, expected_src0_len)
                torch.testing.assert_close(b.trg, expected_trg0)
                torch.testing.assert_close(b.indices, expected_indices)
            total_tokens += b.ntokens
        self.assertEqual(total_tokens, 387)

    def testBatchDevIterator(self):
        batch_size = 3

        # make data iterator
        dev_iter = self.dev_data.make_iter(
            batch_size=batch_size,
            batch_type="sentence",
            shuffle=False,
            pad_index=self.pad_index,
            device=torch.device("cpu"),
        )

        expected_src0 = torch.LongTensor([
            [35, 14, 7, 25, 7, 14, 17, 11, 22, 7, 25, 7, 24, 14, 11, 7, 11, 22, 17, 17,
             7, 23, 10, 22, 16, 14, 19, 28, 10, 9, 20, 7, 11, 8, 10, 9, 7, 41, 3],
            [10, 19, 16, 7, 26, 12, 8, 18, 8, 7, 21, 10, 19, 16, 7, 25, 7, 15, 14, 11,
             11, 7, 10, 19, 16, 7, 15, 14, 7, 23, 10, 9, 7, 27, 3, 1, 1, 1, 1],
            [35, 14, 7, 25, 7, 20, 18, 13, 8, 9, 7, 13, 14, 20, 7, 27, 3, 1, 1, 1, 1,
             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        ])  # yapf: disable
        expected_before_sort_len = torch.LongTensor([35, 17, 39])
        expected_after_sort_len = torch.LongTensor([39, 35, 17])
        expected_trg0 = torch.LongTensor([
            [21, 8, 14, 7, 29, 7, 14, 11, 7, 29, 7, 22, 16, 10, 9, 7, 12, 14,
             7, 28, 12, 11, 16, 10, 20, 31, 12, 13, 24, 7, 37, 3],
            [12, 0, 23, 7, 16, 10, 25, 25, 21, 7, 9, 11, 7, 28, 8, 7, 16, 8,
             15, 8, 7, 27, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [21, 8, 14, 7, 29, 7, 16, 8, 17, 17, 11, 7, 27, 3, 1, 1, 1, 1,
             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        ])  # yapf: disable
        expected_before_sort_idx = torch.LongTensor([0, 1, 2])
        expected_after_sort_idx = torch.LongTensor([2, 0, 1])

        total_samples = 0
        for b in dev_iter:
            self.assertTrue(isinstance(b, Batch))

            # test the sorting by src length
            before_sort = b.src_length
            before_sort_idx = b.indices
            b.sort_by_src_length()
            after_sort = b.src_length
            after_sort_idx = b.indices
            torch.testing.assert_close(
                torch.sort(before_sort, descending=True)[0], after_sort
            )
            if total_samples == 0:
                torch.testing.assert_close(b.src, expected_src0)
                torch.testing.assert_close(before_sort, expected_before_sort_len)
                torch.testing.assert_close(after_sort, expected_after_sort_len)
                torch.testing.assert_close(b.trg, expected_trg0)
                torch.testing.assert_close(before_sort_idx, expected_before_sort_idx)
                torch.testing.assert_close(after_sort_idx, expected_after_sort_idx)
            total_samples += b.nseqs
            self.assertLessEqual(b.nseqs, batch_size)
        self.assertEqual(total_samples, len(self.dev_data))


class TestPrompt(unittest.TestCase):

    def setUp(self):
        # minimal data config
        data_cfg = {
            "dev": "test/data/toy/dev",
            "src": {
                "lang": "src",
                "level": "bpe",
                "lowercase": False,
                "tokenizer_type": "sentencepiece",
                "tokenizer_cfg": {"model_file": "test/data/toy/sp200.model"},
                "voc_file": "test/data/toy/sp200.vocab",
            },
            "trg": {
                "lang": "trg",
                "level": "bpe",
                "lowercase": False,
                "tokenizer_type": "sentencepiece",
                "tokenizer_cfg": {"model_file": "test/data/toy/sp200.model"},
                "voc_file": "test/data/toy/sp200.vocab",
            },
            "dataset_type": "tsv",
            "special_symbols": SimpleNamespace(
                **{
                    "unk_token": "<unk>",
                    "pad_token": "<pad>",
                    "bos_token": "<s>",
                    "eos_token": "</s>",
                    "sep_token": "<sep>",
                    "unk_id": 0,
                    "pad_id": 1,
                    "bos_id": 2,
                    "eos_id": 3,
                    "sep_id": 4,
                    "lang_tags": ["<de>", "<en>"],
                }
            ),
        }
        _, trg_vocab, _, self.dev_data, _ = load_data(data_cfg, datasets=["dev"])
        self.pad_index = trg_vocab.pad_index
        self.eos_index = trg_vocab.eos_index

    def testBatchWithPrompt(self):
        batch_size = 2

        # make data iterator
        dev_iter = self.dev_data.make_iter(
            batch_size=batch_size,
            batch_type="sentence",
            shuffle=False,
            eos_index=self.eos_index,
            pad_index=self.pad_index,
            device=torch.device("cpu"),
        )

        # yapf: disable
        expected_src0 = torch.LongTensor([
            [5, 48, 33, 86, 34, 23, 8, 7, 15, 12, 33, 7, 19, 88, 9, 7, 12, 33, 149, 66,
             36, 7, 18, 4, 7, 196, 24, 7, 19, 7, 26, 69, 14, 120, 24, 26, 7, 18, 3],
            [5, 4, 48, 33, 86, 34, 23, 8, 7, 15, 12, 33, 7, 19, 88, 9, 7, 12, 33, 149,
             66, 36, 7, 18, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        ])
        expected_src_prompt_mask0 = torch.LongTensor([
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        ])
        expected_src_len0 = torch.LongTensor([39, 25])
        expected_trg_input0 = torch.LongTensor([
            [2, 6, 48, 0, 15, 130, 25, 25, 31, 58, 63, 72, 17, 8, 7, 18, 4,
             7, 192, 50, 7, 19, 72, 8, 75, 11, 7, 18],
            [2, 6, 4, 48, 0, 15, 130, 25, 25, 31, 58, 63, 72, 17, 8, 7, 18,
             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        ])
        expected_trg_prompt_mask0 = torch.LongTensor([
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ])
        # yapf: enable

        total_samples = 0
        for b in dev_iter:
            self.assertTrue(isinstance(b, Batch))

            # test the sorting by src length
            before_sort = b.src_length
            b.sort_by_src_length()
            after_sort = b.src_length
            torch.testing.assert_close(
                torch.sort(before_sort, descending=True)[0], after_sort
            )
            if total_samples == 0:
                torch.testing.assert_close(b.src, expected_src0)
                torch.testing.assert_close(b.src_prompt_mask, expected_src_prompt_mask0)
                torch.testing.assert_close(b.src_length, expected_src_len0)
                torch.testing.assert_close(b.trg_input, expected_trg_input0)
                torch.testing.assert_close(b.trg_prompt_mask, expected_trg_prompt_mask0)
            total_samples += b.nseqs
            self.assertLessEqual(b.nseqs, batch_size)
        self.assertEqual(total_samples, len(self.dev_data))
