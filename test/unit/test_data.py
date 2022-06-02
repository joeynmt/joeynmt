import unittest

import torch

from joeynmt.data import load_data, make_data_iter


class TestData(unittest.TestCase):

    def setUp(self):
        self.train_path = "test/data/toy/train"
        self.dev_path = "test/data/toy/dev"
        self.test_path = "test/data/toy/test"
        self.max_length = 10
        self.seed = 42

        # minimal data config
        self.data_cfg = {
            "train": self.train_path,
            "dev": self.dev_path,
            "src": {
                "lang": "de",
                "level": "word",
                "lowercase": False,
                "max_length": self.max_length,
            },
            "trg": {
                "lang": "en",
                "level": "word",
                "lowercase": False,
                "max_length": self.max_length,
            },
            "dataset_type": "plain",
        }

    def testIteratorBatchType(self):

        current_cfg = self.data_cfg.copy()

        # load toy data
        _, trg_vocab, train_data, _, _ = load_data(current_cfg, datasets=["train"])

        # make batches by number of sentences
        train_iter = iter(
            make_data_iter(
                train_data,
                batch_size=10,
                batch_type="sentence",
                shuffle=True,
                seed=self.seed,
                pad_index=trg_vocab.pad_index,
                device=torch.device("cpu"),
                num_workers=0,
            ))
        batch = next(train_iter)

        self.assertEqual(batch.src.shape[0], 10)
        self.assertEqual(batch.trg.shape[0], 10)

        # make batches by number of tokens
        train_iter = iter(
            make_data_iter(
                train_data,
                batch_size=100,
                batch_type="token",
                shuffle=True,
                seed=self.seed,
                pad_index=trg_vocab.pad_index,
                device=torch.device("cpu"),
                num_workers=0,
            ))
        _ = next(train_iter)  # skip a batch
        _ = next(train_iter)  # skip another batch
        batch = next(train_iter)

        self.assertEqual(batch.src.shape, (9, 10))
        self.assertLessEqual(batch.ntokens, 77)
