import unittest

import numpy as np

from joeynmt.data import MonoDataset, TranslationDataset, load_data, \
    make_data_iter

class TestData(unittest.TestCase):

    def setUp(self):
        self.train_path = "test/data/toy/train"
        self.dev_path = "test/data/toy/dev"
        self.test_path = "test/data/toy/test"
        self.levels = ["char", "word"]  # bpe is equivalently processed to word
        self.max_sent_length = 10

        # minimal data config
        self.data_cfg = {"src": "de", "trg": "en", "train": self.train_path,
                         "dev": self.dev_path, "level": "word",
                         "lowercase": False,
                         "max_sent_length": self.max_sent_length}

    def testIteratorBatchType(self):

        current_cfg = self.data_cfg.copy()

        # load toy data
        train_data, dev_data, test_data, src_vocab, trg_vocab = \
            load_data(current_cfg)

        # make batches by number of sentences
        train_iter = iter(make_data_iter(
            train_data, batch_size=10, batch_type="sentence"))
        batch = next(train_iter)

        self.assertEqual(batch.src[0].shape[0], 10)
        self.assertEqual(batch.trg[0].shape[0], 10)

        # make batches by number of tokens
        train_iter = iter(make_data_iter(
            train_data, batch_size=100, batch_type="token"))
        _ = next(train_iter)  # skip a batch
        _ = next(train_iter)  # skip another batch
        batch = next(train_iter)

        self.assertEqual(batch.src[0].shape[0], 8)
        self.assertEqual(np.prod(batch.src[0].shape), 88)
        self.assertLessEqual(np.prod(batch.src[0].shape), 100)

    def testDataLoading(self):
        # test all combinations of configuration settings
        for test_path in [None, self.test_path]:
            for level in self.levels:
                for lowercase in [True, False]:
                    current_cfg = self.data_cfg.copy()
                    current_cfg["level"] = level
                    current_cfg["lowercase"] = lowercase
                    if test_path is not None:
                        current_cfg["test"] = test_path

                    # load the data
                    train_data, dev_data, test_data, src_vocab, trg_vocab = \
                        load_data(current_cfg)

                    self.assertIs(type(train_data), TranslationDataset)
                    self.assertIs(type(dev_data), TranslationDataset)
                    if test_path is not None:
                        # test has no target side
                        self.assertIs(type(test_data), MonoDataset)

                    # check the number of examples loaded
                    if level == "char":
                        # training set is filtered to max_sent_length
                        expected_train_len = 5
                    else:
                        expected_train_len = 382
                    expected_testdev_len = 20  # dev and test have the same len
                    self.assertEqual(len(train_data), expected_train_len)
                    self.assertEqual(len(dev_data), expected_testdev_len)
                    if test_path is None:
                        self.assertIsNone(test_data)
                    else:
                        self.assertEqual(len(test_data), expected_testdev_len)

                    # check the segmentation: src and trg attributes are lists
                    self.assertIs(type(train_data.examples[0].src), list)
                    self.assertIs(type(train_data.examples[0].trg), list)
                    self.assertIs(type(dev_data.examples[0].src), list)
                    self.assertIs(type(dev_data.examples[0].trg), list)
                    if test_path is not None:
                        self.assertIs(type(test_data.examples[0].src), list)
                        self.assertFalse(hasattr(test_data.examples[0], "trg"))

                    # check the length filtering of the training examples
                    self.assertFalse(any([len(ex.src) > self.max_sent_length for
                                          ex in train_data.examples]))
                    self.assertFalse(any([len(ex.trg) > self.max_sent_length for
                                          ex in train_data.examples]))

                    # check the lowercasing
                    if lowercase:
                        self.assertTrue(
                            all([" ".join(ex.src).lower() == " ".join(ex.src)
                                 for ex in train_data.examples]))
                        self.assertTrue(
                            all([" ".join(ex.src).lower() == " ".join(ex.src)
                                 for ex in dev_data.examples]))
                        self.assertTrue(
                            all([" ".join(ex.trg).lower() == " ".join(ex.trg)
                                 for ex in train_data.examples]))
                        self.assertTrue(
                            all([" ".join(ex.trg).lower() == " ".join(ex.trg)
                                 for ex in dev_data.examples]))
                        if test_path is not None:
                            self.assertTrue(
                                all([" ".join(ex.src).lower() == " ".join(
                                    ex.src) for ex in test_data.examples]))

                    # check the first example from the training set
                    expected_srcs = {"char": "Danke.",
                                     "word": "David Gallo: Das ist Bill Lange."
                                             " Ich bin Dave Gallo."}
                    expected_trgs = {"char": "Thank you.",
                                     "word": "David Gallo: This is Bill Lange. "
                                             "I'm Dave Gallo."}
                    if level == "char":
                        if lowercase:
                            comparison_src = list(expected_srcs[level].lower())
                            comparison_trg = list(expected_trgs[level].lower())
                        else:
                            comparison_src = list(expected_srcs[level])
                            comparison_trg = list(expected_trgs[level])
                    else:
                        if lowercase:
                            comparison_src = expected_srcs[level].lower().\
                                split()
                            comparison_trg = expected_trgs[level].lower(). \
                                split()
                        else:
                            comparison_src = expected_srcs[level].split()
                            comparison_trg = expected_trgs[level].split()
                    self.assertEqual(train_data.examples[0].src, comparison_src)
                    self.assertEqual(train_data.examples[0].trg, comparison_trg)

    def testRandomSubset(self):
        # only a random subset should be selected for training
        current_cfg = self.data_cfg.copy()
        current_cfg["random_train_subset"] = -1

        # load the data
        train_data, dev_data, test_data, src_vocab, trg_vocab = \
            load_data(current_cfg)
        assert len(train_data) == 382

        current_cfg["random_train_subset"] = 10
        train_data, dev_data, test_data, src_vocab, trg_vocab = \
            load_data(current_cfg)
        assert len(train_data) == 10
