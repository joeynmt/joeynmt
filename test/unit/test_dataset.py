import tempfile
import unittest
from pathlib import Path

from joeynmt.data import load_data
from joeynmt.datasets import PlaintextDataset, TsvDataset
from joeynmt.helpers import read_list_from_file, write_list_to_file


class TestPlaintextDataset(unittest.TestCase):

    def setUp(self):
        self.train_path = "test/data/toy/train"
        self.dev_path = "test/data/toy/dev"
        self.test_path = "test/data/toy/test"
        self.max_length = 10
        self.min_length = 5
        self.seed = 42

        # minimal data config
        self.data_cfg = {
            "train": self.train_path,
            "dev": self.dev_path,
            "test": self.test_path,
            "src": {
                "lang": "de",
                "level": "word",
                "lowercase": False,
                "max_length": self.max_length,
                "min_length": self.min_length,
            },
            "trg": {
                "lang": "en",
                "level": "word",
                "lowercase": False,
                "max_length": self.max_length,
                "min_length": self.min_length,
            },
            "random_train_subset": 100,
            "random_dev_subset": 100,
            "dataset_type": "plain",
        }

    def testDataLoading(self):
        # load the data
        datasets = ["train", "dev"]
        for test_path in [None, self.test_path]:
            current_cfg = self.data_cfg.copy()
            if test_path is not None:
                if "test" not in datasets:
                    datasets.append("test")
                current_cfg["test"] = test_path
            else:
                if "test" in datasets:
                    datasets.remove("test")

            _, _, train_data, dev_data, test_data = load_data(current_cfg,
                                                              datasets=datasets)

            self.assertIs(type(train_data), PlaintextDataset)
            self.assertIs(type(dev_data), PlaintextDataset)

            self.assertEqual(train_data.split, "train")
            self.assertEqual(dev_data.split, "dev")

            self.assertTrue(train_data.has_trg)
            self.assertTrue(dev_data.has_trg)

            if test_path is None:
                self.assertIsNone(test_data)
            else:
                self.assertIs(type(test_data), PlaintextDataset)
                self.assertEqual(test_data.split, "test")
                self.assertFalse(test_data.has_trg)

            self.assertEqual(train_data.tokenizer[train_data.src_lang].max_length,
                             self.max_length)
            self.assertEqual(train_data.tokenizer[train_data.trg_lang].max_length,
                             self.max_length)

            self.assertEqual(train_data.tokenizer[train_data.src_lang].min_length,
                             self.min_length)
            self.assertEqual(train_data.tokenizer[train_data.trg_lang].min_length,
                             self.min_length)

            # check the number of examples loaded
            # NOTE: since tokenization is applied in batch construction,
            # we cannot compute the length and therefore cannot filter
            # examples out based on the length before batch iteration.
            expected_train_len, after_filtering = 1000, 290
            expected_testdev_len = 20  # dev and test have the same len
            self.assertEqual(len(train_data), expected_train_len)
            self.assertEqual(len(dev_data), expected_testdev_len)
            if test_path is None:
                self.assertIsNone(test_data)
            else:
                self.assertEqual(len(test_data), expected_testdev_len)

            train_ex = [train_data[i] for i in range(len(train_data))]
            train_ex = [(s, t) for s, t in train_ex if s is not None and t is not None]
            self.assertEqual(len(train_ex), after_filtering)

            # check the segmentation: src and trg attributes are lists
            train_src, train_trg = train_ex[0]
            dev_src, dev_trg = dev_data[0]
            self.assertIs(type(train_src), list)
            self.assertIs(type(train_trg), list)
            self.assertIs(type(dev_src), list)
            self.assertIs(type(dev_trg), list)
            if test_path is not None:
                test_src, test_trg = test_data[0]
                self.assertIs(type(test_src), list)
                self.assertIs(test_trg, None)

            # check the length filtering of the training examples
            src_len, trg_len = zip(*train_ex)
            self.assertTrue(
                all(self.min_length <= len(s) <= self.max_length for s in src_len))
            self.assertTrue(
                all(self.min_length <= len(t) <= self.max_length for t in trg_len))

            # check the lowercasing
            if current_cfg["src"]["lowercase"]:
                self.assertTrue(all(ex.lower() == ex for ex in train_data.src))
                self.assertTrue(all(ex.lower() == ex for ex in dev_data.src))
                if test_path is not None:
                    self.assertTrue(all(ex.lower() == ex for ex in test_data.src))
            if current_cfg["trg"]["lowercase"]:
                self.assertTrue(all(ex.lower() == ex for ex in train_data.trg))
                self.assertTrue(all(ex.lower() == ex for ex in dev_data.trg))

            # check dev: no length filtering
            dev_ex = [dev_data[i] for i in range(len(dev_data))]
            dev_src, dev_trg = zip(*dev_ex)
            self.assertEqual(len(dev_ex), expected_testdev_len)
            self.assertEqual(min([len(t) for t in dev_trg]), 4)
            self.assertEqual(max([len(t) for t in dev_trg]), 46)
            self.assertTrue(all(t is not None for t in dev_src))
            self.assertTrue(all(t is not None for t in dev_trg))

    def testRandomSubset(self):
        # Load data
        _, _, train_data, _, test_data = load_data(self.data_cfg,
                                                   datasets=["train", "test"])
        self.assertEqual(len(train_data), 1000)
        self.assertEqual(train_data.random_subset, 100)
        train_data.sample_random_subset(seed=self.seed)
        self.assertEqual(len(train_data), 100)

        train_data.reset_random_subset()
        self.assertEqual(len(train_data), 1000)

        # a random subset can be selected only when len(train_data) > n
        train_data.random_subset = 2000
        with self.assertRaises(AssertionError) as e:
            train_data.sample_random_subset(seed=self.seed)
        self.assertEqual("Can only subsample from train or dev set larger than 2000.",
                         str(e.exception))

        # a random subset should be selected for training only
        self.assertEqual(test_data.random_subset, -1)
        test_data.random_subset = 100
        with self.assertRaises(AssertionError) as e:
            test_data.sample_random_subset(seed=self.seed)
        self.assertEqual("Can only subsample from train or dev set larger than 100.",
                         str(e.exception))


class TestTsvDataset(unittest.TestCase):

    def setUp(self):
        # save toy data temporarily in tsv format
        def _read_write_sents(path, src_lang, trg_lang):
            with tempfile.NamedTemporaryFile(prefix="joeynmt_unittest_",
                                             suffix=".tsv",
                                             delete=False) as temp:
                tsv_file = Path(temp.name)
                src = read_list_from_file(Path(path).with_suffix(f".{src_lang}"))
                if trg_lang:
                    trg = read_list_from_file(Path(path).with_suffix(f".{trg_lang}"))
                    lines = [f"{src_lang}\t{trg_lang}"
                             ] + [f"{s}\t{t}" for s, t in zip(src, trg)]
                else:
                    lines = [src_lang] + src
                write_list_to_file(tsv_file, lines)
            return tsv_file

        train_file = _read_write_sents("test/data/toy/train", "de", "en")
        dev_file = _read_write_sents("test/data/toy/dev", "de", "en")
        test_file = _read_write_sents("test/data/toy/test", "de", None)

        self.max_length = 10
        self.min_length = 5

        # random seed for subsampling
        self.seed = 42

        # minimal data config
        self.data_cfg = {
            "train": (train_file.parent / train_file.stem).as_posix(),
            "dev": (dev_file.parent / dev_file.stem).as_posix(),
            "test": (test_file.parent / test_file.stem).as_posix(),
            "src": {
                "lang": "de",
                "level": "word",
                "lowercase": False,
                "max_length": self.max_length,
                "min_length": self.min_length,
            },
            "trg": {
                "lang": "en",
                "level": "word",
                "lowercase": False,
                "max_length": self.max_length,
                "min_length": self.min_length,
            },
            "random_train_subset": 100,
            "random_dev_subset": 100,
            "dataset_type": "tsv",
        }

    def tearDown(self):
        # delete tmp files
        train_file = Path(self.data_cfg["train"]).with_suffix(".tsv")
        dev_file = Path(self.data_cfg["dev"]).with_suffix(".tsv")
        test_file = Path(self.data_cfg["test"]).with_suffix(".tsv")

        if train_file.is_file():
            train_file.unlink()
        if dev_file.is_file():
            dev_file.unlink()
        if test_file.is_file():
            test_file.unlink()

    def testDataLoading(self):
        try:
            expected_train_len, after_filtering = 1000, 290
            expected_dev_len = 20
            _, _, train_data, dev_data, _ = load_data(self.data_cfg,
                                                      datasets=["train", "dev"])

            self.assertIs(type(train_data), TsvDataset)
            self.assertIs(type(dev_data), TsvDataset)

            self.assertEqual(len(train_data), expected_train_len)
            self.assertEqual(len(dev_data), expected_dev_len)

            self.assertEqual(train_data.split, "train")
            self.assertEqual(dev_data.split, "dev")

            self.assertTrue(train_data.has_trg)
            self.assertTrue(dev_data.has_trg)

            self.assertEqual(train_data.tokenizer[train_data.src_lang].max_length,
                             self.max_length)
            self.assertEqual(train_data.tokenizer[train_data.trg_lang].max_length,
                             self.max_length)

            self.assertEqual(train_data.tokenizer[train_data.src_lang].min_length,
                             self.min_length)
            self.assertEqual(train_data.tokenizer[train_data.trg_lang].min_length,
                             self.min_length)

            train_ex = [train_data[i] for i in range(len(train_data))]
            train_ex = [(s, t) for s, t in train_ex if s is not None and t is not None]
            self.assertEqual(len(train_ex), after_filtering)

            # check the length filtering of the training examples
            src_len, trg_len = zip(*train_ex)
            self.assertTrue(
                all(self.min_length <= len(s) <= self.max_length for s in src_len))
            self.assertTrue(
                all(self.min_length <= len(t) <= self.max_length for t in trg_len))

            dev_ex = [dev_data[i] for i in range(len(dev_data))]
            dev_src, dev_trg = zip(*dev_ex)
            self.assertEqual(len(dev_ex), expected_dev_len)
            self.assertEqual(min([len(t) for t in dev_trg]), 4)
            self.assertEqual(max([len(t) for t in dev_trg]), 46)
            self.assertTrue(all(t is not None for t in dev_src))
            self.assertTrue(all(t is not None for t in dev_trg))

        except ImportError as e:
            # need pandas installed.
            raise unittest.SkipTest(f"{e} Skip.")

    def testRandomSubset(self):
        try:
            # load the data
            _, _, train_data, _, test_data = load_data(self.data_cfg,
                                                       datasets=["train", "test"])
            self.assertEqual(len(train_data), 1000)
            self.assertEqual(train_data.random_subset, 100)
            train_data.sample_random_subset(seed=self.seed)
            self.assertEqual(len(train_data), 100)

            # a random subset can be selected only when len(train_data) > n
            train_data.random_subset = 2000
            with self.assertRaises(AssertionError) as e:
                train_data.sample_random_subset(seed=self.seed)
            self.assertEqual(
                "Can only subsample from train or dev set larger than 2000.",
                str(e.exception))

            # a random subset should be selected for training only
            self.assertEqual(test_data.random_subset, -1)
            test_data.random_subset = 100
            with self.assertRaises(AssertionError) as e:
                test_data.sample_random_subset(seed=self.seed)
            self.assertEqual(
                "Can only subsample from train or dev set larger than 100.",
                str(e.exception))

        except ImportError as e:
            raise unittest.SkipTest(f"{e} Skip.")
