import unittest

from joeynmt.data import load_data
from joeynmt.tokenizers import BasicTokenizer


class TestTokenizer(unittest.TestCase):
    def setUp(self):
        self.train_path = "test/data/toy/train"
        self.dev_path = "test/data/toy/dev"
        self.levels = ["char", "word"]  # bpe is equivalently processed to word
        self.seed = 42

        # minimal data config
        self.data_cfg = {
            "train": self.train_path,
            "dev": self.dev_path,
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
            "dataset_type": "plaintext",
        }

    def testBasicTokenizer(self):
        # first example from the training set
        expected_srcs = {
            "char": "Danke.",
            "word": "David Gallo: Das ist Bill Lange." " Ich bin Dave Gallo.",
        }
        expected_trgs = {
            "char": "Thank you.",
            "word": "David Gallo: This is Bill Lange." " I'm Dave Gallo.",
        }

        # test all combinations of configuration settings
        for level in self.levels:
            for lowercase in [True, False]:
                current_cfg = self.data_cfg.copy()
                current_cfg["src"]["level"] = level
                current_cfg["trg"]["level"] = level
                current_cfg["src"]["lowercase"] = lowercase
                current_cfg["trg"]["lowercase"] = lowercase

                _, _, train_data, _, _ = load_data(current_cfg, datasets=["train"])
                for lang in [train_data.src_lang, train_data.src_lang]:
                    tokenizer = train_data.tokenizer[lang]
                    self.assertIs(type(tokenizer), BasicTokenizer)
                    self.assertIs(tokenizer.lowercase, lowercase)
                    self.assertEqual(tokenizer.level, level)

                # check the first example from the training set
                expected_src = expected_srcs[level]
                expected_trg = expected_trgs[level]
                if lowercase:
                    expected_src = expected_src.lower()
                    expected_trg = expected_trg.lower()

                if level == "char":
                    self.assertEqual(train_data.src[191], expected_src)
                    self.assertEqual(train_data.trg[191], expected_trg)

                    comparison_src = list(expected_src.replace(" ", "▁"))
                    comparison_trg = list(expected_trg.replace(" ", "▁"))

                    train_src, train_trg = train_data[191]
                    self.assertEqual(train_src, comparison_src)
                    self.assertEqual(train_trg, comparison_trg)

                else:
                    self.assertEqual(train_data.src[0], expected_src)
                    self.assertEqual(train_data.trg[0], expected_trg)

                    comparison_src = expected_src.split()
                    comparison_trg = expected_trg.split()

                    train_src, train_trg = train_data[0]
                    self.assertEqual(train_src, comparison_src)
                    self.assertEqual(train_trg, comparison_trg)
