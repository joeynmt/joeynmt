import random
import unittest
from types import SimpleNamespace

from joeynmt.data import load_data
from joeynmt.tokenizers import (
    BasicTokenizer,
    SentencePieceTokenizer,
    SubwordNMTTokenizer,
)


class TestTokenizer(unittest.TestCase):

    def setUp(self):
        self.train_path = "test/data/toy/train"
        self.dev_path = "test/data/toy/dev"

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
            "dataset_type": "plain",
            "special_symbols": SimpleNamespace(
                **{
                    "unk_token": "<unk>",
                    "pad_token": "<pad>",
                    "bos_token": "<s>",
                    "eos_token": "</s>",
                    "sep_token": None,
                    "unk_id": 0,
                    "pad_id": 1,
                    "bos_id": 2,
                    "eos_id": 3,
                    "sep_id": None,
                    "lang_tags": [],
                }
            ),
        }

        # set seed
        seed = 42
        random.seed(seed)

    def testBasicTokenizer(self):
        # first valid example from the training set after filtering
        expected_srcs = {
            "char": "Danke.",
            "word": "David Gallo: Das ist Bill Lange. Ich bin Dave Gallo.",
        }
        expected_trgs = {
            "char": "Thank you.",
            "word": "David Gallo: This is Bill Lange. I'm Dave Gallo.",
        }

        # test all combinations of configuration settings
        for level in ["char", "word"]:
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

                    _, train_src, train_trg = train_data[191]
                    self.assertEqual(train_src, comparison_src)
                    self.assertEqual(train_trg, comparison_trg)

                else:
                    self.assertEqual(train_data.src[0], expected_src)
                    self.assertEqual(train_data.trg[0], expected_trg)

                    comparison_src = expected_src.split()
                    comparison_trg = expected_trg.split()

                    _, train_src, train_trg = train_data[0]
                    self.assertEqual(train_src, comparison_src)
                    self.assertEqual(train_trg, comparison_trg)

    def testSentencepieceTokenizer(self):
        cfg = self.data_cfg.copy()
        for side in ["src", "trg"]:
            cfg[side]["max_length"] = 30
            cfg[side]["level"] = "bpe"
            cfg[side]["tokenizer_type"] = "sentencepiece"
            cfg[side]["tokenizer_cfg"] = {"model_file": "test/data/toy/sp200.model"}
            cfg[side]["voc_file"] = "test/data/toy/sp200.vocab"

        # 6th example from the training set
        expected = {
            "de": {
                "tokenized": [
                    '▁', 'D', 'er', '▁', 'G', 'r', 'o', 'ß', 'te', 'il', '▁der', '▁E',
                    'r', 'd', 'e', '▁ist', '▁M', 'e', 'er', 'w', 'as', 's', 'er', '.'
                ],
                "detokenized": "Der Großteil der Erde ist Meerwasser.",
            }, "en": {
                "tokenized": [
                    '▁M', 'o', 'st', '▁of', '▁the', '▁', 'p', 'l', 'an', 'e', 't',
                    '▁is', '▁', 'o', 'c', 'e', 'an', '▁w', 'at', 'er', '.'
                ],
                "detokenized": "Most of the planet is ocean water.",
            }
        }

        _, _, train_data, _, _ = load_data(cfg, datasets=["train"])

        _, train_src, train_trg = train_data[6]
        for tokenized, lang in [(train_src, train_data.src_lang),
                                (train_trg, train_data.trg_lang)]:
            # check tokenizer
            tokenizer = train_data.tokenizer[lang]
            self.assertIs(type(tokenizer), SentencePieceTokenizer)
            self.assertEqual(tokenizer.level, "bpe")

            # check tokenized sequence
            self.assertEqual(tokenized, expected[lang]['tokenized'])

            # check detokenized sequence
            detokenized = tokenizer.post_process(tokenized)
            self.assertEqual(detokenized, expected[lang]['detokenized'])

            # we cannot set a random seed for sampling.
            # cf) https://github.com/google/sentencepiece/issues/609
            # tokenizer.nbest_size = -1
            # tokenizer.alpha = 0.8
            # dropout = tokenizer(detokenized, is_train=True)
            # self.assertEqual(dropout, expected[lang]['dropout'])

    def testSubwordNMTTokenizer(self):
        cfg = self.data_cfg.copy()
        for side in ["src", "trg"]:
            cfg[side]["max_length"] = 30
            cfg[side]["level"] = "bpe"
            cfg[side]["tokenizer_type"] = "subword-nmt"
            cfg[side]["tokenizer_cfg"] = {"codes": "test/data/toy/bpe200.codes"}
            cfg[side]["voc_file"] = "test/data/toy/bpe200.txt"

        # 191st example from the training set
        expected = {
            "de": {
                "tokenized": ['D@@', 'an@@', 'k@@', 'e.'],
                "dropout": ['D@@', 'a@@', 'n@@', 'k@@', 'e@@', '.'],
                "detokenized": "Danke.",
            }, "en": {
                "tokenized": ['Th@@', 'an@@', 'k', 'y@@', 'ou@@', '.'],
                "dropout": ['T@@', 'ha@@', 'n@@', 'k', 'y@@', 'o@@', 'u@@', '.'],
                "detokenized": "Thank you.",
            }
        }

        _, _, train_data, _, _ = load_data(cfg, datasets=["train"])

        _, train_src, train_trg = train_data[191]
        for tokenized, lang in [(train_src, train_data.src_lang),
                                (train_trg, train_data.trg_lang)]:
            # check tokenizer
            tokenizer = train_data.tokenizer[lang]
            self.assertIs(type(tokenizer), SubwordNMTTokenizer)
            self.assertEqual(tokenizer.level, "bpe")

            # check tokenized sequence
            self.assertEqual(tokenized, expected[lang]['tokenized'])

            # check detokenized sequence
            detokenized = tokenizer.post_process(tokenized)
            self.assertEqual(detokenized, expected[lang]['detokenized'])

            tokenizer.dropout = 0.8
            dropout = tokenizer(detokenized, is_train=True)
            self.assertEqual(dropout, expected[lang]['dropout'])


class TestPrompt(unittest.TestCase):

    def setUp(self):
        self.max_length = 10
        self.min_length = 5

        # minimal data config
        self.data_cfg = {
            "dev": "test/data/toy/dev",
            "src": {
                "lang": "src",
                "level": "bpe",
                "lowercase": False,
                "max_length": 128,
                "min_length": 5,
                "tokenizer_type": "sentencepiece",
                "tokenizer_cfg": {"model_file": "test/data/toy/sp200.model"},
                "voc_file": "test/data/toy/sp200.vocab",
            },
            "trg": {
                "lang": "trg",
                "level": "bpe",
                "lowercase": False,
                "max_length": 128,
                "min_length": 5,
                "tokenizer_type": "sentencepiece",
                "tokenizer_cfg": {"model_file": "test/data/toy/sp200.model"},
                "voc_file": "test/data/toy/sp200.vocab",
            },
            "sample_dev_subset": -1,
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

    def testToknizerWithPrompt(self):
        _, _, _, dev_data, _ = load_data(self.data_cfg, datasets=["dev"])
        self.assertEqual(len(dev_data), 40)

        expected = {
            "src": [
                '<de>', '▁', 'J', 'a', '▁', ',', '▁', 'g', 'ut', 'en', '▁T', 'a', 'g',
                '▁', '.', '<sep>', '▁', 'J', 'a', '▁', ',', '▁', 'al', 's', 'o', '▁',
                ',', '▁was', '▁so', 'll', '▁B', 'i', 'o', 'h', 'a', 'c', 'k', 'ing',
                '▁', 'se', 'in', '▁', '?',
            ],
            "trg": [
                '<en>', '▁', 'Y', 'es', '▁', ',', '▁h', 'e', 'll', 'o', '▁', '.',
                '<sep>', '▁', 'Y', 'es', '▁', ',', '▁so', '▁', ',', '▁w', 'h', 'at',
                '▁is', '▁b', 'i', 'o', 'h', 'a', 'c', 'k', 'ing', '▁', '?',
            ],
        }  # yapf: disable

        dev_src, dev_trg = dev_data.src, dev_data.trg
        _, dev_src_2, dev_trg_2 = dev_data[2]

        for tokenized, orig, side in [(dev_src_2, dev_src[2], dev_data.src_lang),
                                      (dev_trg_2, dev_trg[2], dev_data.trg_lang)]:
            tokenizer = dev_data.tokenizer[side]

            # check tokenized sequence
            self.assertEqual(tokenized, expected[side])

            # check detokenized sequence
            detokenized = tokenizer.post_process(tokenized)
            self.assertEqual(detokenized, orig)
