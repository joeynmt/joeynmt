# coding: utf-8
import unittest
from pathlib import Path

import numpy as np

from joeynmt.data import load_data
from joeynmt.helpers import read_list_from_file
from joeynmt.vocabulary import Vocabulary


class TestVocabulary(unittest.TestCase):

    def setUp(self):
        self.voc_limit = 1000
        self.cfg = {
            "train": "test/data/toy/train",
            "src": {
                "lang": "de",
                "level": "word",
                "lowercase": False,
                "max_length": 30,
                "voc_limit": self.voc_limit,
            },
            "trg": {
                "lang": "en",
                "level": "word",
                "lowercase": False,
                "max_length": 30,
                "voc_limit": self.voc_limit,
            },
            "dataset_type": "plain",
        }

        self.sents = [
            "Die Wahrheit ist, dass die Titanic – obwohl sie alle Kinokassenrekorde "
            "bricht – nicht gerade die aufregendste Geschichte vom Meer ist.",
            "GROẞ",  # ẞ (uppercase) requires Unicode
        ]
        self.word_list = set((" ".join(self.sents)).split())  # only unique tokens
        self.char_list = set(list(" ".join(self.sents)))  # only unique tokens
        self.vocab_file_bpe = Path("test/data/toy/bpe200.txt")
        self.vocab_file_sp = Path("test/data/toy/sp200.txt")
        self.word_vocab = Vocabulary(tokens=sorted(list(self.word_list)))
        self.char_vocab = Vocabulary(tokens=sorted(list(self.char_list)))
        self.specials = ["<unk>", "<pad>", "<s>", "</s>"]

    def testVocabularyFromList(self):
        self.assertEqual(
            len(self.word_vocab) - len(self.word_vocab.specials),
            len(self.word_list),
        )
        self.assertEqual(
            len(self.char_vocab) - len(self.char_vocab.specials),
            len(self.char_list),
        )
        expected_char_itos = [
            " ",
            ",",
            ".",
            "D",
            "G",
            "K",
            "M",
            "O",
            "R",
            "T",
            "W",
            "a",
            "b",
            "c",
            "d",
            "e",
            "f",
            "g",
            "h",
            "i",
            "k",
            "l",
            "m",
            "n",
            "o",
            "r",
            "s",
            "t",
            "u",
            "v",
            "w",
            "ẞ",
            "–",
        ]

        # pylint: disable=protected-access
        self.assertEqual(self.char_vocab._itos, self.specials + expected_char_itos)
        expected_word_itos = [
            "Die",
            "GROẞ",
            "Geschichte",
            "Kinokassenrekorde",
            "Meer",
            "Titanic",
            "Wahrheit",
            "alle",
            "aufregendste",
            "bricht",
            "dass",
            "die",
            "gerade",
            "ist,",
            "ist.",
            "nicht",
            "obwohl",
            "sie",
            "vom",
            "–",
        ]
        self.assertEqual(self.word_vocab._itos, self.specials + expected_word_itos)
        # pylint: enable=protected-access

    def testVocabularyFromFile(self):
        # write vocabs to file and create new ones from those files
        tmp_file_word = Path("tmp.src.word")
        tmp_file_char = Path("tmp.src.char")
        self.word_vocab.to_file(tmp_file_word)
        self.char_vocab.to_file(tmp_file_char)

        word_vocab = Vocabulary(tokens=read_list_from_file(tmp_file_word))
        char_vocab = Vocabulary(tokens=read_list_from_file(tmp_file_char))
        self.assertEqual(self.word_vocab, word_vocab)
        self.assertEqual(self.char_vocab, char_vocab)

        # delete tmp files
        tmp_file_char.unlink()
        tmp_file_word.unlink()

        # pylint: disable=protected-access
        bpe_vocab = Vocabulary(tokens=read_list_from_file(self.vocab_file_bpe))
        expected_bpe_itos = [
            "t@@", "s@@", "e", "e@@", "d@@", "o@@", "b@@", "g@@", "en", "m@@", "u@@"
        ]
        self.assertEqual(bpe_vocab._itos[:15], self.specials + expected_bpe_itos)

        sp_vocab = Vocabulary(tokens=read_list_from_file(self.vocab_file_sp))
        expected_sp_itos = ["▁", "e", "s", "t", "o", "i", "n", "en", "m", "r", "er"]
        self.assertEqual(sp_vocab._itos[:15], self.specials + expected_sp_itos)
        # pylint: enable=protected-access

    def testVocabularyFromDataset(self):
        src_vocab, trg_vocab, _, _, _ = load_data(self.cfg, datasets=["train"])
        self.assertEqual(len(src_vocab), self.voc_limit + len(self.specials))
        self.assertEqual(len(trg_vocab), self.voc_limit + len(self.specials))

        expected_src_itos = [
            "die",
            "und",
            "der",
            "ist",
            "in",
            "das",
            "wir",
            "zu",
            "Sie",
            "es",
            "von",
        ]
        expected_trg_itos = [
            "the",
            "of",
            "to",
            "and",
            "a",
            "that",
            "in",
            "is",
            "you",
            "we",
            "And",
        ]
        # pylint: disable=protected-access
        self.assertEqual(src_vocab._itos[:15], self.specials + expected_src_itos)
        self.assertEqual(trg_vocab._itos[:15], self.specials + expected_trg_itos)

    def testIsUnk(self):
        self.assertTrue(self.word_vocab.is_unk("BLA"))
        self.assertFalse(self.word_vocab.is_unk("Die"))
        self.assertFalse(self.word_vocab.is_unk("GROẞ"))
        self.assertTrue(self.char_vocab.is_unk("x"))
        self.assertFalse(self.char_vocab.is_unk("d"))
        self.assertFalse(self.char_vocab.is_unk("ẞ"))

    def testEncodingDecoding(self):
        tokenized = [s.split() for s in self.sents]
        ids, length = self.word_vocab.sentences_to_ids(tokenized, bos=True, eos=True)
        expected_ids = [
            [
                2, 4, 10, 17, 14, 15, 9, 23, 20, 21, 11, 7, 13, 23, 19, 16, 15, 12, 6,
                22, 8, 18, 3
            ],
            [2, 5, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        ]
        expected_length = [23, 3]
        self.assertEqual(ids, expected_ids)
        self.assertEqual(length, expected_length)

        decoded = self.word_vocab.arrays_to_sentences(np.array(ids),
                                                      cut_at_eos=True,
                                                      skip_pad=True)
        self.assertEqual(decoded[0], ["<s>"] + tokenized[0] + ["</s>"])
        self.assertEqual(decoded[1], ["<s>"] + tokenized[1] + ["</s>"])
