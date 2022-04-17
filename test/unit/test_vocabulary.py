# coding: utf-8
import unittest
from pathlib import Path

from joeynmt.helpers import read_list_from_file
from joeynmt.vocabulary import Vocabulary


class TestVocabulary(unittest.TestCase):
    def setUp(self):
        self.file = "test/data/toy/train.de"
        sent = (
            "Die Wahrheit ist, dass die Titanic – obwohl sie alle "
            "Kinokassenrekorde bricht – nicht gerade die aufregendste "
            "Geschichte vom Meer ist. GROẞ"
        )  # ẞ (uppercase) requires Unicode
        self.word_list = sent.split()  # only unique tokens
        self.char_list = list(sent)
        self.tmp_file_char = Path("tmp.src.char")
        self.tmp_file_word = Path("tmp.src.word")
        self.word_vocab = Vocabulary(tokens=sorted(list(set(self.word_list))))
        self.char_vocab = Vocabulary(tokens=sorted(list(set(self.char_list))))

    def testVocabularyFromList(self):
        self.assertEqual(
            len(self.word_vocab) - len(self.word_vocab.specials),
            len(set(self.word_list)),
        )
        self.assertEqual(
            len(self.char_vocab) - len(self.char_vocab.specials),
            len(set(self.char_list)),
        )
        expected_char_itos = [
            "<unk>",
            "<pad>",
            "<s>",
            "</s>",
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
        self.assertEqual(self.char_vocab._itos, expected_char_itos)
        expected_word_itos = [
            "<unk>",
            "<pad>",
            "<s>",
            "</s>",
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
        self.assertEqual(self.word_vocab._itos, expected_word_itos)
        # pylint: enable=protected-access

    def testVocabularyFromFile(self):
        # write vocabs to file and create new ones from those files
        self.word_vocab.to_file(self.tmp_file_word)
        self.char_vocab.to_file(self.tmp_file_char)

        word_vocab2 = Vocabulary(tokens=read_list_from_file(self.tmp_file_word))
        char_vocab2 = Vocabulary(tokens=read_list_from_file(self.tmp_file_char))
        self.assertEqual(self.word_vocab, word_vocab2)
        self.assertEqual(self.char_vocab, char_vocab2)
        self.tmp_file_char.unlink()
        self.tmp_file_word.unlink()

    def testIsUnk(self):
        self.assertTrue(self.word_vocab.is_unk("BLA"))
        self.assertFalse(self.word_vocab.is_unk("Die"))
        self.assertFalse(self.word_vocab.is_unk("GROẞ"))
        self.assertTrue(self.char_vocab.is_unk("x"))
        self.assertFalse(self.char_vocab.is_unk("d"))
        self.assertFalse(self.char_vocab.is_unk("ẞ"))
