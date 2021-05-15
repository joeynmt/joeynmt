# coding: utf-8
import unittest
import os

from joeynmt.vocabulary import Vocabulary


class TestVocabulary(unittest.TestCase):
    def setUp(self):
        self.file = "test/data/toy/train.de"
        sent = "Die Wahrheit ist, dass die Titanic – obwohl sie alle " \
               "Kinokassenrekorde bricht – nicht gerade die aufregendste " \
               "Geschichte vom Meer ist. GROẞ"  # ẞ (in uppercase) requires Unicode
        self.word_list = sent.split()  # only unique tokens
        self.char_list = list(sent)
        self.temp_file_char = "tmp.src.char"
        self.temp_file_word = "tmp.src.word"
        self.word_vocab = Vocabulary(tokens=sorted(list(set(self.word_list))))
        self.char_vocab = Vocabulary(tokens=sorted(list(set(self.char_list))))

    def testVocabularyFromList(self):
        self.assertEqual(len(self.word_vocab)-len(self.word_vocab.specials),
                         len(set(self.word_list)))
        self.assertEqual(len(self.char_vocab)-len(self.char_vocab.specials),
                         len(set(self.char_list)))
        expected_char_itos = ['<unk>', '<pad>', '<s>', '</s>',
                              ' ', ',', '.', 'D', 'G', 'K', 'M', 'O', 'R', 'T', 'W',
                              'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'k', 'l',
                              'm', 'n', 'o', 'r', 's', 't', 'u', 'v', 'w', 'ẞ', '–']

        self.assertEqual(self.char_vocab.itos, expected_char_itos)
        expected_word_itos = ['<unk>', '<pad>', '<s>', '</s>',
                              'Die', 'GROẞ', 'Geschichte', 'Kinokassenrekorde', 'Meer',
                              'Titanic', 'Wahrheit', 'alle', 'aufregendste',
                              'bricht', 'dass', 'die', 'gerade', 'ist,', 'ist.',
                              'nicht', 'obwohl', 'sie', 'vom', '–']
        self.assertEqual(self.word_vocab.itos, expected_word_itos)

    def testVocabularyFromFile(self):
        # write vocabs to file and create new ones from those files
        self.word_vocab.to_file(self.temp_file_word)
        self.char_vocab.to_file(self.temp_file_char)

        word_vocab2 = Vocabulary(file=self.temp_file_word)
        char_vocab2 = Vocabulary(file=self.temp_file_char)
        self.assertEqual(self.word_vocab.itos, word_vocab2.itos)
        self.assertEqual(self.char_vocab.itos, char_vocab2.itos)
        os.remove(self.temp_file_char)
        os.remove(self.temp_file_word)

    def testIsUnk(self):
        self.assertTrue(self.word_vocab.is_unk("BLA"))
        self.assertFalse(self.word_vocab.is_unk("Die"))
        self.assertFalse(self.word_vocab.is_unk("GROẞ"))
        self.assertTrue(self.char_vocab.is_unk("x"))
        self.assertFalse(self.char_vocab.is_unk("d"))
        self.assertFalse(self.char_vocab.is_unk("ẞ"))
