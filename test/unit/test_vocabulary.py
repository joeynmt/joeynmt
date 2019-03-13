import unittest
import os

from joeynmt.vocabulary import Vocabulary


class TestVocabulary(unittest.TestCase):
    def setUp(self):
        self.file = "test/data/iwslt/train.de"
        sent = "Die Wahrheit ist, dass die Titanic – obwohl sie alle " \
               "Kinokassenrekorde bricht – nicht gerade die aufregendste " \
               "Geschichte vom Meer ist."
        self.word_list = sent.split()  # only unique tokens
        self.char_list = list(sent)
        self.temp_file_char = "tmp.src.char"
        self.temp_file_word = "tmp.src.word"

    def testVocabularyFromListAndFile(self):
        word_vocab = Vocabulary(tokens=self.word_list)
        char_vocab = Vocabulary(tokens=self.char_list)
        self.assertEqual(len(word_vocab), len(set(self.word_list)))
        self.assertEqual(len(char_vocab), len(set(self.char_list)))
        expected_char_stoi = {'M': 6, 'f': 14, 'g': 15, 'W': 8, 'l': 19,
                              'K': 5, 'n': 21, 'b': 10, '–': 29, 'k': 18,
                              'c': 11, 'w': 28, 't': 25, ' ': 0, 'G': 4,
                              'h': 16, 'e': 13, 'd': 12, 'r': 23, 'u': 26,
                              '.': 2, 'v': 27, ',': 1, 's': 24, 'D': 3, 'T': 7,
                              'o': 22, 'i': 17, 'a': 9, 'm': 20}
        expected_char_itos = [' ', ',', '.', 'D', 'G', 'K', 'M', 'T', 'W', 'a',
                              'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'k', 'l',
                              'm', 'n', 'o', 'r', 's', 't', 'u', 'v', 'w', '–']
        self.assertEqual(char_vocab.stoi, expected_char_stoi)
        self.assertEqual(char_vocab.itos, expected_char_itos)
        expected_word_stoi = {'Die': 0, 'Titanic': 4, 'alle': 6, 'Wahrheit': 5,
                              'ist,': 12, 'Geschichte': 1, 'sie': 16, '–': 18,
                              'Kinokassenrekorde': 2, 'vom': 17, 'ist.': 13,
                              'obwohl': 15, 'die': 10, 'Meer': 3, 'nicht': 14,
                              'bricht': 8, 'dass': 9, 'gerade': 11,
                              'aufregendste': 7}
        expected_word_itos = ['Die', 'Geschichte', 'Kinokassenrekorde', 'Meer',
                              'Titanic', 'Wahrheit', 'alle', 'aufregendste',
                              'bricht', 'dass', 'die', 'gerade', 'ist,', 'ist.',
                              'nicht', 'obwohl', 'sie', 'vom', '–']
        self.assertEqual(word_vocab.stoi, expected_word_stoi)
        self.assertEqual(word_vocab.itos, expected_word_itos)

        # write vocabs to file and create new ones from those files
        word_vocab.to_file(self.temp_file_word)
        char_vocab.to_file(self.temp_file_char)

        word_vocab2 = Vocabulary(file=self.temp_file_word)
        char_vocab2 = Vocabulary(file=self.temp_file_char)
        self.assertEqual(word_vocab.itos, word_vocab2.itos)
        self.assertEqual(char_vocab.itos, char_vocab2.itos)

    def tearDown(self):
        # delete temporary vocab files
        os.remove(self.temp_file_char)
        os.remove(self.temp_file_word)


