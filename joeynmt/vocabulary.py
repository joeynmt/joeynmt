# coding: utf-8

"""
Vocabulary module
"""
from collections import defaultdict, Counter
from typing import List
import numpy as np

from torchtext.data import Dataset

from joeynmt.constants import UNK_TOKEN, DEFAULT_UNK_ID, \
    EOS_TOKEN, BOS_TOKEN, PAD_TOKEN


class Vocabulary:
    """ Vocabulary represents mapping between tokens and indices. """

    def __init__(self, tokens: List[str] = None, file: str = None) -> None:
        """
        Create vocabulary from list of tokens or file.

        Special tokens are added if not already in file or list.
        File format: token with index i is in line i.

        :param tokens: list of tokens
        :param file: file to load vocabulary from
        """
        # don't rename stoi and itos since needed for torchtext
        # warning: stoi grows with unknown tokens, don't use for saving or size

        # special symbols
        self.specials = [UNK_TOKEN, PAD_TOKEN, BOS_TOKEN, EOS_TOKEN]

        self.stoi = defaultdict(DEFAULT_UNK_ID)
        self.itos = []
        if tokens is not None:
            self._from_list(tokens)
        elif file is not None:
            self._from_file(file)

    def _from_list(self, tokens: List[str] = None) -> None:
        """
        Make vocabulary from list of tokens.
        Tokens are assumed to be unique and pre-selected.
        Special symbols are added if not in list.

        :param tokens: list of tokens
        """
        self.add_tokens(tokens=self.specials+tokens)
        assert len(self.stoi) == len(self.itos)

    def _from_file(self, file: str) -> None:
        """
        Make vocabulary from contents of file.
        File format: token with index i is in line i.

        :param file: path to file where the vocabulary is loaded from
        """
        tokens = []
        with open(file, "r") as open_file:
            for line in open_file:
                tokens.append(line.strip("\n"))
        self._from_list(tokens)

    def __str__(self) -> str:
        return self.stoi.__str__()

    def to_file(self, file: str) -> None:
        """
        Save the vocabulary to a file, by writing token with index i in line i.

        :param file: path to file where the vocabulary is written
        """
        with open(file, "w") as open_file:
            for t in self.itos:
                open_file.write("{}\n".format(t))

    def add_tokens(self, tokens: List[str]) -> None:
        """
        Add list of tokens to vocabulary

        :param tokens: list of tokens to add to the vocabulary
        """
        for t in tokens:
            new_index = len(self.itos)
            # add to vocab if not already there
            if t not in self.itos:
                self.itos.append(t)
                self.stoi[t] = new_index

    def is_unk(self, token: str) -> bool:
        """
        Check whether a token is covered by the vocabulary

        :param token:
        :return: True if covered, False otherwise
        """
        return self.stoi[token] == DEFAULT_UNK_ID()

    def __len__(self) -> int:
        return len(self.itos)

    def array_to_sentence(self, array: np.array, cut_at_eos=True) -> List[str]:
        """
        Converts an array of IDs to a sentence, optionally cutting the result
        off at the end-of-sequence token.

        :param array: 1D array containing indices
        :param cut_at_eos: cut the decoded sentences at the first <eos>
        :return: list of strings (tokens)
        """
        sentence = []
        for i in array:
            s = self.itos[i]
            if cut_at_eos and s == EOS_TOKEN:
                break
            sentence.append(s)
        return sentence

    def arrays_to_sentences(self, arrays: np.array, cut_at_eos=True) \
            -> List[List[str]]:
        """
        Convert multiple arrays containing sequences of token IDs to their
        sentences, optionally cutting them off at the end-of-sequence token.

        :param arrays: 2D array containing indices
        :param cut_at_eos: cut the decoded sentences at the first <eos>
        :return: list of list of strings (tokens)
        """
        sentences = []
        for array in arrays:
            sentences.append(
                self.array_to_sentence(array=array, cut_at_eos=cut_at_eos))
        return sentences


def build_vocab(field: str, max_size: int, min_freq: int, dataset: Dataset,
                vocab_file: str = None) -> Vocabulary:
    """
    Builds vocabulary for a torchtext `field` from given`dataset` or
    `vocab_file`.

    :param field: attribute e.g. "src"
    :param max_size: maximum size of vocabulary
    :param min_freq: minimum frequency for an item to be included
    :param dataset: dataset to load data for field from
    :param vocab_file: file to store the vocabulary,
        if not None, load vocabulary from here
    :return: Vocabulary created from either `dataset` or `vocab_file`
    """

    if vocab_file is not None:
        # load it from file
        vocab = Vocabulary(file=vocab_file)
    else:
        # create newly
        def filter_min(counter: Counter, min_freq: int):
            """ Filter counter by min frequency """
            filtered_counter = Counter({t: c for t, c in counter.items()
                                        if c >= min_freq})
            return filtered_counter

        def sort_and_cut(counter: Counter, limit: int):
            """ Cut counter to most frequent,
            sorted numerically and alphabetically"""
            # sort by frequency, then alphabetically
            tokens_and_frequencies = sorted(counter.items(),
                                            key=lambda tup: tup[0])
            tokens_and_frequencies.sort(key=lambda tup: tup[1], reverse=True)
            vocab_tokens = [i[0] for i in tokens_and_frequencies[:limit]]
            return vocab_tokens

        tokens = []
        for i in dataset.examples:
            if field == "src":
                tokens.extend(i.src)
            elif field == "trg":
                tokens.extend(i.trg)

        counter = Counter(tokens)
        if min_freq > -1:
            counter = filter_min(counter, min_freq)
        vocab_tokens = sort_and_cut(counter, max_size)
        assert len(vocab_tokens) <= max_size

        vocab = Vocabulary(tokens=vocab_tokens)
        assert len(vocab) <= max_size + len(vocab.specials)
        assert vocab.itos[DEFAULT_UNK_ID()] == UNK_TOKEN

    # check for all except for UNK token whether they are OOVs
    for s in vocab.specials[1:]:
        assert not vocab.is_unk(s)

    return vocab
