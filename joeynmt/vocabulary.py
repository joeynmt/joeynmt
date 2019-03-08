# coding: utf-8

"""
Vocabulary module
"""

from collections import defaultdict
from typing import List

from joeynmt.constants import DEFAULT_UNK_ID


class Vocabulary:
    """ Vocabulary represents mapping between tokens and indices. """

    def __init__(self, tokens: List[str] = None, file: str = None):
        """
        Create vocabulary from list of tokens or file.

        :param tokens: list of tokens
        :param file:
        """
        # don't rename stoi and itos since needed for torchtext
        # warning: stoi grows with unknown tokens, don't use for saving or size
        self.stoi = defaultdict(DEFAULT_UNK_ID)
        self.itos = []
        if tokens is not None:
            self._from_list(tokens)
        elif file is not None:
            self._from_file(file)

    def _from_list(self, tokens: List[str] = None):
        """
        Make vocabulary from list of tokens.

        :param tokens: list of tokens
        :return:
        """
        for i, t in enumerate(tokens):
            self.stoi[t] = i
            self.itos.append(t)
        assert len(self.stoi) == len(self.itos)

    def _from_file(self, file: str):
        """
        Make vocabulary from contents of file.
        Format: token with index i is in line i.

        :param file:
        :return:
        """
        tokens = []
        with open(file, "r") as open_file:
            for line in open_file:
                tokens.append(line.strip("\n"))
        self._from_list(tokens)

    def __str__(self):
        return self.stoi.__str__()

    def to_file(self, file: str):
        """
        Save the vocabulary to a file, by writing token with index i in line i.

        :param file:
        :return:
        """
        with open(file, "w") as open_file:
            for t in self.itos:
                open_file.write("{}\n".format(t))

    def add_tokens(self, tokens: List[str]):
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

    def is_unk(self, token: str):
        """
        Check whether a token is covered by the vocabulary

        :param token:
        :return:
        """
        return self.stoi[token] == DEFAULT_UNK_ID

    def __len__(self):
        return len(self.itos)
