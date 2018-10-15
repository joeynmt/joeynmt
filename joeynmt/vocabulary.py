from collections import defaultdict

from joeynmt.constants import DEFAULT_UNK_ID


class Vocabulary:
    def __init__(self, tokens=None, file=None):
        # don't rename stoi and itos since needed for torchtext
        # warning: stoi grows with unknown tokens, don't use for saving or size
        self.stoi = defaultdict(DEFAULT_UNK_ID)
        self.itos = []
        if tokens is not None:
            self._from_list(tokens)
        elif file is not None:
            self._from_file(file)

    def _from_list(self, tokens):
        for i, t in enumerate(tokens):
            self.stoi[t] = i
            self.itos.append(t)
        assert len(self.stoi) == len(self.itos)

    def _from_file(self, file):
        tokens = []
        with open(file, "r") as open_file:
            for line in open_file:
                tokens.append(line.strip("\n"))
        self._from_list(tokens)

    def __str__(self):
        return self.stoi.__str__()

    def to_file(self, file):
        with open(file, "w") as open_file:
            for i, t in enumerate(self.itos):
                open_file.write("{}\n".format(t))

    def __len__(self):
        return len(self.itos)