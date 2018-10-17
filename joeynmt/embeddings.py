import math
from torch import nn


class Embeddings(nn.Module):

    """
    Simple embeddings class
    """

    def __init__(self,
                 embedding_dim: int = 64,
                 scale: bool = False,
                 vocab_size: int = 0,
                 padding_idx: int = 1):
        """
        Create new embeddings for the vocabulary.
        Use scaling for the Transformer.
        :param embedding_dim:
        :param scale:
        :param vocab_size:
        :param padding_idx:
        """
        super(Embeddings, self).__init__()

        self.embedding_dim = embedding_dim
        self.scale = scale
        self.vocab_size = vocab_size
        self.lut = nn.Embedding(vocab_size, self.embedding_dim,
                                padding_idx=padding_idx)

    def forward(self, x):
        if self.scale:
            return self.lut(x) * math.sqrt(self.embedding_dim)
        return self.lut(x)

    def __repr__(self):
        return "%s(embedding_dim=%d, vocab_size=%d)" % (
            self.__class__.__name__, self.embedding_dim, self.vocab_size)
