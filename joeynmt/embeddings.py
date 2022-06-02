# coding: utf-8
"""
Embedding module
"""

import logging
import math
from pathlib import Path
from typing import Dict

import torch
from torch import Tensor, nn

from joeynmt.helpers import freeze_params
from joeynmt.vocabulary import Vocabulary

logger = logging.getLogger(__name__)


class Embeddings(nn.Module):
    """
    Simple embeddings class
    """

    def __init__(
        self,
        embedding_dim: int = 64,
        scale: bool = False,
        vocab_size: int = 0,
        padding_idx: int = 1,
        freeze: bool = False,
        **kwargs,
    ):
        """
        Create new embeddings for the vocabulary.
        Use scaling for the Transformer.

        :param embedding_dim:
        :param scale:
        :param vocab_size:
        :param padding_idx:
        :param freeze: freeze the embeddings during training
        """
        # pylint: disable=unused-argument
        super().__init__()

        self.embedding_dim = embedding_dim
        self.scale = scale
        self.vocab_size = vocab_size
        self.lut = nn.Embedding(vocab_size, self.embedding_dim, padding_idx=padding_idx)

        if freeze:
            freeze_params(self)

    def forward(self, x: Tensor) -> Tensor:
        """
        Perform lookup for input `x` in the embedding table.

        :param x: index in the vocabulary
        :return: embedded representation for `x`
        """
        if self.scale:
            return self.lut(x) * math.sqrt(self.embedding_dim)
        return self.lut(x)

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}("
                f"embedding_dim={self.embedding_dim}, "
                f"vocab_size={self.vocab_size})")

    # from fairseq
    def load_from_file(self, embed_path: Path, vocab: Vocabulary) -> None:
        """Load pretrained embedding weights from text file.

        - First line is expected to contain vocabulary size and dimension.
          The dimension has to match the model's specified embedding size,
          the vocabulary size is used in logging only.
        - Each line should contain word and embedding weights
          separated by spaces.
        - The pretrained vocabulary items that are not part of the
          joeynmt's vocabulary will be ignored (not loaded from the file).
        - The initialization (specified in config["model"]["embed_initializer"])
          of joeynmt's vocabulary items that are not part of the
          pretrained vocabulary will be kept (not overwritten in this func).
        - This function should be called after initialization!

        Example:
            2 5
            the -0.0230 -0.0264  0.0287  0.0171  0.1403
            at -0.0395 -0.1286  0.0275  0.0254 -0.0932

        :param embed_path: embedding weights text file
        :param vocab: Vocabulary object
        """
        # pylint: disable=logging-too-many-args

        embed_dict: Dict[int, Tensor] = {}
        # parse file
        with embed_path.open("r", encoding="utf-8", errors="ignore") as f_embed:
            vocab_size, d = map(int, f_embed.readline().split())
            assert self.embedding_dim == d, "Embedding dimension doesn't match."
            for line in f_embed.readlines():
                tokens = line.rstrip().split(" ")
                if tokens[0] in vocab.specials or not vocab.is_unk(tokens[0]):
                    embed_dict[vocab.lookup(tokens[0])] = torch.FloatTensor(
                        [float(t) for t in tokens[1:]])

            logger.warning(
                "Loaded %d of %d (%%) tokens in the pre-trained WE.",
                len(embed_dict),
                vocab_size,
                len(embed_dict) / vocab_size,
            )

        # assign
        for idx, weights in embed_dict.items():
            if idx < self.vocab_size:
                assert self.embedding_dim == len(weights)
                self.lut.weight.data[idx] = weights

        logger.warning(
            "Loaded %d of %d (%%) tokens of the JoeyNMT's vocabulary.",
            len(embed_dict),
            len(vocab),
            len(embed_dict) / len(vocab),
        )
