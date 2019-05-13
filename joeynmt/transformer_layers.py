# -*- coding: utf-8 -*-

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from joeynmt.helpers import clones

# This contains special layers for the Transformer.
# Source: http://nlp.seas.harvard.edu/2018/04/03/attention.html


# pylint: disable=arguments-differ
class MultiHeadedAttention(nn.Module):
    """ Multi-headed attention module for Transformer """

    def __init__(self,
                 num_heads: int,
                 hidden_size: int,
                 dropout: float = 0.0,
                 num_linear: int = 4):
        """
        Multi-headed attention

        :param num_heads: number of attention heads
        :param hidden_size: model size
        :param dropout:
        :param num_linear: number of linear layers (default: 4)
        """
        super(MultiHeadedAttention, self).__init__()
        assert hidden_size % num_heads == 0, \
            "hidden_size must be divisible by num_heads"

        # We assume num_linear always equals key_size
        self.key_size = hidden_size // num_heads
        self.num_heads = num_heads
        self.linears = clones(nn.Linear(hidden_size, hidden_size), num_linear)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def _score(self, query: Tensor, key: Tensor, value: Tensor,
              mask: Tensor = None) -> (Tensor, Tensor):
        """
        Computes scaled dot-product attention scores.
        Applying dropout on the attention probabilities.

        :param query:
        :param key:
        :param value:
        :param mask: masks areas where there should be no attention
        :return:
            - attention vector
            - attention probabilities
        """
        scores = torch.matmul(query, key.transpose(-2, -1)) \
            / math.sqrt(query.size(-1))
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        p_attn = F.softmax(scores, dim=-1)
        p_attn = self.dropout(p_attn)
        return torch.matmul(p_attn, value), p_attn

    def forward(self, query: Tensor, key: Tensor, value: Tensor,
                mask: Tensor = None) -> Tensor:
        """
        Compute multi-headed attention.

        :param query:
        :param key:
        :param value:
        :param mask:
        :return: output vector (linear transformation of context vector)
        """

        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        batch_size = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [l(x).view(
            batch_size, -1, self.num_heads, self.key_size).transpose(1, 2)
            for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = self._score(query, key, value, mask=mask)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(batch_size, -1, self.num_heads * self.d_k)
        return self.linears[-1](x)


# pylint: disable=arguments-differ
class PositionwiseFeedForward(nn.Module):
    """
    Position-wise Feed-forward layer
    Projects to ff_size and then back down to input_size.
    """

    def __init__(self, input_size, ff_size, dropout=0.1):
        """
        Initializes position-wise feed-forward layer.
        :param input_size: dimensionality of the input.
        :param ff_size: dimensionality of intermediate representation
        :param dropout:
        """
        super(PositionwiseFeedForward, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(input_size, ff_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_size, input_size)
        )

    def forward(self, x):
        return self.layer(x)


# pylint: disable=arguments-differ
class PositionalEncoding(nn.Module):
    """
    Pre-compute position encodings (PE).
    In forward pass, this adds the position-encodings to the
    input for as many time steps as necessary.
    """

    def __init__(self, input_size: int, dropout: float = 0.0,
                 max_len: int = 5000):
        """
        :param input_size: dimensionality of the model
        :param dropout:
        :param max_len: maximum length for which to pre-compute PE
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, input_size)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, input_size, 2, dtype=torch.float32) *
            -(math.log(10000.0) / input_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Add PE to x.
        :param x:
        :return: x with position encodings added.
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


# pylint: disable=arguments-differ
class ResidualLayer(nn.Module):
    """
    Transformer residual connection
    Applies sublayer to x, layer norm, dropout, then adds x.
    """

    def __init__(self, size: int, dropout: float = 0.0):
        """
        :param size: size of input
        :param dropout: dropout probability
        """
        super(ResidualLayer, self).__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, sublayer: nn.Module) -> Tensor:
        """
        Apply layer norm on input, feed through sublayer module, apply dropout
        and add to input

        :param x: input
        :param sublayer: sublayer to build residual connection around
        :return: tensor in the same shape as input
        """
        return x + self.dropout(sublayer(self.norm(x)))


class TransformerEncoderLayer(nn.Module):
    """
    One Transformer encoder layer has a Multi-head attention layer plus
    a position-wise feed-forward layer.
    """

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        # one residual layer around attention, another one for output layer
        self.res_layer = clones(ResidualLayer(size, dropout), 2)
        self.size = size

    # pylint: disable=arguments-differ
    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        """

        :param x: layer input
        :param mask: input mask
        :return: output tensor
        """
        x = self.res_layer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.res_layer[1](x, self.feed_forward)


class TransformerDecoderLayer(nn.Module):
    """
    Transformer decoder layer.

    Consists of self-attention, source-attention, and feed-forward.
    """

    def __init__(self, size: int, self_attn: MultiHeadedAttention,
                 src_attn: MultiHeadedAttention,
                 feed_forward: PositionwiseFeedForward, dropout: float = 0.0):
        """
        Represents a single Transformer decoder layer.
        It attends to the source representation and the previous decoder states.

        :param size: layer size
        :param self_attn: self-attention module
        :param src_attn: src attention module
        :param feed_forward: feed-forward layer
        :param dropout: dropout applied between layers
        """
        super(TransformerDecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        # residual layers: one for src attention, one for trg attention,
        # one for feed-forward layer
        self.res_layer = clones(ResidualLayer(size, dropout), 3)

    # pylint: disable=arguments-differ
    def forward(self,
                x: Tensor = None,
                memory: Tensor = None,
                src_mask: Tensor = None,
                trg_mask: Tensor = None) -> Tensor:
        """
        Forward pass of a single Transformer decoder layer.

        :param x: inputs
        :param memory: source representations
        :param src_mask: source mask
        :param trg_mask: target mask (so as to not condition on future steps)
        :return: output tensor
        """
        x = self.res_layer[0](x, lambda x: self.self_attn(x, x, x, trg_mask))
        x = self.res_layer[1](x, lambda x: self.src_attn(
            x, memory, memory, src_mask))
        return self.res_layer[2](x, self.feed_forward)
