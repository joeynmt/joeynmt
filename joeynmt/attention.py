# coding: utf-8
"""
Attention modules
"""
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn


class AttentionMechanism(nn.Module):
    """
    Base attention class
    """

    def forward(self, *inputs):
        raise NotImplementedError("Implement this.")


class BahdanauAttention(AttentionMechanism):
    """
    Implements Bahdanau (MLP) attention

    Section A.1.2 in https://arxiv.org/abs/1409.0473.
    """

    def __init__(self, hidden_size: int = 1, key_size: int = 1, query_size: int = 1):
        """
        Creates attention mechanism.

        :param hidden_size: size of the projection for query and key
        :param key_size: size of the attention input keys
        :param query_size: size of the query
        """

        super().__init__()

        self.key_layer = nn.Linear(key_size, hidden_size, bias=False)
        self.query_layer = nn.Linear(query_size, hidden_size, bias=False)
        self.energy_layer = nn.Linear(hidden_size, 1, bias=False)

        self.proj_keys = None  # to store projected keys
        self.proj_query = None  # projected query

    def forward(self, query: Tensor, mask: Tensor,
                values: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Bahdanau MLP attention forward pass.

        :param query: the item (decoder state) to compare with the keys/memory,
            shape (batch_size, 1, decoder.hidden_size)
        :param mask: mask out keys position (0 in invalid positions, 1 else),
            shape (batch_size, 1, src_length)
        :param values: values (encoder states),
            shape (batch_size, src_length, encoder.hidden_size)
        :return:
            - context vector of shape (batch_size, 1, value_size),
            - attention probabilities of shape (batch_size, 1, src_length)
        """
        # pylint: disable=arguments-differ
        self._check_input_shapes_forward(query=query, mask=mask, values=values)

        assert mask is not None, "mask is required"
        assert self.proj_keys is not None, "projection keys have to get pre-computed"

        # We first project the query (the decoder state).
        # The projected keys (the encoder states) were already pre-computed.
        self.compute_proj_query(query)

        # Calculate scores.
        # proj_keys: batch x src_len x hidden_size
        # proj_query: batch x 1 x hidden_size
        scores = self.energy_layer(torch.tanh(self.proj_query + self.proj_keys))
        # scores: batch x src_len x 1

        scores = scores.squeeze(2).unsqueeze(1)
        # scores: batch x 1 x time

        # mask out invalid positions by filling the masked out parts with -inf
        scores = torch.where(mask > 0, scores, scores.new_full([1], -np.inf))

        # turn scores to probabilities
        alphas = F.softmax(scores, dim=-1)  # batch x 1 x time

        # the context vector is the weighted sum of the values
        context = alphas @ values  # batch x 1 x value_size

        return context, alphas

    def compute_proj_keys(self, keys: Tensor) -> None:
        """
        Compute the projection of the keys.
        Is efficient if pre-computed before receiving individual queries.

        :param keys:
        :return:
        """
        self.proj_keys = self.key_layer(keys)

    def compute_proj_query(self, query: Tensor):
        """
        Compute the projection of the query.

        :param query:
        :return:
        """
        self.proj_query = self.query_layer(query)

    def _check_input_shapes_forward(self, query: Tensor, mask: Tensor,
                                    values: Tensor) -> None:
        """
        Make sure that inputs to `self.forward` are of correct shape.
        Same input semantics as for `self.forward`.

        :param query:
        :param mask:
        :param values:
        :return:
        """
        assert query.shape[0] == values.shape[0] == mask.shape[0]
        assert query.shape[1] == 1 == mask.shape[1]
        assert query.shape[2] == self.query_layer.in_features
        assert values.shape[2] == self.key_layer.in_features
        assert mask.shape[2] == values.shape[1]

    def __repr__(self):
        return "BahdanauAttention"


class LuongAttention(AttentionMechanism):
    """
    Implements Luong (bilinear / multiplicative) attention.

    Eq. 8 ("general") in http://aclweb.org/anthology/D15-1166.
    """

    def __init__(self, hidden_size: int = 1, key_size: int = 1):
        """
        Creates attention mechanism.

        :param hidden_size: size of the key projection layer, has to be equal
            to decoder hidden size
        :param key_size: size of the attention input keys
        """

        super().__init__()
        self.key_layer = nn.Linear(in_features=key_size,
                                   out_features=hidden_size,
                                   bias=False)
        self.proj_keys = None  # projected keys

    def forward(self, query: Tensor, mask: Tensor,
                values: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Luong (multiplicative / bilinear) attention forward pass.
        Computes context vectors and attention scores for a given query and
        all masked values and returns them.

        :param query: the item (decoder state) to compare with the keys/memory,
            shape (batch_size, 1, decoder.hidden_size)
        :param mask: mask out keys position (0 in invalid positions, 1 else),
            shape (batch_size, 1, src_length)
        :param values: values (encoder states),
            shape (batch_size, src_length, encoder.hidden_size)
        :return:
            - context vector of shape (batch_size, 1, value_size),
            - attention probabilities of shape (batch_size, 1, src_length)
        """
        # pylint: disable=arguments-differ
        self._check_input_shapes_forward(query=query, mask=mask, values=values)

        assert self.proj_keys is not None, "projection keys have to get pre-computed"
        assert mask is not None, "mask is required"

        # scores: batch_size x 1 x src_length
        scores = query @ self.proj_keys.transpose(1, 2)

        # mask out invalid positions by filling the masked out parts with -inf
        scores = torch.where(mask > 0, scores, scores.new_full([1], -np.inf))

        # turn scores to probabilities
        alphas = F.softmax(scores, dim=-1)  # batch x 1 x src_len

        # the context vector is the weighted sum of the values
        context = alphas @ values  # batch x 1 x values_size

        return context, alphas

    def compute_proj_keys(self, keys: Tensor) -> None:
        """
        Compute the projection of the keys and assign them to `self.proj_keys`.
        This pre-computation is efficiently done for all keys
        before receiving individual queries.

        :param keys: shape (batch_size, src_length, encoder.hidden_size)
        """
        # proj_keys: batch x src_len x hidden_size
        self.proj_keys = self.key_layer(keys)

    def _check_input_shapes_forward(self, query: Tensor, mask: Tensor,
                                    values: Tensor) -> None:
        """
        Make sure that inputs to `self.forward` are of correct shape.
        Same input semantics as for `self.forward`.

        :param query:
        :param mask:
        :param values:
        :return:
        """
        assert query.shape[0] == values.shape[0] == mask.shape[0]
        assert query.shape[1] == 1 == mask.shape[1]
        assert query.shape[2] == self.key_layer.out_features
        assert values.shape[2] == self.key_layer.in_features
        assert mask.shape[2] == values.shape[1]

    def __repr__(self):
        return "LuongAttention"
