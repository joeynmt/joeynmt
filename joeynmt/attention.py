# coding: utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionMechanism(nn.Module):

    def __init__(self):
        super(AttentionMechanism, self).__init__()

    def forward(self, *input):
        raise NotImplementedError("Implement this.")


class BahdanauAttention(AttentionMechanism):
    """
    Implements Bahdanau (MLP) attention
    """

    def __init__(self, hidden_size=1, key_size=1, query_size=1):
        """
        Creates attention mechanism.

        :param hidden_size:
        :param key_size:
        :param query_size:
        """

        super(BahdanauAttention, self).__init__()

        self.key_layer = nn.Linear(key_size, hidden_size, bias=False)
        self.query_layer = nn.Linear(query_size, hidden_size, bias=False)
        self.energy_layer = nn.Linear(hidden_size, 1, bias=False)

        self.proj_keys = None   # to store projected keys
        self.proj_query = None  # projected query

    def forward(self, query: torch.Tensor = None,
                mask: torch.Tensor = None,
                values: torch.Tensor = None):
        """
        Bahdanau additive attention forward pass.

        :param query: the item to compare with the keys/memory (e.g. decoder state)
        :param mask: mask to mask out keys position
        :param values: values (e.g. typically encoder states)
        :return: context vector, attention probabilities
        """

        assert mask is not None, "mask is required"
        assert self.proj_keys is not None,\
            "projection keys have to get pre-computed"

        # We first project the query (the decoder state).
        # The projected keys (the encoder states) were already pre-computated.
        self.compute_proj_query(query)

        # Calculate scores.
        # proj_keys: batch x src_len x hidden_size
        # proj_query: batch x 1 x hidden_size
        scores = self.energy_layer(torch.tanh(self.proj_query + self.proj_keys))
        # scores: batch x src_len x 1

        scores = scores.squeeze(2).unsqueeze(1)
        # scores: batch x 1 x time

        # mask out invalid positions by filling the masked out parts with -inf
        scores = torch.where(mask, scores, scores.new_full([1], float('-inf')))

        # turn scores to probabilities
        alphas = F.softmax(scores, dim=-1)  # batch x 1 x time

        # the context vector is the weighted sum of the values
        context = alphas @ values  # batch x 1 x value_size

        # context shape: [B, 1, 2D], alphas shape: [B, 1, M]
        return context, alphas

    def compute_proj_keys(self, keys):
        """
        Compute the projection of the keys.
        Is efficient if pre-computed before receiving individual queries.

        :param keys:
        :return:
        """
        self.proj_keys = self.key_layer(keys)

    def compute_proj_query(self, query):
        """
        Compute the projection of the query.

        :param query:
        :return:
        """
        self.proj_query = self.query_layer(query)

    def __repr__(self):
        return "BahdanauAttention"


class LuongAttention(AttentionMechanism):
    """
    Implements Luong (bilinear / multiplicative) attention
    """

    def __init__(self, hidden_size: int = 1, key_size: int = 1):
        """
        Creates attention mechanism.

        :param hidden_size:
        :param key_size:
        """

        super(LuongAttention, self).__init__()
        self.key_layer = nn.Linear(key_size, hidden_size, bias=False)
        self.proj_keys = None  # projected keys

    def forward(self, query: torch.Tensor = None,
                mask: torch.Tensor = None,
                values: torch.Tensor = None):
        """
        Luong (multiplicative / bilinear) attention forward pass.

        :param query: the item to compare with the keys/memory
        (e.g. decoder state)
        :param mask: mask to mask out keys position
        :param values: values (e.g. typically encoder states)
        :return: context vector, attention probabilities
        """

        assert self.proj_keys is not None,\
            "projection keys have to get pre-computed"
        assert mask is not None, "mask is required"

        # query:     batch x 1 x hidden_size
        # proj_keys: batch x src_len x hidden_size
        # result:    batch x 1 x src_len
        scores = query @ self.proj_keys.transpose(1, 2)

        # mask out invalid positions by filling the masked out parts with -inf
        # mask: batch x 1 x src_len
        # scores.data.masked_fill_(mask == 0, float('-inf'))
        scores = torch.where(mask, scores, scores.new_full([1], float('-inf')))

        # turn scores to probabilities
        alphas = F.softmax(scores, dim=-1)  # batch x 1 x src_len

        # the context vector is the weighted sum of the values
        context = alphas @ values  # batch x 1 x values_size

        return context, alphas

    def compute_proj_keys(self, keys):
        """
        Compute the projection of the keys.
        Is efficient if pre-computed before receiving individual queries.

        :param keys:
        :return:
        """
        self.proj_keys = self.key_layer(keys)

    def __repr__(self):
        return "LuongAttention"
