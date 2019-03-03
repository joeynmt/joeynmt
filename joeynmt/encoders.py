# coding: utf-8
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from joeynmt.helpers import freeze_params, clones
from joeynmt.transformer import \
    SublayerConnection, MultiHeadedAttention, PositionwiseFeedForward

"""
Various encoders
"""


# TODO make general encoder class
class Encoder(nn.Module):
    """
    Base encoder class
    """
    _output_size = 0

    @property
    def output_size(self):
        return self._output_size

    pass


class RecurrentEncoder(Encoder):
    """Encodes a sequence of word embeddings"""

    def __init__(self,
                 type: str = "gru",
                 hidden_size: int = 1,
                 emb_size: int = 1,
                 num_layers: int = 1,
                 dropout: float = 0.,
                 bidirectional: bool = True,
                 freeze: bool = False,
                 **kwargs):
        """
        Create a new recurrent encoder.

        :param type:
        :param hidden_size:
        :param emb_size:
        :param num_layers:
        :param dropout:
        :param bidirectional:
        :param freeze: freeze the parameters of the encoder during training
        :param kwargs:
        """

        super(RecurrentEncoder, self).__init__()

        self.rnn_input_dropout = torch.nn.Dropout(p=dropout, inplace=False)
        self.type = type

        rnn = nn.GRU if type == "gru" else nn.LSTM

        self.rnn = rnn(
            emb_size, hidden_size, num_layers, batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.)

        self._output_size = 2 * hidden_size if bidirectional else hidden_size

        if freeze:
            freeze_params(self)

    def forward(self, x, x_length, mask):
        """
        Applies a bidirectional RNN to sequence of embeddings x.
        The input mini-batch x needs to be sorted by src length.
        x should have dimensions [batch, time, dim].
        The masks indicates padding areas (zeros where padding).

        :param x:
        :param x_length:
        :param mask:
        :return:
        """
        # apply dropout ot the rnn input
        x = self.rnn_input_dropout(x)

        packed = pack_padded_sequence(x, x_length, batch_first=True)
        output, hidden = self.rnn(packed)

        if isinstance(hidden, tuple):
            hidden, memory_cell = hidden

        output, _ = pad_packed_sequence(output, batch_first=True)
        # hidden: dir*layers x batch x hidden
        # output: batch x max_length x directions*hidden
        batch_size = hidden.size()[1]
        # separate final hidden states by layer and direction
        hidden_layerwise = hidden.view(self.rnn.num_layers,
                                       2 if self.rnn.bidirectional else 1,
                                       batch_size, self.rnn.hidden_size)
        # final_layers: layers x directions x batch x hidden

        # concatenate the final states of the last layer for each directions
        # thanks to pack_padded_sequence final states don't include padding
        fwd_hidden_last = hidden_layerwise[-1:, 0]
        bwd_hidden_last = hidden_layerwise[-1:, 1]

        # only feed the final state of the top-most layer to the decoder
        hidden_concat = torch.cat(
            [fwd_hidden_last, bwd_hidden_last], dim=2).squeeze(0)
        # final: batch x directions*hidden
        return output, hidden_concat

    def __repr__(self):
        return "%s(%r)" % (self.__class__.__name__, self.rnn)


class TransformerEncoder(nn.Module):
    """
    Transformer Encoder
    """

    def __init__(self, hidden_size=512, ff_size=2048,
                 num_layers=8, num_heads=4, dropout=0.1, **kwargs):
        super(TransformerEncoder, self).__init__()

        # build all (num_layers) layers
        layers = []
        for _ in range(num_layers):
            layer = TransformerEncoderLayer(
                hidden_size,
                MultiHeadedAttention(num_heads, hidden_size),
                PositionwiseFeedForward(hidden_size, ff_size, dropout), dropout)
            layers.append(layer)

        self.layers = nn.ModuleList(layers)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x, lengths, mask):
        """
        Pass the input (and mask) through each layer in turn.
        :param x: 
        :param lengths: for API compatibility
        :param mask:
        :return:
        """
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x), None

    def __repr__(self):
        return "%s(num_layers=%r, num_heads=%r)" % (
            self.__class__.__name__, len(self.layers),
            self.layers[0].self_attn.h)


class TransformerEncoderLayer(nn.Module):
    """
    One Transformer encoder layer has a Multi-head attention layer plus
    a position-wise feed-forward layer.
    """

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        """

        :param x:
        :param mask:
        :return:
        """
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)
