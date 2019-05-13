# coding: utf-8

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from joeynmt.helpers import freeze_params
from joeynmt.transformer_layers import TransformerEncoderLayer, \
    MultiHeadedAttention, PositionwiseFeedForward, PositionalEncoding


#pylint: disable=abstract-method
class Encoder(nn.Module):
    """
    Base encoder class
    """
    @property
    def output_size(self):
        """
        Return the output size

        :return:
        """
        return self._output_size


class RecurrentEncoder(Encoder):
    """Encodes a sequence of word embeddings"""

    #pylint: disable=unused-argument
    def __init__(self,
                 rnn_type: str = "gru",
                 hidden_size: int = 1,
                 emb_size: int = 1,
                 num_layers: int = 1,
                 dropout: float = 0.,
                 bidirectional: bool = True,
                 freeze: bool = False,
                 **kwargs) -> None:
        """
        Create a new recurrent encoder.

        :param rnn_type:
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
        self.type = rnn_type
        self.emb_size = emb_size

        rnn = nn.GRU if rnn_type == "gru" else nn.LSTM

        self.rnn = rnn(
            emb_size, hidden_size, num_layers, batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.)

        self._output_size = 2 * hidden_size if bidirectional else hidden_size

        if freeze:
            freeze_params(self)

    #pylint: disable=invalid-name, unused-argument
    def _check_shapes_input_forward(self, embed_src: Tensor, src_length: Tensor,
                                    mask: Tensor) -> None:
        """
        Make sure the shape of the inputs to `self.forward` are correct.
        Same input semantics as `self.forward`.

        :param embed_src: embedded source tokens
        :param src_length: source length
        :param mask: source mask
        """
        assert embed_src.shape[0] == src_length.shape[0]
        assert embed_src.shape[2] == self.emb_size
       # assert mask.shape == embed_src.shape
        assert len(src_length.shape) == 1

    #pylint: disable=arguments-differ
    def forward(self, embed_src: Tensor, src_length: Tensor, mask: Tensor) \
            -> (Tensor, Tensor):
        """
        Applies a bidirectional RNN to sequence of embeddings x.
        The input mini-batch x needs to be sorted by src length.
        x and mask should have the same dimensions [batch, time, dim].

        :param embed_src: embedded src inputs,
            shape (batch_size, src_len, embed_size)
        :param src_length: length of src inputs
            (counting tokens before padding), shape (batch_size)
        :param mask: indicates padding areas (zeros where padding), shape
            (batch_size, src_len, embed_size)
        :return:
            - output: hidden states with
                shape (batch_size, max_length, directions*hidden),
            - hidden_concat: last hidden state with
                shape (batch_size, directions*hidden)
        """
        self._check_shapes_input_forward(embed_src=embed_src,
                                         src_length=src_length,
                                         mask=mask)

        # apply dropout ot the rnn input
        embed_src = self.rnn_input_dropout(embed_src)

        packed = pack_padded_sequence(embed_src, src_length, batch_first=True)
        output, hidden = self.rnn(packed)

        #pylint: disable=unused-variable
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
        #pylint: disable=no-member
        hidden_concat = torch.cat(
            [fwd_hidden_last, bwd_hidden_last], dim=2).squeeze(0)
        # final: batch x directions*hidden
        return output, hidden_concat

    def __repr__(self):
        return "%s(%r)" % (self.__class__.__name__, self.rnn)


class TransformerEncoder(Encoder):
    """
    Transformer Encoder
    """

    #pylint: disable=unused-argument
    def __init__(self,
                 hidden_size: int = 512,
                 ff_size: int = 2048,
                 num_layers: int = 8,
                 num_heads: int = 4,
                 dropout: float = 0.1,
                 freeze: bool = False,
                 **kwargs):
        """
        Initializes the Transformer.
        :param hidden_size: hidden size and size of embeddings
        :param ff_size: position-wise feed-forward layer size.
          (Typically this is 2*hidden_size.)
        :param num_layers: number of layers
        :param num_heads: number of heads for multi-headed attention
        :param dropout: dropout probability
        :param freeze: freeze the parameters of the encoder during training
        :param kwargs:
        """
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
        self.pe = PositionalEncoding(hidden_size, dropout=dropout)
        self._output_size = hidden_size

        if freeze:
            freeze_params(self)

    #pylint: disable=arguments-differ
    def forward(self,
                embed_src: Tensor,
                src_length: Tensor,
                mask: Tensor) -> (Tensor, Tensor):
        """
        Pass the input (and mask) through each layer in turn.
        Applies a Transformer encoder to sequence of embeddings x.
        The input mini-batch x needs to be sorted by src length.
        x and mask should have the same dimensions [batch, time, dim].

        :param embed_src: embedded src inputs,
            shape (batch_size, src_len, embed_size)
        :param src_length: length of src inputs
            (counting tokens before padding), shape (batch_size)
        :param mask: indicates padding areas (zeros where padding), shape
            (batch_size, src_len, embed_size)
        :return:
            - output: hidden states with
                shape (batch_size, max_length, directions*hidden),
            - hidden_concat: last hidden state with
                shape (batch_size, directions*hidden)
        """
        x = self.pe(embed_src)  # add position encoding to word embeddings

        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x), None

    def __repr__(self):
        return "%s(num_layers=%r, num_heads=%r)" % (
            self.__class__.__name__, len(self.layers),
            self.layers[0].self_attn.h)
