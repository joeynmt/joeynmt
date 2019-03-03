# coding: utf-8
import torch
import torch.nn as nn
from torch import Tensor
from joeynmt.attention import BahdanauAttention, LuongAttention, AttentionMechanism
from joeynmt.encoders import Encoder
from joeynmt.helpers import freeze_params, clones
from joeynmt.transformer import SublayerConnection, MultiHeadedAttention, \
    PositionwiseFeedForward, subsequent_mask


# TODO make general decoder class
class Decoder(nn.Module):
    pass


class RecurrentDecoder(Decoder):
    """A conditional RNN decoder with attention."""

    def __init__(self,
                 type: str = "gru",
                 emb_size: int = 0,
                 hidden_size: int = 0,
                 encoder: Encoder = None,
                 attention: str = "bahdanau",
                 num_layers: int = 0,
                 vocab_size: int = 0,
                 dropout: float = 0.,
                 hidden_dropout: float = 0.,
                 bridge: bool = False,
                 input_feeding: bool = True,
                 freeze: bool = False,
                 **kwargs):
        """
        Create a recurrent decoder.
        If `bridge` is True, the decoder hidden states are initialized from a
        projection of the encoder states, else they are initialized with zeros.

        :param type:
        :param emb_size:
        :param hidden_size:
        :param encoder:
        :param attention:
        :param num_layers:
        :param vocab_size:
        :param dropout:
        :param hidden_dropout:
        :param bridge:
        :param input_feeding:
        :param freeze: freeze the parameters of the decoder during training
        :param kwargs:
        """

        super(RecurrentDecoder, self).__init__()

        self.rnn_input_dropout = torch.nn.Dropout(p=dropout, inplace=False)
        self.type = type
        self.hidden_dropout = torch.nn.Dropout(p=hidden_dropout, inplace=False)
        self.hidden_size = hidden_size

        rnn = nn.GRU if type == "gru" else nn.LSTM

        self.input_feeding = input_feeding
        if self.input_feeding: # Luong-style
            # combine embedded prev word +attention vector before feeding to rnn
            self.rnn_input_size = emb_size + hidden_size
        else:
            # just feed prev word embedding
            self.rnn_input_size = emb_size

        # the decoder RNN
        self.rnn = rnn(self.rnn_input_size, hidden_size, num_layers,
                       batch_first=True,
                       dropout=dropout if num_layers > 1 else 0.)

        # combine output with context vector before output layer (Luong-style)
        self.att_vector_layer = nn.Linear(
            hidden_size + encoder.output_size, hidden_size, bias=True)

        self.output_layer = nn.Linear(hidden_size, vocab_size, bias=False)
        self.output_size = vocab_size

        if attention == "bahdanau":
            self.attention = BahdanauAttention(hidden_size=hidden_size,
                                               key_size=encoder.output_size,
                                               query_size=hidden_size)
        elif attention == "luong":
            self.attention = LuongAttention(hidden_size=hidden_size,
                                            key_size=encoder.output_size)
        else:
            raise ValueError("Unknown attention mechanism: %s" % attention)

        self.num_layers = num_layers
        self.hidden_size = hidden_size

        # to initialize from the final encoder state of last layer
        self.bridge = bridge
        if self.bridge:
            self.bridge_layer = nn.Linear(
                encoder.output_size, hidden_size, bias=True)

        if freeze:
            freeze_params(self)

    def _forward_step(self,
                      prev_embed: Tensor = None,
                      prev_att_vector: Tensor = None,  # context or att vector
                      encoder_output: Tensor = None,
                      src_mask: Tensor = None,
                      hidden: Tensor = None):
        """
        Perform a single decoder step (1 word)

        :param prev_embed:
        :param prev_att_vector:
        :param encoder_output:
        :param src_mask:
        :param hidden:
        :return:
        """

        # loop:
        # 1. rnn input = concat(prev_embed, prev_output [possibly empty])
        # 2. update RNN with rnn_input
        # 3. calculate attention and context/attention vector
        # 4. repeat

        # update rnn hidden state
        if self.input_feeding:
            rnn_input = torch.cat([prev_embed, prev_att_vector], dim=2)
        else:
            rnn_input = prev_embed

        rnn_input = self.rnn_input_dropout(rnn_input)

        # rnn_input: batch x 1 x emb+2*enc_size
        _, hidden = self.rnn(rnn_input, hidden)

        # use new (top) decoder layer as attention query
        if isinstance(hidden, tuple):
            query = hidden[0][-1].unsqueeze(1)
        else:
            query = hidden[-1].unsqueeze(1)  # [#layers, B, D] -> [B, 1, D]

        # compute context vector using attention mechanism
        # only use last layer for attention mechanism
        # key projections are pre-computed
        context, att_probs = self.attention(
            query=query, values=encoder_output, mask=src_mask)

        # return attention vector (Luong)
        # combine context with decoder hidden state before prediction
        att_vector_input = torch.cat([query, context], dim=2)
        att_vector_input = self.hidden_dropout(att_vector_input)

        # batch x 1 x 2*enc_size+hidden_size
        att_vector = torch.tanh(self.att_vector_layer(att_vector_input))

        # output: batch x 1 x dec_size
        return att_vector, hidden, att_probs

    def forward(self, trg_embed, encoder_output, encoder_hidden,
                src_mask, unrol_steps, hidden=None, prev_att_vector=None,
                **kwargs):
        """
         Unroll the decoder one step at a time for `unrol_steps` steps.

        :param trg_embed:
        :param encoder_output:
        :param encoder_hidden:
        :param src_mask:
        :param unrol_steps:
        :param hidden:
        :param prev_att_vector:
        :return:
        """

        # initialize decoder hidden state from final encoder hidden state
        if hidden is None:
            hidden = self.init_hidden(encoder_hidden)

        # pre-compute projected encoder outputs
        # (the "keys" for the attention mechanism)
        # this is only done for efficiency
        if hasattr(self.attention, "compute_proj_keys"):
            self.attention.compute_proj_keys(encoder_output)

        # here we store all intermediate attention vectors (used for prediction)
        att_vectors = []
        att_probs = []

        batch_size = encoder_output.size(0)

        if prev_att_vector is None:
            with torch.no_grad():
                prev_att_vector = encoder_output.new_zeros(
                    [batch_size, 1, self.hidden_size])

        # unroll the decoder RN N for max_len steps
        for i in range(unrol_steps):
            prev_embed = trg_embed[:, i].unsqueeze(1)  # batch, 1, emb
            prev_att_vector, hidden, att_prob = self._forward_step(
                prev_embed=prev_embed,
                prev_att_vector=prev_att_vector,
                encoder_output=encoder_output,
                src_mask=src_mask,
                hidden=hidden)
            att_vectors.append(prev_att_vector)
            att_probs.append(att_prob)

        att_vectors = torch.cat(att_vectors, dim=1)
        att_probs = torch.cat(att_probs, dim=1)
        # att_probs: batch, max_len, src_length
        outputs = self.output_layer(att_vectors)
        # outputs: batch, max_len, vocab_size
        return outputs, hidden, att_probs, att_vectors

    def init_hidden(self, encoder_final):
        """
        Returns the initial decoder state,
        conditioned on the final encoder state of the last encoder layer.

        :param encoder_final:
        :return:
        """
        batch_size = encoder_final.size(0)

        # for multiple layers: is the same for all layers
        if self.bridge and encoder_final is not None:
            h = torch.tanh(
                self.bridge_layer(encoder_final)).unsqueeze(0).repeat(
                self.num_layers, 1, 1)  # num_layers x batch_size x hidden_size

        else:  # initialize with zeros
            with torch.no_grad():
                h = encoder_final.new_zeros(self.num_layers, batch_size,
                                            self.hidden_size)

        return (h, h) if isinstance(self.rnn, nn.LSTM) else h

    def __repr__(self):
        return "RecurrentDecoder(rnn=%r, attention=%r)" % (
            self.rnn, self.attention)


class TransformerDecoder(nn.Module):
    """
    A transformer decoder with N masked layers.
    Decoder layers are masked so that an attention head cannot see the future.
    """

    def __init__(self, num_layers=4, num_heads=8,
                 hidden_size=512, ff_size=2048, dropout=0.1,
                 vocab_size=1,
                 **kwargs):
        super(TransformerDecoder, self).__init__()

        # build all (num_layers) layers
        layers = []
        for _ in range(num_layers):
            layer = TransformerDecoderLayer(
                hidden_size,
                MultiHeadedAttention(num_heads, hidden_size),
                MultiHeadedAttention(num_heads, hidden_size),
                PositionwiseFeedForward(hidden_size, ff_size, dropout), dropout)
            layers.append(layer)

        self.layers = nn.ModuleList(layers)
        self.norm = nn.LayerNorm(hidden_size)
        self.output_layer = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self,
                trg_embed = None,
                encoder_output = None,
                encoder_hidden = None,
                src_mask = None,
                unrol_steps = None,
                hidden = None,
                trg_mask = None,
                **kwargs):
        """
        :param trg_embed: target embeddings
        :param encoder_output: source representations
        :param encoder_hidden: unused
        :param src_mask:
        :param unrol_steps: unused
        :param hidden: unused
        :param trg_mask:
        :param kwargs:
        :return:
        """
        assert trg_mask is not None, "trg_mask required for Transformer"

        x = trg_embed
        trg_mask = trg_mask.unsqueeze(-2) & subsequent_mask(trg_embed.size(1)).type_as(trg_mask)

        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, trg_mask)

        x = self.norm(x)
        output = self.output_layer(x)

        return output, x, None, None

    def __repr__(self):
        return "%s(num_layers=%r, num_heads=%r)" % (
            self.__class__.__name__, len(self.layers),
            self.layers[0].self_attn.h)


class TransformerDecoderLayer(nn.Module):
    """
    Transformer decoder layer.
    Consists of self-attention, source-attention, and feed-forward.
    """

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(TransformerDecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)
