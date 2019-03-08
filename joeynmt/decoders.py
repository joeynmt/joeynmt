# coding: utf-8
import torch
import torch.nn as nn
from torch import Tensor
from joeynmt.attention import BahdanauAttention, LuongAttention, AttentionMechanism
from joeynmt.encoders import Encoder
from joeynmt.helpers import freeze_params


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
        Create a recurrent decoder with attention.

        :param type: rnn type, valid options: "lstm", "gru"
        :param emb_size: target embedding size
        :param hidden_size: size of the RNN
        :param encoder: encoder connected to this decoder
        :param attention: type of attention, valid options: "bahdanau", "luong"
        :param num_layers: number of recurrent layers
        :param vocab_size: target vocabulary size
        :param hidden_dropout: Is applied to the input to the attentional layer.
        :param dropout: Is applied to the input to the RNN.
        :param bridge: If True, the decoder hidden states are initialized from a
        projection of the encoder states, else they are initialized with zeros.
        :param input_feeding: Use Luong's input feeding.
        :param freeze: Freeze the parameters of the decoder during training.
        :param kwargs:
        """

        super(RecurrentDecoder, self).__init__()

        self.rnn_input_dropout = torch.nn.Dropout(p=dropout, inplace=False)
        self.type = type
        self.hidden_dropout = torch.nn.Dropout(p=hidden_dropout, inplace=False)
        self.hidden_size = hidden_size
        self.emb_size = emb_size

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

    def _check_shapes_input_forward_step(self,
                                         prev_embed: Tensor,
                                         prev_att_vector: Tensor,
                                         encoder_output: Tensor,
                                         src_mask: Tensor,
                                         hidden: Tensor):
        """
        Make sure the input shapes to `self._forward_step` are correct.
        Same inputs as `self._forward_step`.

        :param prev_embed:
        :param prev_att_vector:
        :param encoder_output:
        :param src_mask:
        :param hidden:
        :return:
        """
        assert prev_embed.shape[1:] == torch.Size([1, self.emb_size])
        assert prev_att_vector.shape[1:] == torch.Size(
            [1, self.hidden_size])
        assert prev_att_vector.shape[0] == prev_embed.shape[0]
        assert encoder_output.shape[0] == prev_embed.shape[0]
        assert len(encoder_output.shape) == 3
        assert src_mask.shape[0] == prev_embed.shape[0]
        assert src_mask.shape[1] == 1
        assert src_mask.shape[2] == encoder_output.shape[1]
        assert hidden.shape[0] == self.num_layers
        assert hidden.shape[1] == prev_embed.shape[0]
        assert hidden.shape[2] == self.hidden_size

    def _check_shapes_input_forward(self,
                                    trg_embed: Tensor,
                                    encoder_output: Tensor,
                                    encoder_hidden: Tensor,
                                    src_mask: Tensor,
                                    hidden: Tensor = None,
                                    prev_att_vector: Tensor = None):
        """
        Make sure that inputs to `self.forward` are of correct shape.
        Same input semantics as for `self.forward`.

        :param trg_embed:
        :param encoder_output:
        :param encoder_hidden:
        :param src_mask:
        :param hidden:
        :param prev_att_vector:
        :return:
        """
        assert len(encoder_output.shape) == 3
        assert len(encoder_hidden.shape) == 2
        assert encoder_hidden.shape[0] == encoder_output.shape[0]
        assert encoder_hidden.shape[-1] == encoder_output.shape[-1]
        assert src_mask.shape[1] == 1
        assert src_mask.shape[0] == encoder_output.shape[0]
        assert src_mask.shape[2] == encoder_output.shape[1]
        assert trg_embed.shape[0] == encoder_output.shape[0]
        assert trg_embed.shape[2] == self.emb_size
        if hidden is not None:
            assert hidden.shape[1] == encoder_output.shape[0]
            assert hidden.shape[2] == self.hidden_size
        if prev_att_vector is not None:
            assert prev_att_vector.shape[0] == encoder_output.shape[0]
            assert prev_att_vector.shape[2] == self.hidden_size
            assert prev_att_vector.shape[1] == 1

    def _forward_step(self,
                      prev_embed: Tensor,
                      prev_att_vector: Tensor,  # context or att vector
                      encoder_output: Tensor,
                      src_mask: Tensor,
                      hidden: Tensor):
        """
        Perform a single decoder step (1 token).

        1. `rnn_input`: concat(prev_embed, prev_att_vector [possibly empty])
        2. update RNN with `rnn_input`
        3. calculate attention and context/attention vector

        :param prev_embed: embedded previous token,
        shape (batch_size, 1, embed_size)
        :param prev_att_vector: previous attention vector,
        shape (batch_size, 1, hidden_size)
        :param encoder_output: encoder hidden states for attention context,
        shape (batch_size, src_length, encoder.output_size)
        :param src_mask: src mask, 1s for area before <eos>, 0s elsewhere
        shape (batch_size, 1, src_length)
        :param hidden: previous hidden state,
        shape (num_layers, batch_size, hidden_size)
        :return: att_vector: new attention vector with
        shape (batch_size, 1, hidden_size),
        hidden: new hidden state with shape (batch_size, 1, hidden_size),
        att_probs: attention probabilities with shape (batch_size, 1, src_len)
        """

        # shape checks
        self._check_shapes_input_forward_step(prev_embed = prev_embed,
                                              prev_att_vector = prev_att_vector,
                                              encoder_output = encoder_output,
                                              src_mask = src_mask,
                                              hidden = hidden)

        if self.input_feeding:
            # concatenate the input with the previous attention vector
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
        # batch x 1 x 2*enc_size+hidden_size
        att_vector_input = self.hidden_dropout(att_vector_input)

        att_vector = torch.tanh(self.att_vector_layer(att_vector_input))

        # output: batch x 1 x hidden_size
        return att_vector, hidden, att_probs

    def forward(self,
                trg_embed: Tensor,
                encoder_output: Tensor,
                encoder_hidden: Tensor,
                src_mask: Tensor,
                unrol_steps: int,
                hidden: Tensor = None,
                prev_att_vector: Tensor = None):
        """
         Unroll the decoder one step at a time for `unrol_steps` steps.
         For every step, the `_forward_step` function is called internally.

         During training, the target inputs (`trg_embed') are already known for
         the full sequence, so the full unrol is done.
         In this case, `hidden` and `prev_att_vector` are None.

         For inference, this function is called with one step at a time since
         embedded targets are the predictions from the previous time step.
         In this case, `hidden` and `prev_att_vector` are fed from the output
         of the previous call of this function (from the 2nd step on).

         `src_mask` is needed to mask out the areas of the encoder states that
         should not receive any attention,
         which is everything after the first <eos>.

         The `encoder_output` are the hidden states from the encoder and are
         used as context for the attention.

         The `encoder_hidden` is the last encoder hidden state that is used to
         initialize the first hidden decoder state
         (in case of `self.bridge == True`).

        :param trg_embed: emdedded target inputs,
        shape (batch_size, trg_length, embed_size)
        :param encoder_output: hidden states from the encoder,
         shape (batch_size, src_length, encoder.output_size)
        :param encoder_hidden: last state from the encoder, shape
        (batch_size x encoder.output_size)
        :param src_mask: mask for src states: 0s for padded areas,
        1s for the rest, shape (batch_size, 1, src_length)
        :param unrol_steps: number of steps to unrol the decoder RNN
        :param hidden: previous decoder hidden state,
        if not given it's initialized as in `self.init_hidden`,
        shape (num_layers, batch_size, hidden_size)
        :param prev_att_vector: previous attentional vector,
        if not given it's initialized with zeros,
        shape (batch_size, 1, hidden_size)
        :return: outputs: shape (batch_size, unrol_steps, vocab_size),
        hidden: last hidden state with shape
        (num_layers, batch_size, hidden_size),
        att_probs: attention probabilities with
        shape (batch_size, unrol_steps, src_length),
        att_vectors: attentional vectors with shape
        (batch_size, unrol_steps, hidden_size)
        """

        # shape checks
        self._check_shapes_input_forward(
            trg_embed = trg_embed,
            encoder_output = encoder_output,
            encoder_hidden = encoder_hidden,
            src_mask = src_mask,
            hidden = hidden,
            prev_att_vector = prev_att_vector)

        # initialize decoder hidden state from final encoder hidden state
        if hidden is None:
            hidden = self._init_hidden(encoder_hidden)

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

        # unroll the decoder RNN for `unrol_steps` steps
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
        # att_vectos: batch, unrol_steps, hidden_size
        att_probs = torch.cat(att_probs, dim=1)
        # att_probs: batch, unrol_steps, src_length
        outputs = self.output_layer(att_vectors)
        # outputs: batch, unrol_steps, vocab_size
        return outputs, hidden, att_probs, att_vectors

    def _init_hidden(self,
                     encoder_final: Tensor = None):
        """
        Returns the initial decoder state,
        conditioned on the final encoder state of the last encoder layer.

        In case of `self.bridge == True` and a given `encoder_final`,
        this is a projection of the encoder state.
        For LSTMs we initialize both the hidden state and the memory cell
        with the same projection of the encoder hidden state.

        Otherwise it is initialized with zeros.

        :param encoder_final: final state from the last layer of the encoder,
        shape (batch_size, encoder_hidden_size)
        :return: hidden state if GRU, (hidden state, memory cell) if LSTM,
        shape (batch_size, hidden_size)
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
