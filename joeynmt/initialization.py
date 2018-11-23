# coding: utf-8
import torch
import torch.nn as nn
import math
from torch.nn.init import _calculate_fan_in_and_fan_out

def orthogonal_rnn_init_(cell, gain=1):
    """
    Orthogonal initialization of recurrent weights
    RNN parameters contain 3 or 4 matrices in one parameter, so we slice it.
    """
    with torch.no_grad():
        for _, hh, _, _ in cell.all_weights:
            for i in range(0, hh.size(0), cell.hidden_size):
                nn.init.orthogonal_(hh.data[i:i + cell.hidden_size], gain=gain)


def lstm_forget_gate_init_(cell, value=1.):
    """
    Initialize LSTM forget gates with 1.

    :param cell:
    :param value:
    :return:
    """
    with torch.no_grad():
        for _, _, ih_b, hh_b in cell.all_weights:
            l = len(ih_b)
            ih_b.data[l // 4:l // 2].fill_(value)
            hh_b.data[l // 4:l // 2].fill_(value)


def xavier_uniform_n_(w, gain=1., n=4):
    """
    Xavier initializer for parameters that combine multiple matrices in one
    parameter for efficiency. This is e.g. used for GRU and LSTM parameters,
    where e.g. all gates are computed at the same time by 1 big matrix.

    :param w:
    :param gain:
    :param n:
    :return:
    """
    with torch.no_grad():
        fan_in, fan_out = _calculate_fan_in_and_fan_out(w)
        assert fan_out % n == 0, "fan_out should be divisible by n"
        fan_out = fan_out // n
        std = gain * math.sqrt(2.0 / (fan_in + fan_out))
        a = math.sqrt(3.0) * std
        nn.init.uniform_(w, -a, a)


def initialize_model(model, cfg, src_padding_idx, trg_padding_idx):
    """
    This initializes a model based on the provided config.

    All initializer configuration is part of the `model` section of the
    configuration file.
    For an example, see e.g. [iwslt_envi_xnmt.yaml](https://github.com/joeynmt/joeynmt/blob/master/configs/iwslt_envi_xnmt.yaml#L47).

    The main initializer is set using the `initializer` key.
    Possible values are `xavier`, `uniform`, `normal` or `zeros`.
    (`xavier` is the default).

    When an initializer is set to `uniform`, then `init_weight` sets the
    range for the values (-init_weight, init_weight).

    When an initializer is set to `normal`, then `init_weight` sets the
    standard deviation for the weights (with mean 0).

    The word embedding initializer is set using `embed_initializer` and takes
    the same values. The default is `normal` with `embed_init_weight = 0.01`.

    Biases are initialized separately using `bias_initializer`.
    The default is `zeros`, but you can use the same initializers as
    the main initializer.

    Set `init_rnn_orthogonal` to True if you want RNN orthogonal initialization
    (for recurrent matrices). Default is False.

    `lstm_forget_gate` controls how the LSTM forget gate is initialized.
    Default is `1`.

    :param model: model to initialize
    :param cfg: the model configuration
    :param src_padding_idx: index of source padding token
    :param trg_padding_idx: index of target padding token
    :return:
    """

    # defaults: xavier, embeddings: normal 0.01, biases: zeros, no orthogonal
    gain = 1.
    init = cfg.get("initializer", "xavier")
    init_weight = float(cfg.get("init_weight", 0.01))
    embed_init = cfg.get("embed_initializer", "normal")
    embed_init_weight = float(cfg.get("embed_init_weight", 0.01))
    bias_init = cfg.get("bias_initializer", "zeros")
    bias_init_weight = float(cfg.get("bias_init_weight", 0.01))

    scale_src_emb = cfg["encoder"]["embeddings"].get("scale", False)
    scale_trg_emb = cfg["decoder"]["embeddings"].get("scale", False)

    def _parse_init(s, scale):
        scale = float(scale)
        assert scale > 0., "incorrect init_weight"
        if s.lower() == "xavier":
            return lambda p: nn.init.xavier_uniform_(p, gain=gain)
        elif s.lower() == "uniform":
            return lambda p: nn.init.uniform_(p, a=-scale, b=scale)
        elif s.lower() == "normal":
            return lambda p: nn.init.normal_(p, mean=0., std=scale)
        elif s.lower() == "zeros":
            return lambda p: nn.init.zeros_(p)
        else:
            raise ValueError("unknown initializer")

    init_fn_ = _parse_init(init, init_weight)
    embed_init_fn_ = _parse_init(embed_init, embed_init_weight)
    bias_init_fn_ = _parse_init(bias_init, bias_init_weight)

    with torch.no_grad():
        for name, p in model.named_parameters():

            if "embed" in name:
                embed_init_fn_(p)

                # scale embeddings if xavier (more variance)
                if embed_init == "xavier":
                    if ("src" in name and scale_src_emb) or \
                            ("trg" in name and scale_trg_emb):
                        dim = p.size(1)
                        p.data *= math.sqrt(dim)

            elif "bias" in name:
                bias_init_fn_(p)

            elif len(p.size()) > 1:

                # RNNs combine multiple matrices is one, which messes up
                # xavier initialization
                if init == "xavier" and "rnn" in name:
                    n = 1
                    if "encoder" in name:
                        n = 4 if isinstance(model.encoder.rnn, nn.LSTM) else 3
                    elif "decoder" in name:
                        n = 4 if isinstance(model.decoder.rnn, nn.LSTM) else 3
                    xavier_uniform_n_(p.data, gain=gain, n=n)
                else:
                    init_fn_(p)

            else:
                raise RuntimeError("unexpected init situation")

        # zero out paddings
        model.src_embed.lut.weight.data[src_padding_idx].zero_()
        model.trg_embed.lut.weight.data[trg_padding_idx].zero_()

        orthogonal = cfg.get("init_rnn_orthogonal", False)
        lstm_forget_gate = cfg.get("lstm_forget_gate", 1.)

        # encoder rnn orthogonal initialization & LSTM forget gate
        if hasattr(model.encoder, "rnn"):

            if orthogonal:
                orthogonal_rnn_init_(model.encoder.rnn)

            if isinstance(model.encoder.rnn, nn.LSTM):
                lstm_forget_gate_init_(model.encoder.rnn, lstm_forget_gate)

        # decoder rnn orthogonal initialization & LSTM forget gate
        if hasattr(model.decoder, "rnn"):

            if orthogonal:
                orthogonal_rnn_init_(model.decoder.rnn)

            if isinstance(model.decoder.rnn, nn.LSTM):
                lstm_forget_gate_init_(model.decoder.rnn, lstm_forget_gate)
