# coding: utf-8
"""
Implements custom initialization
"""

import math

import torch
from torch import Tensor, nn
from torch.nn.init import _calculate_fan_in_and_fan_out

from joeynmt.helpers import ConfigurationError


def orthogonal_rnn_init_(cell: nn.RNNBase, gain: float = 1.0) -> None:
    """
    Orthogonal initialization of recurrent weights
    RNN parameters contain 3 or 4 matrices in one parameter, so we slice it.
    """
    with torch.no_grad():
        for _, hh, _, _ in cell.all_weights:
            for i in range(0, hh.size(0), cell.hidden_size):
                nn.init.orthogonal_(hh.data[i:i + cell.hidden_size], gain=gain)


def lstm_forget_gate_init_(cell: nn.RNNBase, value: float = 1.0) -> None:
    """
    Initialize LSTM forget gates with `value`.

    :param cell: LSTM cell
    :param value: initial value, default: 1
    """
    with torch.no_grad():
        for _, _, ih_b, hh_b in cell.all_weights:
            length = len(ih_b)
            ih_b.data[length // 4:length // 2].fill_(value)
            hh_b.data[length // 4:length // 2].fill_(value)


def xavier_uniform_n_(w: Tensor, gain: float = 1.0, n: int = 4) -> None:
    """
    Xavier initializer for parameters that combine multiple matrices in one
    parameter for efficiency. This is e.g. used for GRU and LSTM parameters,
    where e.g. all gates are computed at the same time by 1 big matrix.

    :param w: parameter
    :param gain: default 1
    :param n: default 4
    """
    with torch.no_grad():
        fan_in, fan_out = _calculate_fan_in_and_fan_out(w)
        assert fan_out % n == 0, "fan_out should be divisible by n"
        fan_out //= n
        std = gain * math.sqrt(2.0 / (fan_in + fan_out))
        a = math.sqrt(3.0) * std
        nn.init.uniform_(w, -a, a)


def compute_alpha_beta(num_enc_layers: int, num_dec_layers: int) -> float:
    """
    DeepNet: compute alpha/beta value suggested in https://arxiv.org/abs/2203.00555
    """
    return {
        "alpha": {
            "encoder": 0.81 * (num_enc_layers**4 * num_dec_layers)**(1 / 16),
            "decoder": (3 * num_dec_layers)**(1 / 4),
        },
        "beta": {
            "encoder": 0.87 * (num_enc_layers**4 * num_dec_layers)**(-1 / 16),
            "decoder": (12 * num_dec_layers)**(-1 / 4),
        },
    }


def initialize_model(model: nn.Module, cfg: dict, src_padding_idx: int,
                     trg_padding_idx: int) -> None:
    """
    This initializes a model based on the provided config.

    All initializer configuration is part of the `model` section of the configuration
    file. For an example, see e.g. `https://github.com/joeynmt/joeynmt/blob/main/
    configs/iwslt14_ende_spm.yaml`.

    The main initializer is set using the `initializer` key. Possible values are
    `xavier`, `uniform`, `normal` or `zeros`. (`xavier` is the default).

    When an initializer is set to `uniform`, then `init_weight` sets the range for
    the values (-init_weight, init_weight).

    When an initializer is set to `normal`, then `init_weight` sets the standard
    deviation for the weights (with mean 0).

    The word embedding initializer is set using `embed_initializer` and takes the same
    values. The default is `normal` with `embed_init_weight = 0.01`.

    Biases are initialized separately using `bias_initializer`. The default is `zeros`,
    but you can use the same initializers as the main initializer.

    Set `init_rnn_orthogonal` to True if you want RNN orthogonal initialization (for
    recurrent matrices). Default is False.

    `lstm_forget_gate` controls how the LSTM forget gate is initialized. Default is `1`.

    :param model: model to initialize
    :param cfg: the model configuration
    :param src_padding_idx: index of source padding token
    :param trg_padding_idx: index of target padding token
    """
    # pylint: disable=too-many-branches
    # defaults: xavier gain 1.0, embeddings: normal 0.01, biases: zeros, no orthogonal
    gain = float(cfg.get("init_gain", 1.0))  # for xavier
    init = cfg.get("initializer", "xavier")
    init_weight = float(cfg.get("init_weight", 0.01))

    embed_init = cfg.get("embed_initializer", "normal")
    embed_init_weight = float(cfg.get("embed_init_weight", 0.01))
    embed_gain = float(cfg.get("embed_init_gain", 1.0))  # for xavier

    bias_init = cfg.get("bias_initializer", "zeros")
    bias_init_weight = float(cfg.get("bias_init_weight", 0.01))

    if (init == "xavier_normal"
            and cfg["encoder"]["type"] == cfg["decoder"]["type"] == "transformer"):
        # apply `alpha`: weight factor for residual connection
        deepnet = {
            "xavier_normal":
            compute_alpha_beta(cfg["encoder"]["num_layers"],
                               cfg["decoder"]["num_layers"])
        }
        for layer in model.encoder.layers:
            layer.alpha = deepnet["xavier_normal"]["alpha"]["encoder"]
            layer.feed_forward.alpha = deepnet["xavier_normal"]["alpha"]["encoder"]
        for layer in model.decoder.layers:
            layer.alpha = deepnet["xavier_normal"]["alpha"]["decoder"]
            layer.feed_forward.alpha = deepnet["xavier_normal"]["alpha"]["decoder"]

    def _parse_init(s: str, scale: float, _gain: float):
        # pylint: disable=no-else-return,unnecessary-lambda
        scale = float(scale)
        assert scale > 0.0, "incorrect init_weight"
        if s.lower() == "xavier":
            return lambda p: nn.init.xavier_uniform_(p, gain=_gain)
        elif s.lower() == "xavier_normal":
            return lambda p: nn.init.xavier_normal_(p, gain=_gain)
        elif s.lower() == "uniform":
            return lambda p: nn.init.uniform_(p, a=-scale, b=scale)
        elif s.lower() == "normal":
            return lambda p: nn.init.normal_(p, mean=0.0, std=scale)
        elif s.lower() == "zeros":
            return lambda p: nn.init.zeros_(p)
        else:
            raise ConfigurationError("Unknown initializer.")

    init_fn_ = _parse_init(init, init_weight, gain)
    embed_init_fn_ = _parse_init(embed_init, embed_init_weight, embed_gain)
    bias_init_fn_ = _parse_init(bias_init, bias_init_weight, gain)

    with torch.no_grad():
        for name, p in model.named_parameters():

            if "embed" in name:
                embed_init_fn_(p)

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

                elif init == "xavier_normal" and init in deepnet:
                    # use beta value suggested in https://arxiv.org/abs/2203.00555
                    beta = 1.0
                    if ("pwff_layer" in name or "v_layer" in name
                            or "output_layer" in name):
                        if "encoder" in name:
                            beta = deepnet[init]["beta"]["encoder"]
                        elif "decoder" in name:
                            beta = deepnet[init]["beta"]["decoder"]
                    nn.init.xavier_normal_(p, gain=beta)

                else:
                    init_fn_(p)

        # zero out paddings
        model.src_embed.lut.weight.data[src_padding_idx].zero_()
        model.trg_embed.lut.weight.data[trg_padding_idx].zero_()

        orthogonal = cfg.get("init_rnn_orthogonal", False)
        lstm_forget_gate = cfg.get("lstm_forget_gate", 1.0)

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
