# coding: utf-8
"""
Module to represents whole models
"""
import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from joeynmt.decoders import Decoder, RecurrentDecoder, TransformerDecoder
from joeynmt.embeddings import Embeddings
from joeynmt.encoders import Encoder, RecurrentEncoder, TransformerEncoder
from joeynmt.helpers import ConfigurationError
from joeynmt.initialization import initialize_model
from joeynmt.loss import XentLoss
from joeynmt.vocabulary import Vocabulary

logger = logging.getLogger(__name__)


class Model(nn.Module):
    """
    Base Model class
    """

    # pylint: disable=too-many-instance-attributes

    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
        src_embed: Embeddings,
        trg_embed: Embeddings,
        src_vocab: Vocabulary,
        trg_vocab: Vocabulary,
    ) -> None:
        """
        Create a new encoder-decoder model

        :param encoder: encoder
        :param decoder: decoder
        :param src_embed: source embedding
        :param trg_embed: target embedding
        :param src_vocab: source vocabulary
        :param trg_vocab: target vocabulary
        """
        super().__init__()

        self.src_embed = src_embed
        self.trg_embed = trg_embed
        self.encoder = encoder
        self.decoder = decoder
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab
        self.pad_index = self.trg_vocab.pad_index
        self.bos_index = self.trg_vocab.bos_index
        self.eos_index = self.trg_vocab.eos_index
        self.unk_index = self.trg_vocab.unk_index
        self._loss_function = None  # set by the TrainManager

    @property
    def loss_function(self):
        return self._loss_function

    @loss_function.setter
    def loss_function(self, cfg: Tuple):
        loss_type, label_smoothing = cfg
        assert loss_type == "crossentropy"
        self._loss_function = XentLoss(pad_index=self.pad_index,
                                       smoothing=label_smoothing)

    def forward(self,
                return_type: str = None,
                **kwargs) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Interface for multi-gpu

        For DataParallel, We need to encapsulate all model call: `model.encode()`,
        `model.decode()`, and `model.encode_decode()` by `model.__call__()`.
        `model.__call__()` triggers model.forward() together with pre hooks and post
        hooks, which takes care of multi-gpu distribution.

        :param return_type: one of {"loss", "encode", "decode"}
        """
        if return_type is None:
            raise ValueError("Please specify return_type: "
                             "{`loss`, `encode`, `decode`}.")

        if return_type == "loss":
            assert self.loss_function is not None
            assert "trg" in kwargs and "trg_mask" in kwargs  # need trg to compute loss

            out, _, _, _ = self._encode_decode(**kwargs)

            # compute log probs
            log_probs = F.log_softmax(out, dim=-1)

            # compute batch loss
            # pylint: disable=not-callable
            batch_loss = self.loss_function(log_probs, **kwargs)

            # count correct tokens before decoding (for accuracy)
            trg_mask = kwargs["trg_mask"].squeeze(1)
            assert kwargs["trg"].size() == trg_mask.size()
            n_correct = torch.sum(
                log_probs.argmax(-1).masked_select(trg_mask).eq(
                    kwargs["trg"].masked_select(trg_mask)))

            # return batch loss
            #     = sum over all elements in batch that are not pad
            return_tuple = (batch_loss, log_probs, None, n_correct)

        elif return_type == "encode":
            kwargs["pad"] = True  # TODO: only if multi-gpu
            encoder_output, encoder_hidden = self._encode(**kwargs)

            # return encoder outputs
            return_tuple = (encoder_output, encoder_hidden, None, None)

        elif return_type == "decode":
            outputs, hidden, att_probs, att_vectors = self._decode(**kwargs)

            # return decoder outputs
            return_tuple = (outputs, hidden, att_probs, att_vectors)

        return tuple(return_tuple)

    def _encode_decode(
        self,
        src: Tensor,
        trg_input: Tensor,
        src_mask: Tensor,
        src_length: Tensor,
        trg_mask: Tensor = None,
        **kwargs,
    ) -> Tensor:
        """
        First encodes the source sentence.
        Then produces the target one word at a time.

        :param src: source input
        :param trg_input: target input
        :param src_mask: source mask
        :param src_length: length of source inputs
        :param trg_mask: target mask
        :return: decoder outputs
        """
        encoder_output, encoder_hidden = self._encode(src=src,
                                                      src_length=src_length,
                                                      src_mask=src_mask,
                                                      **kwargs)

        unroll_steps = trg_input.size(1)

        return self._decode(
            encoder_output=encoder_output,
            encoder_hidden=encoder_hidden,
            src_mask=src_mask,
            trg_input=trg_input,
            unroll_steps=unroll_steps,
            trg_mask=trg_mask,
            **kwargs,
        )

    def _encode(self, src: Tensor, src_length: Tensor, src_mask: Tensor,
                **_kwargs) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Encodes the source sentence.

        :param src:
        :param src_length:
        :param src_mask:
        :return:
            - encoder_outputs
            - hidden_concat
            - src_mask
        """
        src_input = src if self.src_embed is None else self.src_embed(src)
        return self.encoder(src_input, src_length, src_mask, **_kwargs)

    def _decode(
        self,
        encoder_output: Tensor,
        encoder_hidden: Tensor,
        src_mask: Tensor,
        trg_input: Tensor,
        unroll_steps: int,
        decoder_hidden: Tensor = None,
        att_vector: Tensor = None,
        trg_mask: Tensor = None,
        **_kwargs,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        Decode, given an encoded source sentence.

        :param encoder_output: encoder states for attention computation
        :param encoder_hidden: last encoder state for decoder initialization
        :param src_mask: source mask, 1 at valid tokens
        :param trg_input: target inputs
        :param unroll_steps: number of steps to unrol the decoder for
        :param decoder_hidden: decoder hidden state (optional)
        :param att_vector: previous attention vector (optional)
        :param trg_mask: mask for target steps
        :return: decoder outputs
            - decoder_output
            - decoder_hidden
            - att_prob
            - att_vector
        """
        return self.decoder(
            trg_embed=self.trg_embed(trg_input),
            encoder_output=encoder_output,
            encoder_hidden=encoder_hidden,
            src_mask=src_mask,
            unroll_steps=unroll_steps,
            hidden=decoder_hidden,
            prev_att_vector=att_vector,
            trg_mask=trg_mask,
            **_kwargs,
        )

    def __repr__(self) -> str:
        """
        String representation: a description of encoder, decoder and embeddings

        :return: string representation
        """
        return (f"{self.__class__.__name__}(\n"
                f"\tencoder={self.encoder},\n"
                f"\tdecoder={self.decoder},\n"
                f"\tsrc_embed={self.src_embed},\n"
                f"\ttrg_embed={self.trg_embed},\n"
                f"\tloss_function={self.loss_function})")

    def log_parameters_list(self) -> None:
        """
        Write all model parameters (name, shape) to the log.
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        n_params = sum([np.prod(p.size()) for p in model_parameters])
        logger.info("Total params: %d", n_params)
        trainable_params = [n for (n, p) in self.named_parameters() if p.requires_grad]
        logger.debug("Trainable parameters: %s", sorted(trainable_params))
        assert trainable_params


class _DataParallel(nn.DataParallel):
    """DataParallel wrapper to pass through the model attributes"""

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


def build_model(cfg: dict = None,
                src_vocab: Vocabulary = None,
                trg_vocab: Vocabulary = None) -> Model:
    """
    Build and initialize the model according to the configuration.

    :param cfg: dictionary configuration containing model specifications
    :param src_vocab: source vocabulary
    :param trg_vocab: target vocabulary
    :return: built and initialized model
    """
    logger.info("Building an encoder-decoder model...")
    enc_cfg = cfg["encoder"]
    dec_cfg = cfg["decoder"]

    src_pad_index = src_vocab.pad_index
    trg_pad_index = trg_vocab.pad_index

    src_embed = Embeddings(
        **enc_cfg["embeddings"],
        vocab_size=len(src_vocab),
        padding_idx=src_pad_index,
    )

    # this ties source and target embeddings for softmax layer tying, see further below
    if cfg.get("tied_embeddings", False):
        if src_vocab == trg_vocab:
            trg_embed = src_embed  # share embeddings for src and trg
        else:
            raise ConfigurationError(
                "Embedding cannot be tied since vocabularies differ.")
    else:
        trg_embed = Embeddings(
            **dec_cfg["embeddings"],
            vocab_size=len(trg_vocab),
            padding_idx=trg_pad_index,
        )

    # build encoder
    enc_dropout = enc_cfg.get("dropout", 0.0)
    enc_emb_dropout = enc_cfg["embeddings"].get("dropout", enc_dropout)
    if enc_cfg.get("type", "recurrent") == "transformer":
        assert enc_cfg["embeddings"]["embedding_dim"] == enc_cfg["hidden_size"], (
            "for transformer, emb_size must be "
            "the same as hidden_size")
        emb_size = src_embed.embedding_dim
        encoder = TransformerEncoder(
            **enc_cfg,
            emb_size=emb_size,
            emb_dropout=enc_emb_dropout,
            pad_index=src_pad_index,
        )
    else:
        encoder = RecurrentEncoder(
            **enc_cfg,
            emb_size=src_embed.embedding_dim,
            emb_dropout=enc_emb_dropout,
        )

    # build decoder
    dec_dropout = dec_cfg.get("dropout", 0.0)
    dec_emb_dropout = dec_cfg["embeddings"].get("dropout", dec_dropout)
    if dec_cfg.get("type", "transformer") == "transformer":
        decoder = TransformerDecoder(
            **dec_cfg,
            encoder=encoder,
            vocab_size=len(trg_vocab),
            emb_size=trg_embed.embedding_dim,
            emb_dropout=dec_emb_dropout,
        )
    else:
        decoder = RecurrentDecoder(
            **dec_cfg,
            encoder=encoder,
            vocab_size=len(trg_vocab),
            emb_size=trg_embed.embedding_dim,
            emb_dropout=dec_emb_dropout,
        )

    model = Model(
        encoder=encoder,
        decoder=decoder,
        src_embed=src_embed,
        trg_embed=trg_embed,
        src_vocab=src_vocab,
        trg_vocab=trg_vocab,
    )

    # tie softmax layer with trg embeddings
    if cfg.get("tied_softmax", False):
        if trg_embed.lut.weight.shape == model.decoder.output_layer.weight.shape:
            # (also) share trg embeddings and softmax layer:
            model.decoder.output_layer.weight = trg_embed.lut.weight
        else:
            raise ConfigurationError(
                "For tied_softmax, the decoder embedding_dim and decoder hidden_size "
                "must be the same. The decoder must be a Transformer.")

    # custom initialization of model parameters
    initialize_model(model, cfg, src_pad_index, trg_pad_index)

    # initialize embeddings from file
    enc_embed_path = enc_cfg["embeddings"].get("load_pretrained", None)
    dec_embed_path = dec_cfg["embeddings"].get("load_pretrained", None)
    if enc_embed_path and src_vocab is not None:
        logger.info("Loading pretrained src embeddings...")
        model.src_embed.load_from_file(Path(enc_embed_path), src_vocab)
    if (dec_embed_path and src_vocab is not None
            and not cfg.get("tied_embeddings", False)):
        logger.info("Loading pretrained trg embeddings...")
        model.trg_embed.load_from_file(Path(dec_embed_path), trg_vocab)

    logger.info("Enc-dec model built.")
    return model
