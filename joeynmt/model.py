# coding: utf-8
"""
Module to represents whole models
"""
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

from joeynmt.initialization import initialize_model
from joeynmt.embeddings import Embeddings
from joeynmt.encoders import Encoder, RecurrentEncoder, TransformerEncoder
from joeynmt.decoders import Decoder, RecurrentDecoder, TransformerDecoder
from joeynmt.constants import PAD_TOKEN, EOS_TOKEN, BOS_TOKEN
from joeynmt.vocabulary import Vocabulary
from joeynmt.helpers import ConfigurationError


class Model(nn.Module):
    """
    Base Model class
    """

    def __init__(self,
                 encoder: Encoder,
                 decoder: Decoder,
                 src_embed: Embeddings,
                 trg_embed: Embeddings,
                 src_vocab: Vocabulary,
                 trg_vocab: Vocabulary) -> None:
        """
        Create a new encoder-decoder model

        :param encoder: encoder
        :param decoder: decoder
        :param src_embed: source embedding
        :param trg_embed: target embedding
        :param src_vocab: source vocabulary
        :param trg_vocab: target vocabulary
        """
        super(Model, self).__init__()

        self.src_embed = src_embed
        self.trg_embed = trg_embed
        self.encoder = encoder
        self.decoder = decoder
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab
        self.bos_index = self.trg_vocab.stoi[BOS_TOKEN]
        self.pad_index = self.trg_vocab.stoi[PAD_TOKEN]
        self.eos_index = self.trg_vocab.stoi[EOS_TOKEN]
        self.loss_function = None # set by the TrainManager

    def forward(self, return_type: str = None, **kwargs) \
            -> (Tensor, Tensor, Tensor, Tensor):
        """ Interface for multi-gpu

        :param return_type: one of {"loss", "encode", "decode"}
        """
        if return_type is None:
            raise ValueError("Please specify return_type: "
                             "{`loss`, `encode`, `decode`}.")

        elif return_type == "loss":
            assert self.loss_function is not None

            out, _, _, _ = self._encode_decode(
                src=kwargs["src"],
                trg_input=kwargs["trg_input"],
                src_mask=kwargs["src_mask"],
                src_length=kwargs["src_length"],
                trg_mask=kwargs["trg_mask"])

            # compute log probs
            log_probs = F.log_softmax(out, dim=-1)

            # compute batch loss
            batch_loss = self.loss_function(log_probs, kwargs["trg"])

            # return batch loss = sum over all elements in batch that are not pad
            return batch_loss, None, None, None

        elif return_type == "encode":
            encoder_output, encoder_hidden = self._encode(
                src=kwargs["src"],
                src_length=kwargs["src_length"],
                src_mask=kwargs["src_mask"])
            # return encoder outputs
            return encoder_output, encoder_hidden, None, None

        elif return_type == "decode":
            outputs, hidden, att_probs, att_vectors = self._decode(
                trg_input=kwargs["trg_input"],
                encoder_output=kwargs["encoder_output"],
                encoder_hidden=kwargs["encoder_hidden"],
                src_mask=kwargs["src_mask"],
                unroll_steps=kwargs["unroll_steps"],
                decoder_hidden=kwargs["decoder_hidden"],
                att_vector=kwargs.get("att_vector", None),
                trg_mask=kwargs.get("trg_mask", None))

            # return decoder outputs
            return outputs, hidden, att_probs, att_vectors

    # pylint: disable=arguments-differ
    def _encode_decode(self, src: Tensor, trg_input: Tensor, src_mask: Tensor,
                       src_length: Tensor, trg_mask: Tensor = None) \
            -> (Tensor, Tensor, Tensor, Tensor):
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
                                                      src_mask=src_mask)
        unroll_steps = trg_input.size(1)
        return self._decode(encoder_output=encoder_output,
                            encoder_hidden=encoder_hidden,
                            src_mask=src_mask, trg_input=trg_input,
                            unroll_steps=unroll_steps,
                            trg_mask=trg_mask)

    def _encode(self, src: Tensor, src_length: Tensor, src_mask: Tensor) \
            -> (Tensor, Tensor):
        """
        Encodes the source sentence.

        :param src:
        :param src_length:
        :param src_mask:
        :return: encoder outputs (output, hidden_concat)
        """
        return self.encoder(self.src_embed(src), src_length, src_mask)

    def _decode(self, encoder_output: Tensor, encoder_hidden: Tensor,
                src_mask: Tensor, trg_input: Tensor,
                unroll_steps: int, decoder_hidden: Tensor = None,
                att_vector: Tensor = None, trg_mask: Tensor = None) \
            -> (Tensor, Tensor, Tensor, Tensor):
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
        :return: decoder outputs (outputs, hidden, att_probs, att_vectors)
        """
        return self.decoder(trg_embed=self.trg_embed(trg_input),
                            encoder_output=encoder_output,
                            encoder_hidden=encoder_hidden,
                            src_mask=src_mask,
                            unroll_steps=unroll_steps,
                            hidden=decoder_hidden,
                            prev_att_vector=att_vector,
                            trg_mask=trg_mask)

#    def get_loss_for_batch(self, batch: Batch, loss_function: nn.Module) \
#            -> Tensor:
#        """
#        Compute non-normalized loss and number of tokens for a batch
#
#        :param batch: batch to compute loss for
#        :param loss_function: loss function, computes for input and target
#            a scalar loss for the complete batch
#        :return: batch_loss: sum of losses over non-pad elements in the batch
#        """
#        # pylint: disable=unused-variable
#        out, hidden, att_probs, _ = self.encode_decode(
#            src=batch.src, trg_input=batch.trg_input,
#            src_mask=batch.src_mask, src_length=batch.src_length,
#            trg_mask=batch.trg_mask)
#
#        # compute log probs
#        log_probs = F.log_softmax(out, dim=-1)
#
#        # compute batch loss
#        batch_loss = loss_function(log_probs, batch.trg)
#        # return batch loss = sum over all elements in batch that are not pad
#        return batch_loss

    def __repr__(self) -> str:
        """
        String representation: a description of encoder, decoder and embeddings

        :return: string representation
        """
        return "%s(\n" \
               "\tencoder=%s,\n" \
               "\tdecoder=%s,\n" \
               "\tsrc_embed=%s,\n" \
               "\ttrg_embed=%s)" % (self.__class__.__name__, self.encoder,
                                    self.decoder, self.src_embed, self.trg_embed)


class _DataParallel(nn.DataParallel):
    """ DataParallel wrapper to pass through the model attributes """
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
    src_padding_idx = src_vocab.stoi[PAD_TOKEN]
    trg_padding_idx = trg_vocab.stoi[PAD_TOKEN]

    src_embed = Embeddings(
        **cfg["encoder"]["embeddings"], vocab_size=len(src_vocab),
        padding_idx=src_padding_idx)

    # this ties source and target embeddings
    # for softmax layer tying, see further below
    if cfg.get("tied_embeddings", False):
        if src_vocab.itos == trg_vocab.itos:
            # share embeddings for src and trg
            trg_embed = src_embed
        else:
            raise ConfigurationError(
                "Embedding cannot be tied since vocabularies differ.")
    else:
        trg_embed = Embeddings(
            **cfg["decoder"]["embeddings"], vocab_size=len(trg_vocab),
            padding_idx=trg_padding_idx)

    # build encoder
    enc_dropout = cfg["encoder"].get("dropout", 0.)
    enc_emb_dropout = cfg["encoder"]["embeddings"].get("dropout", enc_dropout)
    if cfg["encoder"].get("type", "recurrent") == "transformer":
        assert cfg["encoder"]["embeddings"]["embedding_dim"] == \
               cfg["encoder"]["hidden_size"], \
               "for transformer, emb_size must be hidden_size"

        encoder = TransformerEncoder(**cfg["encoder"],
                                     emb_size=src_embed.embedding_dim,
                                     emb_dropout=enc_emb_dropout)
    else:
        encoder = RecurrentEncoder(**cfg["encoder"],
                                   emb_size=src_embed.embedding_dim,
                                   emb_dropout=enc_emb_dropout)

    # build decoder
    dec_dropout = cfg["decoder"].get("dropout", 0.)
    dec_emb_dropout = cfg["decoder"]["embeddings"].get("dropout", dec_dropout)
    if cfg["decoder"].get("type", "recurrent") == "transformer":
        decoder = TransformerDecoder(
            **cfg["decoder"], encoder=encoder, vocab_size=len(trg_vocab),
            emb_size=trg_embed.embedding_dim, emb_dropout=dec_emb_dropout)
    else:
        decoder = RecurrentDecoder(
            **cfg["decoder"], encoder=encoder, vocab_size=len(trg_vocab),
            emb_size=trg_embed.embedding_dim, emb_dropout=dec_emb_dropout)

    model = Model(encoder=encoder, decoder=decoder,
                  src_embed=src_embed, trg_embed=trg_embed,
                  src_vocab=src_vocab, trg_vocab=trg_vocab)

    # tie softmax layer with trg embeddings
    if cfg.get("tied_softmax", False):
        if trg_embed.lut.weight.shape == \
                model.decoder.output_layer.weight.shape:
            # (also) share trg embeddings and softmax layer:
            model.decoder.output_layer.weight = trg_embed.lut.weight
        else:
            raise ConfigurationError(
                "For tied_softmax, the decoder embedding_dim and decoder "
                "hidden_size must be the same."
                "The decoder must be a Transformer.")

    # custom initialization of model parameters
    initialize_model(model, cfg, src_padding_idx, trg_padding_idx)

    return model
