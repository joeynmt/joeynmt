# coding: utf-8
"""
Search module
"""
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from joeynmt.batch import Batch
from joeynmt.decoders import RecurrentDecoder, TransformerDecoder
from joeynmt.helpers import tile
from joeynmt.model import Model


__all__ = ["greedy", "transformer_greedy", "beam_search", "run_batch"]


def greedy(
    src_mask: Tensor,
    max_output_length: int,
    model: Model,
    encoder_output: Tensor,
    encoder_hidden: Tensor,
    generate_unk: bool = False,
) -> Tuple[np.array, np.array]:
    """
    Greedy decoding. Select the token word highest probability at each time step.
    This function is a wrapper that calls recurrent_greedy for recurrent decoders and
    transformer_greedy for transformer decoders.

    :param src_mask: mask for source inputs, 0 for positions after </s>
    :param max_output_length: maximum length for the hypotheses
    :param model: model to use for greedy decoding
    :param encoder_output: encoder hidden states for attention
    :param encoder_hidden: encoder last state for decoder initialization
    :param generate_unk: whether to generate UNK token. if folse,
            the probability of UNK token will artificially be set to zero.
    :return:
        - stacked_output: output hypotheses (2d array of indices),
        - stacked_attention_scores: attention scores (3d array)
    """
    # pylint: disable=no-else-return,unused-argument
    if isinstance(model.decoder, TransformerDecoder):
        # Transformer greedy decoding
        return transformer_greedy(
            src_mask, max_output_length, model, encoder_output, encoder_hidden
        )
    elif isinstance(model.decoder, RecurrentDecoder):
        return recurrent_greedy(
            src_mask, max_output_length, model, encoder_output, encoder_hidden
        )
    else:
        raise NotImplementedError(
            f"model.decoder({model.decoder.__class__.__name__}) not supported."
        )


def recurrent_greedy(
    src_mask: Tensor,
    max_output_length: int,
    model: Model,
    encoder_output: Tensor,
    encoder_hidden: Tensor,
    generate_unk: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Greedy decoding: in each step, choose the word that gets highest score.
    Version for recurrent decoder.

    :param src_mask: mask for source inputs, 0 for positions after </s>
    :param max_output_length: maximum length for the hypotheses
    :param model: model to use for greedy decoding
    :param encoder_output: encoder hidden states for attention
    :param encoder_hidden: encoder last state for decoder initialization
    :param generate_unk: whether to generate UNK token. if false,
            the probability of UNK token will artificially be set to zero.
    :return:
        - stacked_output: output hypotheses (2d array of indices),
        - stacked_attention_scores: attention scores (3d array)
    """
    bos_index = model.bos_index
    eos_index = model.eos_index
    unk_index = model.unk_index
    batch_size = src_mask.size(0)
    prev_y = src_mask.new_full(
        size=[batch_size, 1], fill_value=bos_index, dtype=torch.long
    )
    output = []
    attention_scores = []
    hidden = None
    prev_att_vector = None
    finished = src_mask.new_zeros((batch_size, 1)).byte()

    for _ in range(max_output_length):
        # decode one single step
        with torch.no_grad():
            logits, hidden, att_probs, prev_att_vector = model(
                return_type="decode",
                trg_input=prev_y,
                encoder_output=encoder_output,
                encoder_hidden=encoder_hidden,
                src_mask=src_mask,
                unroll_steps=1,
                decoder_hidden=hidden,
                att_vector=prev_att_vector,
            )
            # logits: batch x time=1 x vocab (logits)

        # greedy decoding: choose arg max over vocabulary in each step
        if not generate_unk:
            logits[:, :, unk_index] = float("-inf")
        next_word = torch.argmax(logits, dim=-1)  # batch x time=1
        output.append(next_word.squeeze(1).detach().cpu().numpy())
        prev_y = next_word
        attention_scores.append(att_probs.squeeze(1).detach().cpu().numpy())
        # batch, max_src_length

        # check if previous symbol was <eos>
        is_eos = torch.eq(next_word, eos_index)
        finished += is_eos
        # stop predicting if <eos> reached for all elements in batch
        if (finished >= 1).sum() == batch_size:
            break

    stacked_output = np.stack(output, axis=1)  # batch, time
    stacked_attention_scores = np.stack(attention_scores, axis=1)
    return stacked_output, stacked_attention_scores


def transformer_greedy(
    src_mask: Tensor,
    max_output_length: int,
    model: Model,
    encoder_output: Tensor,
    encoder_hidden: Tensor,
    generate_unk: bool = False,
) -> Tuple[np.ndarray, None]:
    """
    Special greedy function for transformer, since it works differently.
    The transformer remembers all previous states and attends to them.

    :param src_mask: mask for source inputs, 0 for positions after </s>
    :param max_output_length: maximum length for the hypotheses
    :param model: model to use for greedy decoding
    :param encoder_output: encoder hidden states for attention
    :param encoder_hidden: encoder final state (unused in Transformer)
    :param generate_unk: whether to generate UNK token. if folse, the probability of UNK
        token will artificially be set to zero.
    :return:
        - stacked_output: output hypotheses (2d array of indices),
        - stacked_attention_scores: attention scores (3d array)
    """
    # pylint: disable=unused-argument
    bos_index = model.bos_index
    eos_index = model.eos_index
    unk_index = model.unk_index
    batch_size = src_mask.size(0)

    # start with BOS-symbol for each sentence in the batch
    ys = encoder_output.new_full([batch_size, 1], bos_index, dtype=torch.long)

    # a subsequent mask is intersected with this in decoder forward pass
    trg_mask = src_mask.new_ones([1, 1, 1])
    if isinstance(model, torch.nn.DataParallel):
        trg_mask = torch.stack([src_mask.new_ones([1, 1]) for _ in model.device_ids])

    finished = src_mask.new_zeros(batch_size).byte()

    for _ in range(max_output_length):
        with torch.no_grad():
            nll_logits, _, _, _ = model(
                return_type="decode",
                trg_input=ys,  # model.trg_embed(ys) # embed the previous tokens
                encoder_output=encoder_output,
                encoder_hidden=None,
                src_mask=src_mask,
                unroll_steps=None,
                decoder_hidden=None,
                trg_mask=trg_mask,
            )
            logits = nll_logits[:, -1]
            if not generate_unk:
                logits[:, unk_index] = float("-inf")
            _, next_word = torch.max(logits, dim=1)
            next_word = next_word.data
            ys = torch.cat([ys, next_word.unsqueeze(-1)], dim=1)

        # check if previous symbol was <eos>
        is_eos = torch.eq(next_word, eos_index)
        finished += is_eos
        # stop predicting if <eos> reached for all elements in batch
        if (finished >= 1).sum() == batch_size:
            break

    ys = ys[:, 1:]  # remove BOS-symbol
    return ys.detach().cpu().numpy(), None


def beam_search(
    model: Model,
    size: int,
    encoder_output: Tensor,
    encoder_hidden: Tensor,
    src_mask: Tensor,
    max_output_length: int,
    alpha: float,
    n_best: int = 1,
    generate_unk=False,
) -> Tuple[np.ndarray, None]:
    """
    Beam search with size k. In each decoding step, find the k most likely partial
    hypotheses. Inspired by OpenNMT-py, adapted for Transformer.

    :param model:
    :param size: size of the beam
    :param encoder_output:
    :param encoder_hidden:
    :param src_mask:
    :param max_output_length:
    :param alpha: `alpha` factor for length penalty
    :param n_best: return this many hypotheses, <= beam (currently only 1)
    :param generate_unk: whether to generate UNK token. if folse,
            the probability of UNK token will artificially be set to zero.
    :return:
        - stacked_output: output hypotheses (2d array of indices),
        - stacked_attention_scores: attention scores (3d array)
    """
    # pylint: disable=too-many-statements,too-many-branches
    assert size > 0, "Beam size must be >0."
    assert n_best <= size, f"Can only return {size} best hypotheses."

    # init
    bos_index = model.bos_index
    eos_index = model.eos_index
    pad_index = model.pad_index
    unk_index = model.unk_index
    trg_vocab_size = model.decoder.output_size
    device = encoder_output.device
    transformer = isinstance(model.decoder, TransformerDecoder)
    batch_size = src_mask.size(0)
    att_vectors = None  # not used for Transformer
    hidden = None  # not used for Transformer
    trg_mask = None  # not used for RNN

    # Recurrent models only: initialize RNN hidden state
    if not transformer:
        # pylint: disable=protected-access
        # tile encoder states and decoder initial states beam_size times
        hidden = model.decoder._init_hidden(encoder_hidden)
        hidden = tile(hidden, size, dim=1)  # layers x batch*k x dec_hidden_size
        # DataParallel splits batch along the 0th dim.
        # Place back the batch_size to the 1st dim here.
        if isinstance(hidden, tuple):
            h, c = hidden
            hidden = (h.permute(1, 0, 2), c.permute(1, 0, 2))
        else:
            hidden = hidden.permute(1, 0, 2)
            # batch*k x layers x dec_hidden_size

    encoder_output = tile(
        encoder_output.contiguous(), size, dim=0
    )  # batch*k x src_len x enc_hidden_size
    src_mask = tile(src_mask, size, dim=0)  # batch*k x 1 x src_len

    # Transformer only: create target mask
    if transformer:
        trg_mask = src_mask.new_ones([1, 1, 1])
        if isinstance(model, torch.nn.DataParallel):
            trg_mask = torch.stack(
                [src_mask.new_ones([1, 1]) for _ in model.device_ids]
            )

    # numbering elements in the batch
    batch_offset = torch.arange(batch_size, dtype=torch.long, device=device)

    # numbering elements in the extended batch, i.e. k copies of each batch element
    beam_offset = torch.arange(
        0, batch_size * size, step=size, dtype=torch.long, device=device
    )

    # keeps track of the top beam size hypotheses to expand for each element in the
    # batch to be further decoded (that are still "alive")
    alive_seq = torch.full(
        [batch_size * size, 1], bos_index, dtype=torch.long, device=device
    )

    # Give full probability to the first beam on the first step.
    topk_log_probs = torch.zeros(batch_size, size, device=device)
    topk_log_probs[:, 1:] = float("-inf")

    # Structure that holds finished hypotheses.
    hypotheses = [[] for _ in range(batch_size)]

    results = {
        "predictions": [[] for _ in range(batch_size)],
        "scores": [[] for _ in range(batch_size)],
        "gold_score": [0] * batch_size,
    }
    is_finished = torch.full(
        [batch_size, size],
        False,
        dtype=torch.bool,
        device=device,
    )
    for step in range(max_output_length):
        # print(transformer, step, alive_seq, results)
        if transformer:
            # This decides which part of the predicted sentence we feed to the decoder
            # to make the next prediction.
            # For Transformer, we feed the complete predicted sentence so far.
            decoder_input = alive_seq  # complete prediction so far

            # expand current hypotheses
            # decode one single step
            # logits: logits for final softmax
            with torch.no_grad():
                nll_logits, _, _, _ = model(
                    return_type="decode",
                    encoder_output=encoder_output,
                    encoder_hidden=None,  # only for initializing decoder_hidden
                    src_mask=src_mask,
                    trg_input=decoder_input,  # trg_embed = embed(decoder_input)
                    decoder_hidden=None,  # don't need to keep it for transformer
                    att_vector=None,  # don't need to keep it for transformer
                    unroll_steps=1,
                    trg_mask=trg_mask,  # subsequent mask for Transformer only
                )

            # For the Transformer we made predictions for all time steps up to this
            # point, so we only want to know about the last time step.
            logits = nll_logits[:, -1]
            hidden = None
        else:
            # For Recurrent models, only feed the previous trg word prediction
            decoder_input = alive_seq[:, -1].view(-1, 1)  # only the last word

            with torch.no_grad():
                # pylint: disable=unused-variable
                logits, hidden, att_scores, att_vectors = model(
                    return_type="decode",
                    encoder_output=encoder_output,
                    encoder_hidden=None,  # only for initializing decoder_hidden
                    src_mask=src_mask,
                    trg_input=decoder_input,  # trg_embed = embed(decoder_input)
                    decoder_hidden=hidden,
                    att_vector=att_vectors,
                    unroll_steps=1,
                    trg_mask=None,  # subsequent mask for Transformer only
                )

        # batch*k x trg_vocab
        log_probs = F.log_softmax(logits, dim=-1).squeeze(1)
        if not generate_unk:
            log_probs[:, unk_index] = float("-inf")

        # multiply probs by the beam probability (=add logprobs)
        log_probs += topk_log_probs.view(-1).unsqueeze(1)
        curr_scores = log_probs.clone()

        # compute length penalty
        if alpha > 0:
            length_penalty = ((5.0 + (step + 1)) / 6.0) ** alpha
            curr_scores /= length_penalty

        # flatten log_probs into a list of possibilities
        curr_scores = curr_scores.reshape(-1, size * trg_vocab_size)

        # pick currently best top k hypotheses (flattened order)
        topk_scores, topk_ids = curr_scores.topk(size, dim=-1)

        if alpha > -1:
            # recover original log probs
            topk_log_probs = topk_scores * length_penalty
        else:
            topk_log_probs = topk_scores.clone()

        # reconstruct beam origin and true word ids from flattened order
        topk_beam_index = topk_ids.div(trg_vocab_size, rounding_mode="floor")
        topk_ids = topk_ids.fmod(trg_vocab_size)

        # map beam_index to batch_index in the flat representation
        batch_index = topk_beam_index + beam_offset[
            : topk_beam_index.size(0)
        ].unsqueeze(1)
        select_indices = batch_index.view(-1)

        # append latest prediction
        alive_seq = torch.cat(
            [alive_seq.index_select(0, select_indices), topk_ids.view(-1, 1)],
            -1,
        )  # batch_size*k x hyp_len

        is_finished = topk_ids.eq(eos_index) | is_finished | topk_scores.eq(-np.inf)
        if step + 1 == max_output_length:
            is_finished.fill_(True)
        # end condition is whether all beams are finished
        end_condition = is_finished.all(-1)

        # save finished hypotheses
        if is_finished.any():
            predictions = alive_seq.view(-1, size, alive_seq.size(-1))
            for i in range(is_finished.size(0)):  # loop over batch instances
                b = batch_offset[i].item()
                if end_condition[i]:
                    is_finished[i].fill_(1)
                finished_hyp = is_finished[i].nonzero(as_tuple=False).view(-1)
                # store finished hypotheses for this batch
                # `finished_hyp` has shape (batch_size, beam_size, steps_so_far)
                for j in finished_hyp:  # loop over beam candidates
                    # Check if the prediction has more than one EOS. If it has more
                    # than one EOS, it means that the prediction should have already
                    # been added to the hypotheses, so you don't have to add them again.
                    if (predictions[i, j, 1:] == eos_index).nonzero(
                        as_tuple=False
                    ).numel() < 2:
                        # ignore start_token
                        hypotheses[b].append((topk_scores[i, j], predictions[i, j, 1:]))
                # if the batch reached the end, save the n_best hypotheses
                if end_condition[i]:
                    best_hyp = sorted(hypotheses[b], key=lambda x: x[0], reverse=True)
                    for n, (score, pred) in enumerate(best_hyp):
                        if n >= n_best:
                            break
                        results["scores"][b].append(score)
                        results["predictions"][b].append(pred)
            non_finished = end_condition.eq(False).nonzero(as_tuple=False).view(-1)
            # if all sentences are translated, no need to go further
            if len(non_finished) == 0:
                break
            # remove finished instances for the next step
            topk_log_probs = topk_log_probs.index_select(0, non_finished)
            batch_index = batch_index.index_select(0, non_finished)
            batch_offset = batch_offset.index_select(0, non_finished)
            alive_seq = predictions.index_select(0, non_finished).view(
                -1, alive_seq.size(-1)
            )
            is_finished = is_finished.index_select(0, non_finished)

        # reorder indices, outputs and masks
        select_indices = batch_index.view(-1)
        encoder_output = encoder_output.index_select(0, select_indices)
        src_mask = src_mask.index_select(0, select_indices)

        if hidden is not None and not transformer:
            if isinstance(hidden, tuple):
                # for LSTMs, states are tuples of tensors
                h, c = hidden
                h = h.index_select(0, select_indices)
                c = c.index_select(0, select_indices)
                hidden = (h, c)
            else:
                # for GRUs, states are single tensors
                hidden = hidden.index_select(0, select_indices)

        if att_vectors is not None:
            att_vectors = att_vectors.index_select(0, select_indices)

    def pad_and_stack_hyps(hyps, pad_value):
        filled = (
            np.ones((len(hyps), max([h.shape[0] for h in hyps])), dtype=int) * pad_value
        )
        for j, h in enumerate(hyps):
            for k, i in enumerate(h):
                filled[j, k] = i
        return filled

    # from results to stacked outputs
    final_outputs = pad_and_stack_hyps(
        [u.cpu().numpy() for r in results["predictions"] for u in r],
        pad_value=pad_index,
    )
    return final_outputs, None


def run_batch(
    model: Model,
    batch: Batch,
    max_output_length: int,
    beam_size: int,
    beam_alpha: float,
    n_best: int = 1,
    generate_unk: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get outputs and attentions scores for a given batch

    :param model: Model class
    :param batch: batch to generate hypotheses for
    :param max_output_length: maximum length of hypotheses
    :param beam_size: size of the beam for beam search, if 0 use greedy
    :param beam_alpha: alpha value for beam search
    :param n_best: candidates to return
    :returns:
        - stacked_output: hypotheses for batch,
        - stacked_attention_scores: attention scores for batch
    """
    with torch.no_grad():
        encoder_output, encoder_hidden, _, _ = model(
            return_type="encode", **vars(batch)
        )

    # if maximum output length is not globally specified, adapt to src len
    if max_output_length is None:
        max_output_length = int(max(batch.src_length.cpu().numpy()) * 1.5)

    # decoding
    if beam_size < 2:  # greedy
        stacked_output, stacked_attention_scores = greedy(
            src_mask=batch.src_mask,
            max_output_length=max_output_length,
            model=model,
            encoder_output=encoder_output,
            encoder_hidden=encoder_hidden,
            generate_unk=generate_unk,
        )

    else:  # beam search
        stacked_output, stacked_attention_scores = beam_search(
            model=model,
            size=beam_size,
            encoder_output=encoder_output,
            encoder_hidden=encoder_hidden,
            src_mask=batch.src_mask,
            max_output_length=max_output_length,
            alpha=beam_alpha,
            n_best=n_best,
            generate_unk=generate_unk,
        )

    return stacked_output, stacked_attention_scores
