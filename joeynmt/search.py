# coding: utf-8
"""
Search module
"""
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from joeynmt.batch import Batch
from joeynmt.decoders import RecurrentDecoder, TransformerDecoder
from joeynmt.helpers import tile
from joeynmt.model import Model

__all__ = ["greedy", "beam_search", "search"]


def greedy(
    src_mask: Tensor,
    max_output_length: int,
    model: Model,
    encoder_output: Tensor,
    encoder_hidden: Tensor,
    **kwargs,
) -> Tuple[np.array, np.array, np.ndarray]:
    """
    Greedy decoding. Select the token word highest probability at each time step.
    This function is a wrapper that calls recurrent_greedy for recurrent decoders and
    transformer_greedy for transformer decoders.

    :param src_mask: mask for source inputs, 0 for positions after </s>
    :param max_output_length: maximum length for the hypotheses
    :param model: model to use for greedy decoding
    :param encoder_output: encoder hidden states for attention
    :param encoder_hidden: encoder last state for decoder initialization
    :return:
        - stacked_output: output hypotheses (2d array of indices),
        - stacked_scores: scores (2d array of token-wise log probabilities),
        - stacked_attention_scores: attention scores (3d array)
    """
    # pylint: disable=no-else-return
    if isinstance(model.decoder, TransformerDecoder):
        return transformer_greedy(
            src_mask,
            max_output_length,
            model,
            encoder_output,
            encoder_hidden,
            **kwargs,
        )
    elif isinstance(model.decoder, RecurrentDecoder):
        return recurrent_greedy(src_mask, max_output_length, model, encoder_output,
                                encoder_hidden, **kwargs)
    else:
        raise NotImplementedError(
            f"model.decoder({model.decoder.__class__.__name__}) not supported.")


def recurrent_greedy(
    src_mask: Tensor,
    max_output_length: int,
    model: Model,
    encoder_output: Tensor,
    encoder_hidden: Tensor,
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Greedy decoding: in each step, choose the word that gets highest score.
    Version for recurrent decoder.

    :param src_mask: mask for source inputs, 0 for positions after </s>
    :param max_output_length: maximum length for the hypotheses
    :param model: model to use for greedy decoding
    :param encoder_output: encoder hidden states for attention
    :param encoder_hidden: encoder last state for decoder initialization
    :return:
        - stacked_output: output hypotheses (2d array of indices),
        - stacked_scores: scores (2d array of token-wise log probabilities),
        - stacked_attention_scores: attention scores (3d array)
    """
    bos_index = model.bos_index
    eos_index = model.eos_index
    unk_index = model.unk_index
    batch_size = src_mask.size(0)
    min_output_length: int = kwargs.get("min_output_length", 1)
    generate_unk: bool = kwargs.get("generate_unk", True)  # whether to generate UNK
    return_prob: bool = kwargs.get("return_prob", "none") == "hyp"
    prev_y = src_mask.new_full((batch_size, 1), fill_value=bos_index, dtype=torch.long)

    output = []
    scores = []
    attention_scores = []
    hidden = None
    prev_att_vector = None
    finished = src_mask.new_zeros((batch_size, 1)).byte()

    for step in range(max_output_length):
        # decode one single step
        with torch.no_grad():
            out, hidden, att_probs, prev_att_vector = model(
                return_type="decode",
                trg_input=prev_y,
                encoder_output=encoder_output,
                encoder_hidden=encoder_hidden,
                src_mask=src_mask,
                unroll_steps=1,
                decoder_hidden=hidden,
                att_vector=prev_att_vector,
            )
            # out: batch x time=1 x vocab (logits)

        if return_prob:
            out = F.log_softmax(out, dim=-1)

        if not generate_unk:
            out[:, :, unk_index] = float("-inf")

        # don't generate EOS until we reached min_output_length
        if step < min_output_length:
            out[:, :, eos_index] = float("-inf")

        # greedy decoding: choose arg max over vocabulary in each step
        prob, next_word = torch.max(out, dim=-1)  # batch x time=1
        output.append(next_word.squeeze(1).detach().cpu().numpy())
        if return_prob:
            scores.append(prob.squeeze(1).detach().cpu().numpy())
        prev_y = next_word
        attention_scores.append(att_probs.squeeze(1).detach().cpu().numpy())
        # shape: (batch_size, max_src_length)

        # check if previous symbol was <eos>
        is_eos = torch.eq(next_word, eos_index)
        finished += is_eos
        # stop predicting if <eos> reached for all elements in batch
        if (finished >= 1).sum() == batch_size:
            break

    stacked_output = np.stack(output, axis=1)  # batch, time
    stacked_scores = np.stack(scores, axis=1) if return_prob else None
    stacked_attention_scores = np.stack(attention_scores, axis=1)
    return stacked_output, stacked_scores, stacked_attention_scores


def transformer_greedy(
    src_mask: Tensor,
    max_output_length: int,
    model: Model,
    encoder_output: Tensor,
    encoder_hidden: Tensor,
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray, None]:
    """
    Special greedy function for transformer, since it works differently.
    The transformer remembers all previous states and attends to them.

    :param src_mask: mask for source inputs, 0 for positions after </s>
    :param max_output_length: maximum length for the hypotheses
    :param model: model to use for greedy decoding
    :param encoder_output: encoder hidden states for attention
    :param encoder_hidden: encoder final state (unused in Transformer)
    :return:
        - stacked_output: output hypotheses (2d array of indices),
        - stacked_scores: scores (2d array of token-wise log probabilities),
        - stacked_attention_scores: attention scores (3d array)
    """
    # pylint: disable=unused-argument
    bos_index = model.bos_index
    eos_index = model.eos_index
    unk_index = model.unk_index
    pad_index = model.pad_index
    batch_size, _, src_len = src_mask.size()

    # options to control generation
    generate_unk: bool = kwargs.get("generate_unk", True)  # whether to generate UNK
    return_attention: bool = kwargs.get("return_attention", False)
    return_prob: bool = kwargs.get("return_prob", "none") == "hyp"
    min_output_length: int = kwargs.get("min_output_length", 1)
    repetition_penalty: float = kwargs.get("repetition_penalty", -1)
    no_repeat_ngram_size: int = kwargs.get("no_repeat_ngram_size", -1)
    encoder_input: Tensor = kwargs.get("encoder_input", None)  # for repetition blocker
    compute_softmax: bool = (return_prob or repetition_penalty > 0
                             or no_repeat_ngram_size > 0 or encoder_input is not None)

    # start with BOS-symbol for each sentence in the batch
    ys = encoder_output.new_full((batch_size, 1), bos_index, dtype=torch.long)

    # placeholder for scores
    yv = ys.new_zeros((batch_size, 1), dtype=torch.float) if return_prob else None

    # placeholder for attentions
    yt = ys.new_zeros((batch_size, 1, src_len), dtype=torch.float) \
        if return_attention else None

    # a subsequent mask is intersected with this in decoder forward pass
    trg_mask = src_mask.new_ones([1, 1, 1])
    if isinstance(model, torch.nn.DataParallel):
        trg_mask = torch.stack([src_mask.new_ones([1, 1]) for _ in model.device_ids])

    finished = src_mask.new_zeros(batch_size).byte()

    for step in range(max_output_length):
        with torch.no_grad():
            out, _, att, _ = model(
                return_type="decode",
                trg_input=ys,  # model.trg_embed(ys) # embed the previous tokens
                encoder_output=encoder_output,
                encoder_hidden=None,
                src_mask=src_mask,
                unroll_steps=None,
                decoder_hidden=None,
                trg_mask=trg_mask,
                return_attention=return_attention,
            )
            out = out[:, -1]  # logits
            if not generate_unk:
                out[:, unk_index] = float("-inf")

            # don't generate EOS until we reached min_output_length
            if step < min_output_length:
                out[:, eos_index] = float("-inf")

            if compute_softmax:
                out = F.log_softmax(out, dim=-1)

                # ngram blocker
                if no_repeat_ngram_size > 1:
                    out = block_repeat_ngrams(
                        ys,
                        out,
                        no_repeat_ngram_size,
                        step,
                        src_tokens=encoder_input,
                        exclude_tokens=[bos_index, eos_index, unk_index, pad_index],
                    )

                # repetition_penalty
                if repetition_penalty > 1.0:
                    out = penalize_repetition(
                        ys,
                        out,
                        repetition_penalty,
                        exclude_tokens=[bos_index, eos_index, unk_index, pad_index],
                    )
                    if encoder_input is not None:
                        out = penalize_repetition(
                            encoder_input,
                            out,
                            repetition_penalty,
                            exclude_tokens=[bos_index, eos_index, unk_index, pad_index],
                        )

            # take the most likely token
            prob, next_word = torch.max(out, dim=1)
            next_word = next_word.data
            ys = torch.cat([ys, next_word.unsqueeze(-1)], dim=1)
            if return_prob:
                prob = prob.data
                yv = torch.cat([yv, prob.unsqueeze(-1)], dim=1)
            if return_attention:
                assert att is not None
                att = att.data[:, -1, :].unsqueeze(1)  # take last trg token only
                yt = torch.cat([yt, att], dim=1)  # (batch_size, trg_len, src_len)

        # check if previous symbol was <eos>
        is_eos = torch.eq(next_word, eos_index)
        finished += is_eos
        # stop predicting if <eos> reached for all elements in batch
        if (finished >= 1).sum() == batch_size:
            break

    # remove BOS-symbol
    output = ys[:, 1:].detach().cpu().numpy()
    scores = yv[:, 1:].detach().cpu().numpy() if return_prob else None
    attention = yt[:, 1:, :].detach().cpu().numpy() if return_attention else None
    return output, scores, attention


def beam_search(
    model: Model,
    beam_size: int,
    encoder_output: Tensor,
    encoder_hidden: Tensor,
    src_mask: Tensor,
    max_output_length: int,
    alpha: float,
    n_best: int = 1,
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray, None]:
    """
    Beam search with size k. In each decoding step, find the k most likely partial
    hypotheses. Inspired by OpenNMT-py, adapted for Transformer.

    :param model:
    :param beam_size: size of the beam
    :param encoder_output:
    :param encoder_hidden:
    :param src_mask:
    :param max_output_length:
    :param alpha: `alpha` factor for length penalty
    :param n_best: return this many hypotheses, <= beam (currently only 1)
    :return:
        - stacked_output: output hypotheses (2d array of indices),
        - stacked_scores: scores (2d array of sequence-wise log probabilities),
        - stacked_attention_scores: attention scores (3d array)
    """
    # pylint: disable=too-many-statements,too-many-branches
    assert beam_size > 0, "Beam size must be >0."
    assert n_best <= beam_size, f"Can only return {beam_size} best hypotheses."

    # Take the best 2 x {beam_size} predictions so as to avoid duplicates in generation.
    # yet, only return {n_best} hypotheses.
    # beam_size = 2 * beam_size

    # init
    bos_index = model.bos_index
    eos_index = model.eos_index
    pad_index = model.pad_index
    unk_index = model.unk_index
    batch_size = src_mask.size(0)

    generate_unk: bool = kwargs.get("generate_unk", True)  # whether to generate UNK
    return_prob: bool = kwargs.get("return_prob", "none") == "hyp"
    min_output_length: int = kwargs.get("min_output_length", 1)
    repetition_penalty: float = kwargs.get("repetition_penalty", -1)
    no_repeat_ngram_size: int = kwargs.get("no_repeat_ngram_size", -1)
    encoder_input: Tensor = kwargs.get("encoder_input", None)  # for repetition blocker

    trg_vocab_size = model.decoder.output_size
    device = encoder_output.device
    is_transformer = isinstance(model.decoder, TransformerDecoder)

    att_vectors = None  # for RNN only, not used for Transformer
    hidden = None  # for RNN only, not used for Transformer
    trg_mask = None  # for Transformer only, not used for RNN

    # Recurrent models only: initialize RNN hidden state
    if not is_transformer:
        # pylint: disable=protected-access
        # tile encoder states and decoder initial states beam_size times
        # `hidden` shape: (layers, batch_size * beam_size, dec_hidden_size)
        hidden = model.decoder._init_hidden(encoder_hidden)
        hidden = tile(hidden, beam_size, dim=1)
        # DataParallel splits batch along the 0th dim.
        # Place back the batch_size to the 1st dim here.
        # `hidden` shape: (batch_size * beam_size, layers, dec_hidden_size)
        if isinstance(hidden, tuple):
            h, c = hidden
            hidden = (h.permute(1, 0, 2), c.permute(1, 0, 2))
        else:
            hidden = hidden.permute(1, 0, 2)

    # `encoder_output` shape: (batch_size * beam_size, src_len, enc_hidden_size)
    encoder_output = tile(encoder_output.contiguous(), beam_size, dim=0)
    # `src_mask` shape: (batch_size * beam_size, 1, src_len)
    src_mask = tile(src_mask, beam_size, dim=0)
    # `encoder_input` shape: (batch_size * beam_size, src_len)
    if encoder_input is not None:  # used in src-side repetition blocker
        encoder_input = tile(encoder_input.contiguous(), beam_size,
                             dim=0).view(batch_size * beam_size, -1)
        assert encoder_input.size(0) == batch_size * beam_size, (
            encoder_input.size(0),
            batch_size * beam_size,
        )

    # Transformer only: create target mask
    if is_transformer:
        trg_mask = src_mask.new_ones([1, 1, 1])
        if isinstance(model, torch.nn.DataParallel):
            trg_mask = torch.stack(
                [src_mask.new_ones([1, 1]) for _ in model.device_ids])

    # numbering elements in the batch
    batch_offset = torch.arange(batch_size, dtype=torch.long, device=device)

    # numbering elements in the extended batch, i.e. k copies of each batch element
    beam_offset = torch.arange(0,
                               batch_size * beam_size,
                               step=beam_size,
                               dtype=torch.long,
                               device=device)

    # keeps track of the top beam size hypotheses to expand for each element in the
    # batch to be further decoded (that are still "alive")
    # `alive_seq` shape: (batch_size * beam_size, hyp_len) ... now hyp_len = 1
    alive_seq = torch.full((batch_size * beam_size, 1),
                           bos_index,
                           dtype=torch.long,
                           device=device)

    # Give full probability (=zero in log space) to the first beam on the first step.
    # `topk_log_probs` shape: (batch_size, beam_size)
    topk_log_probs = torch.zeros(batch_size, beam_size, device=device)
    topk_log_probs[:, 1:] = float("-inf")

    # Structure that holds finished hypotheses.
    hypotheses = [[] for _ in range(batch_size)]

    results = {
        "predictions": [[] for _ in range(batch_size)],
        "scores": [[] for _ in range(batch_size)],
    }

    # indicator if the generation is finished
    # `is_finished` shape: (batch_size, beam_size)
    is_finished = torch.full((batch_size, beam_size),
                             False,
                             dtype=torch.bool,
                             device=device)

    for step in range(max_output_length):
        if is_transformer:
            # For Transformer, we feed the complete predicted sentence so far.
            decoder_input = alive_seq

            # decode one single step
            with torch.no_grad():
                logits, _, _, _ = model(  # logits before final softmax
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
            logits = logits[:, -1]
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

        # compute log probability distribution over trg vocab
        # `log_probs` shape: (remaining_batch_size * beam_size, trg_vocab)
        log_probs = F.log_softmax(logits, dim=-1).squeeze(1)
        if not generate_unk:
            log_probs[:, unk_index] = float("-inf")

        # don't generate EOS until we reached min_output_length
        if step < min_output_length:
            log_probs[:, eos_index] = float("-inf")

        # block repetitions
        if no_repeat_ngram_size > 0:
            log_probs = block_repeat_ngrams(
                alive_seq,
                log_probs,
                no_repeat_ngram_size,
                step,
                src_tokens=encoder_input,
                exclude_tokens=[bos_index, eos_index, unk_index, pad_index],
            )

        if repetition_penalty > 1.0:
            log_probs = penalize_repetition(
                alive_seq,
                log_probs,
                repetition_penalty,
                exclude_tokens=[bos_index, eos_index, unk_index, pad_index],
            )
            if encoder_input is not None:  # src
                log_probs = penalize_repetition(
                    encoder_input,
                    log_probs,
                    repetition_penalty,
                    exclude_tokens=[bos_index, eos_index, unk_index, pad_index],
                )

        # multiply probs by the beam probability (=add logprobs)
        # `log_probs` shape: (remaining_batch_size * beam_size, trg_vocab)
        log_probs += topk_log_probs.view(-1).unsqueeze(1)  # add column-wise
        curr_scores = log_probs.clone()

        # compute length penalty
        if alpha > 0:
            length_penalty = ((5.0 + (step + 1)) / 6.0)**alpha
            curr_scores /= length_penalty

        # flatten log_probs into a list of possibilities
        # `curr_scores` shape: (remaining_batch_size, beam_size * trg_vocab_size)
        curr_scores = curr_scores.reshape(-1, beam_size * trg_vocab_size)

        # pick currently best top k hypotheses (flattened order)
        # `topk_scores` and `topk_ids` shape: (remaining_batch_size, beam_size)
        topk_scores, topk_ids = curr_scores.topk(beam_size, dim=-1)

        if alpha > 0:
            # recover original log probs
            topk_log_probs = topk_scores * length_penalty
        else:
            topk_log_probs = topk_scores.clone()

        # reconstruct beam origin and true word ids from flattened order
        topk_beam_index = topk_ids.div(trg_vocab_size, rounding_mode="floor")
        topk_ids = topk_ids.fmod(trg_vocab_size)  # resolve true word ids

        # map topk_beam_index to batch_index in the flat representation
        # `batch_index` shape: (remaining_batch_size, beam_size)
        batch_index = topk_beam_index + beam_offset[:topk_ids.size(0)].unsqueeze(1)
        select_indices = batch_index.view(-1)

        # append latest prediction
        # `alive_seq` shape: (remaining_batch_size * beam_size, hyp_len)
        alive_seq = torch.cat(
            [alive_seq.index_select(0, select_indices),
             topk_ids.view(-1, 1)],
            -1,
        )
        # `is_finished` shape: (remaining_batch_size, beam_size)
        is_finished = topk_ids.eq(eos_index) | is_finished | topk_scores.eq(-np.inf)
        if step + 1 == max_output_length:
            is_finished.fill_(True)

        # end condition is whether all beam candidates in each example are finished
        end_condition = is_finished.all(-1)  # shape: (remaining_batch_size,)

        # save finished hypotheses
        if is_finished.any():
            # `predictions` shape: (remaining_batch_size, beam_size, hyp_len)
            predictions = alive_seq.view(-1, beam_size, alive_seq.size(-1))

            for i in range(is_finished.size(0)):  # loop over remaining examples
                b = batch_offset[i].item()  # index of that example in the batch
                if end_condition[i]:
                    is_finished[i].fill_(True)
                # indices of finished beam candidates for this example (1d tensor)
                # i.e. finished_hyp = [0, 1] means 0th and 1st candidates reached eos
                finished_hyp = is_finished[i].nonzero(as_tuple=False).view(-1)
                for j in finished_hyp:  # loop over finished beam candidates
                    n_eos = (predictions[i, j, 1:] == eos_index).count_nonzero().item()
                    if n_eos > 1:  # pylint: disable=no-else-continue
                        # If the prediction has more than one EOS, it means that the
                        # prediction should have already been added to the hypotheses,
                        # so we don't add them again.
                        continue
                    elif (n_eos == 0 and step + 1 == max_output_length) or (
                            n_eos == 1 and predictions[i, j, -1] == eos_index):
                        # If the prediction has no EOS, it means we reached max length.
                        # If the prediction has exactly one EOS, it should be the last
                        # token of the sequence. Then we add it to the hypotheses.
                        # We exclude the candidate which has one EOS but some other
                        # token was appended after EOS.
                        hypotheses[b].append((topk_scores[i, j], predictions[i, j, 1:]))

                # if all nbest candidates of the i-th example reached the end, save them
                if end_condition[i]:
                    best_hyp = sorted(hypotheses[b], key=lambda x: x[0], reverse=True)
                    for n, (score, pred) in enumerate(best_hyp):
                        if n >= n_best:
                            break
                        if len(pred) < max_output_length:
                            assert (
                                pred[-1] == eos_index
                            ), f"adding a candidate which doesn't end with eos: {pred}"
                        results["scores"][b].append(score)
                        results["predictions"][b].append(pred)

            # batch indices of the examples which contain unfinished candidates
            unfinished = end_condition.eq(False).nonzero(as_tuple=False).view(-1)
            # if all examples are translated, no need to go further
            if len(unfinished) == 0:
                break
            # remove finished examples for the next step
            # shape: (remaining_batch_size, beam_size)
            batch_index = batch_index.index_select(0, unfinished)
            topk_log_probs = topk_log_probs.index_select(0, unfinished)
            is_finished = is_finished.index_select(0, unfinished)
            batch_offset = batch_offset.index_select(0, unfinished)

            # **CAUTION:** `alive_seq` still can contain finished beam candidates
            # because we only remove finished examples. For instance, beam_size = 3,
            # 2 sents remain in batch and all 3 candidates of the 1st sent is finished.
            #     end_condition = [True, False]
            #     unfinished = [1]  # 2nd sent (idx 1) remaining in batch is unfinished
            # Say, the first and the second beam candidate of the second example are
            # finished but the third candidate of the second example is still alive.
            # Then we include all three candidates of the second example in `alive_seq`,
            # even though the 1st and 2nd candidates of the second example are finished.
            #     alive_seq = [
            #         [5, 8, 7, 3, 1],  # eos_index = 3; already finished in prev step
            #         [4, 9, 6, 5, 3],  # eos_index = 3; finished in this step
            #         [4, 9, 5, 6, 7],  # not finished yet
            #     ]
            # Yet, we won't add already finished candidates to the `hypotheses` list,
            # but only the candidates that finished in the very current time step.
            # TODO: release the space of finished ones, explore more unfinished ones.
            # `alive_seq` shape: (remaining_batch_size * beam_size, hyp_len)
            alive_seq = predictions.index_select(0, unfinished).view(
                -1, alive_seq.size(-1))
            if encoder_input is not None:
                src_len = encoder_input.size(-1)
                encoder_input = (encoder_input.view(-1, beam_size,
                                                    src_len).index_select(
                                                        0,
                                                        unfinished).view(-1, src_len))
                assert encoder_input.size(0) == alive_seq.size(0)

        # reorder indices, outputs and masks
        select_indices = batch_index.view(-1)
        encoder_output = encoder_output.index_select(0, select_indices)
        src_mask = src_mask.index_select(0, select_indices)

        if hidden is not None and not is_transformer:
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

    def pad_and_stack_hyps(hyps: List[np.ndarray]):
        filled = (np.ones(
            (len(hyps), max([h.shape[0] for h in hyps])), dtype=int) * pad_index)
        for j, h in enumerate(hyps):
            for k, i in enumerate(h):
                filled[j, k] = i
        return filled

    # from results to stacked outputs
    # `final_outputs`: shape (batch_size * n_best, hyp_len)
    final_outputs = pad_and_stack_hyps(
        [u.cpu().numpy() for r in results["predictions"] for u in r], )

    # sequence-wise log probabilities (summed up over the sequence)
    # `scores`: shape (batch_size * n_best, 1)
    scores = (np.array([[u.item()] for r in results["scores"]
                        for u in r]) if return_prob else None)
    return final_outputs, scores, None


def search(
    model: Model,
    batch: Batch,
    max_output_length: int,
    beam_size: int,
    beam_alpha: float,
    n_best: int = 1,
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Get outputs and attentions scores for a given batch.

    :param model: Model class
    :param batch: batch to generate hypotheses for
    :param max_output_length: maximum length of hypotheses
    :param beam_size: size of the beam for beam search, if 0 use greedy
    :param beam_alpha: alpha value for beam search
    :param n_best: candidates to return
    :returns:
        - stacked_output: hypotheses for batch,
        - stacked_scores: log probabilities for batch,
        - stacked_attention_scores: attention scores for batch
    """
    with torch.no_grad():
        encoder_output, encoder_hidden, _, _ = model(return_type="encode",
                                                     **vars(batch))

    # if maximum output length is not globally specified, adapt to src len
    if max_output_length < 0:
        max_output_length = int(max(batch.src_length.cpu().numpy()) * 1.5)

    # block src-side repetition (to avoid untranslated copy in trg)
    if (kwargs.get("no_repeat_ngram_size", -1) > 1
            or kwargs.get("repetition_penalty", -1) > 1):
        kwargs["encoder_input"] = batch.src

    # decoding
    if beam_size < 2:  # greedy
        stacked_output, stacked_scores, stacked_attention_scores = greedy(
            src_mask=batch.src_mask,
            max_output_length=max_output_length,
            model=model,
            encoder_output=encoder_output,
            encoder_hidden=encoder_hidden,
            **kwargs,
        )

    else:  # beam search
        stacked_output, stacked_scores, stacked_attention_scores = beam_search(
            model=model,
            beam_size=beam_size,
            encoder_output=encoder_output,
            encoder_hidden=encoder_hidden,
            src_mask=batch.src_mask,
            max_output_length=max_output_length,
            alpha=beam_alpha,
            n_best=n_best,
            **kwargs,
        )

    return stacked_output, stacked_scores, stacked_attention_scores


def block_repeat_ngrams(tokens: Tensor, scores: Tensor, no_repeat_ngram_size: int,
                        step: int, **kwargs) -> Tensor:
    """
    For each hypothesis, check a list of previous ngrams and set associated log probs
    to -inf. Taken from fairseq's NGramRepeatBlock.

    :param tokens: target tokens generated so far
    :param scores: log probabilities of the next token to generate in this time step
    :param no_repeat_ngram_size: ngram size to prohibit
    :param step: generation step (= length of hypotheses so far)
    """
    hyp_size = tokens.size(0)
    banned_batch_tokens = [set([]) for _ in range(hyp_size)]

    trg_tokens = tokens.cpu().tolist()
    check_end_pos = step + 2 - no_repeat_ngram_size
    offset = no_repeat_ngram_size - 1

    src_tokens = kwargs.get("src_tokens", None)
    if src_tokens is not None:
        src_length = src_tokens.size(-1)
        assert src_tokens.size(0) == hyp_size, (src_tokens.size(), hyp_size)
        src_tokens = src_tokens.cpu().tolist()
    exclude_tokens = kwargs.get("exclude_tokens", [])

    # get repeated ngrams
    for hyp_idx in range(hyp_size):
        if len(trg_tokens[hyp_idx]) > no_repeat_ngram_size:
            # (n-1) token prefix at the time step
            #                       0  1  2  3  4  <- step
            # if tokens[hyp_idx] = [2, 5, 5, 6, 5]    at step 4 with ngram_size = 3,
            #                                ^  ^  ^
            # then ngram_to_check = [6, 5], and set the token in the next position to
            # -inf, if there are ngrams starts with [6, 5].
            ngram_to_check = trg_tokens[hyp_idx][-offset:]

            for i in range(1, check_end_pos):  # ignore BOS
                if ngram_to_check == trg_tokens[hyp_idx][i:i + offset]:
                    banned_batch_tokens[hyp_idx].add(trg_tokens[hyp_idx][i + offset])

            # src_tokens
            if src_tokens is not None:
                check_end_pos_src = src_length + 1 - no_repeat_ngram_size
                for i in range(check_end_pos_src):  # no BOS in src
                    if ngram_to_check == src_tokens[hyp_idx][i:i + offset]:
                        banned_batch_tokens[hyp_idx].add(src_tokens[hyp_idx][i +
                                                                             offset])

    # set the score of the banned tokens to -inf
    for i, banned_tokens in enumerate(banned_batch_tokens):
        banned_tokens = set(banned_tokens) - set(exclude_tokens)
        scores[i, list(banned_tokens)] = float("-inf")
    return scores


def penalize_repetition(tokens: Tensor,
                        scores: Tensor,
                        penalty: float,
                        exclude_tokens: List[int] = None) -> Tensor:
    """
    Reduce probability of the given tokens.
    Taken from Huggingface's RepetitionPenaltyLogitsProcessor.

    :param tokens: token ids to penalize
    :param scores: log probabilities of the next token to generate
    :param penalty: penalty value, bigger value implies less probability
    :param exclude_tokens: list of token ids to exclude from penalizing
    """
    scores_before = scores if exclude_tokens else None
    score = torch.gather(scores, 1, tokens)

    # if score < 0 then repetition penalty has to be multiplied
    # to reduce the previous token probability
    score = torch.where(score < 0, score * penalty, score / penalty)

    scores.scatter_(1, tokens, score)

    # exclude special tokens
    if exclude_tokens:
        for token in exclude_tokens:
            # pylint: disable=unsubscriptable-object
            scores[:, token] = scores_before[:, token]
    return scores
