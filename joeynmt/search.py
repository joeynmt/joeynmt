# coding: utf-8
import torch
import torch.nn.functional as F
import numpy as np

from joeynmt.helpers import tile


def greedy(src_mask, embed, bos_index, max_output_length, decoder,
           encoder_output, encoder_hidden):
    """
    Greedy decoding: in each step, choose the word that gets highest score.

    :param src_mask:
    :param embed:
    :param bos_index:
    :param max_output_length:
    :param decoder:
    :param encoder_output:
    :param encoder_hidden:
    :return:
    """
    batch_size = src_mask.size(0)
    prev_y = src_mask.new_full(size=[batch_size, 1], fill_value=bos_index,
                               dtype=torch.long)
    output = []
    attention_scores = []
    hidden = None
    prev_att_vector = None
    for t in range(max_output_length):
        # decode one single step
        out, hidden, att_probs, prev_att_vector = decoder(
            encoder_output=encoder_output,
            encoder_hidden=encoder_hidden,
            src_mask=src_mask,
            trg_embed=embed(prev_y),
            hidden=hidden,
            prev_att_vector=prev_att_vector,
            unrol_steps=1)
        # out: batch x time=1 x vocab (logits)

        # greedy decoding: choose arg max over vocabulary in each step
        next_word = torch.argmax(out, dim=-1)  # batch x time=1
        output.append(next_word.squeeze(1).cpu().numpy())
        prev_y = next_word
        attention_scores.append(att_probs.squeeze(1).cpu().numpy())
        # batch, max_src_lengths
    stacked_output = np.stack(output, axis=1)  # batch, time
    stacked_attention_scores = np.stack(attention_scores, axis=1)
    return stacked_output, stacked_attention_scores


def beam_search(decoder, size, bos_index, eos_index, pad_index, encoder_output,
                encoder_hidden, src_mask, max_output_length, alpha, embed,
                n_best=1):
    """
    Beam search with size k. Follows OpenNMT-py implementation.
    In each decoding step, find the k most likely partial hypotheses.

    :param decoder:
    :param size: size of the beam
    :param bos_index:
    :param eos_index:
    :param pad_index:
    :param encoder_output:
    :param encoder_hidden:
    :param src_mask:
    :param max_output_length:
    :param alpha: `alpha` factor for length penalty
    :param embed:
    :param n_best: return this many hypotheses, <= beam
    :return:
    """
    # init
    batch_size = src_mask.size(0)
    hidden = decoder.init_hidden(encoder_hidden)

    # tile hidden decoder states and encoder output beam_size times
    hidden = tile(hidden, size, dim=1)  # layers x batch*k x dec_hidden_size
    att_vectors = None

    encoder_output = tile(encoder_output.contiguous(), size,
                          dim=0)  # batch*k x src_len x enc_hidden_size

    src_mask = tile(src_mask, size, dim=0)  # batch*k x 1 x src_len

    batch_offset = torch.arange(
        batch_size, dtype=torch.long, device=encoder_output.device)
    beam_offset = torch.arange(
        0,
        batch_size * size,
        step=size,
        dtype=torch.long,
        device=encoder_output.device)
    alive_seq = torch.full(
        [batch_size * size, 1],
        bos_index,
        dtype=torch.long,
        device=encoder_output.device)

    # Give full probability to the first beam on the first step.
    topk_log_probs = (torch.tensor([0.0] + [float("-inf")] * (size - 1),
                                   device=encoder_output.device).repeat(
        batch_size))

    # Structure that holds finished hypotheses.
    hypotheses = [[] for _ in range(batch_size)]

    results = {}
    results["predictions"] = [[] for _ in range(batch_size)]
    results["scores"] = [[] for _ in range(batch_size)]
    results["gold_score"] = [0] * batch_size

    for step in range(max_output_length):
        decoder_input = alive_seq[:, -1].view(-1, 1)

        # expand current hypotheses
        # decode one single step
        # out: logits for final softmax
        out, hidden, att_scores, att_vectors = decoder(
            encoder_output=encoder_output,
            encoder_hidden=encoder_hidden,
            src_mask=src_mask,
            trg_embed=embed(decoder_input),
            hidden=hidden,
            prev_att_vector=att_vectors,
            unrol_steps=1)

        log_probs = F.log_softmax(out, dim=-1).squeeze(1)  # batch*k x trg_vocab

        # multiply probs by the beam probability (=add logprobs)
        log_probs += topk_log_probs.view(-1).unsqueeze(1)
        curr_scores = log_probs

        # compute length penalty
        if alpha > -1:
            length_penalty = ((5.0 + (step + 1)) / 6.0) ** alpha
            curr_scores /= length_penalty

        # flatten log_probs into a list of possibilities
        curr_scores = curr_scores.reshape(-1, size * decoder.output_size)

        # pick currently best top k hypotheses (flattened order)
        topk_scores, topk_ids = curr_scores.topk(size, dim=-1)

        if alpha > -1:
            # recover original log probs
            topk_log_probs = topk_scores * length_penalty

        # reconstruct beam origin and true word ids from flattened order
        topk_beam_index = topk_ids.div(decoder.output_size)
        topk_ids = topk_ids.fmod(decoder.output_size)

        # map beam_index to batch_index in the flat representation
        batch_index = (
            topk_beam_index
            + beam_offset[:topk_beam_index.size(0)].unsqueeze(1))
        select_indices = batch_index.view(-1)

        # append latest prediction
        alive_seq = torch.cat(
            [alive_seq.index_select(0, select_indices),
             topk_ids.view(-1, 1)], -1)  # batch_size*k x hyp_len

        is_finished = topk_ids.eq(eos_index)
        if step + 1 == max_output_length:
            is_finished.fill_(1)
        # end condition is whether the top beam is finished
        end_condition = is_finished[:, 0].eq(1)

        # save finished hypotheses
        if is_finished.any():
            predictions = alive_seq.view(-1, size, alive_seq.size(-1))
            for i in range(is_finished.size(0)):
                b = batch_offset[i]
                if end_condition[i]:
                    is_finished[i].fill_(1)
                finished_hyp = is_finished[i].nonzero().view(-1)
                # store finished hypotheses for this batch
                for j in finished_hyp:
                    hypotheses[b].append((
                        topk_scores[i, j],
                        predictions[i, j, 1:])  # ignore start_token
                    )
                # if the batch reached the end, save the n_best hypotheses
                if end_condition[i]:
                    best_hyp = sorted(
                        hypotheses[b], key=lambda x: x[0], reverse=True)
                    for n, (score, pred) in enumerate(best_hyp):
                        if n >= n_best:
                            break
                        results["scores"][b].append(score)
                        results["predictions"][b].append(pred)
            non_finished = end_condition.eq(0).nonzero().view(-1)
            # if all sentences are translated, no need to go further
            if len(non_finished) == 0:
                break
            # remove finished batches for the next step
            topk_log_probs = topk_log_probs.index_select(0, non_finished)
            batch_index = batch_index.index_select(0, non_finished)
            batch_offset = batch_offset.index_select(0, non_finished)
            alive_seq = predictions.index_select(0, non_finished) \
                .view(-1, alive_seq.size(-1))

            # reorder indices, outputs and masks
            select_indices = batch_index.view(-1)
            encoder_output = encoder_output.index_select(0, select_indices)
            src_mask = src_mask.index_select(0, select_indices)

            if isinstance(hidden, tuple):
                # for LSTMs, states are tuples of tensors
                h, c = hidden
                h = h.index_select(1, select_indices)
                c = c.index_select(1, select_indices)
                hidden = (h, c)
            else:
                # for GRUs, states are single tensors
                hidden = hidden.index_select(1, select_indices)

            att_vectors = att_vectors.index_select(0, select_indices)

    def pad_and_stack_hyps(hyps, pad_value):
        filled = np.ones((len(hyps), max([h.shape[0] for h in hyps])),
                         dtype=int) * pad_value
        for j, h in enumerate(hyps):
            for k, i in enumerate(h):
                filled[j, k] = i
        return filled

    # from results to stacked outputs
    assert n_best == 1
    # only works for n_best=1 for now
    final_outputs = pad_and_stack_hyps([r[0].cpu().numpy() for r in
                                        results["predictions"]],
                                       pad_value=pad_index)

    # TODO also return attention scores and probabilities
    return final_outputs, None