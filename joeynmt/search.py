# coding: utf-8
import torch
import torch.nn.functional as F
import numpy as np

from joeynmt.helpers import tile

def greedy(src_mask, embed, bos_index, max_output_length, decoder,
           encoder_output, encoder_hidden):
    batch_size = src_mask.size(0)
    prev_y = src_mask.new_full(size=[batch_size, 1], fill_value=bos_index, dtype=torch.long)
    #prev_y = torch.ones(batch_size, 1).fill_(bos_index).long(). #.type_as(src_mask)
    output = []
    attention_scores = []
    hidden = None
    prev_att_vector = None
    for t in range(max_output_length):
        # decode one single step
        #trg_embed, encoder_output, encoder_hidden,
        #src_mask, unrol_steps, hidden = None)
        out, hidden, att_probs, prev_att_vector = decoder(
            encoder_output=encoder_output,
            encoder_hidden=encoder_hidden,
            src_mask=src_mask,
            trg_embed=embed(prev_y),
            hidden=hidden,
            prev_att_vector=prev_att_vector,
            unrol_steps=1)
        # out: batch x time=1 x vocab (logits)

        # greedy decoding
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
    """ Beam search with size k"""
    # init
    batch_size = src_mask.size(0)
    # print("batch_size", batch_size)
    hidden = decoder.init_hidden(encoder_hidden)

    # tile hidden decoder states and encoder output beam_size times.
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
        #  print("STEP {}".format(step))
        decoder_input = alive_seq[:, -1].view(-1, 1)
        #  print("decoder input", decoder_input.size())
        #  print("encoder_output", encoder_output.size())

        # expand current
        # decode one single step
        # out = logits for final softmax
        out, hidden, att_scores, att_vectors = decoder(
            encoder_output=encoder_output,
            encoder_hidden=encoder_hidden,
            src_mask=src_mask,
            trg_embed=embed(decoder_input),
            hidden=hidden,
            prev_att_vector=att_vectors,
            unrol_steps=1)

        log_probs = F.log_softmax(out, dim=-1).squeeze(1)  # batch*k x trg_vocab
        #     print("log_probs", log_probs.size())

        # Multiply probs by the beam probability.
        log_probs += topk_log_probs.view(-1).unsqueeze(1)

        curr_scores = log_probs

        # compute length penalty
        if alpha > -1:
            length_penalty = ((5.0 + (step + 1)) / 6.0) ** alpha
            #       print("length penalty", length_penalty)
            curr_scores /= length_penalty

        # Flatten probs into a list of possibilities.
        curr_scores = curr_scores.reshape(-1, size * decoder.output_size)
        #   print("curr", curr_scores.size())  # batch x k*trg_vocab_size
        # pick currently best top k hypotheses (flattened order)
        topk_scores, topk_ids = curr_scores.topk(size, dim=-1)
        # print(topk_scores, topk_ids)

        if alpha > -1:
            # Recover log probs.
            topk_log_probs = topk_scores * length_penalty

        # Resolve beam origin and true word ids from flattened order
        topk_beam_index = topk_ids.div(decoder.output_size)
        #   print("topk beam", topk_beam_index)
        topk_ids = topk_ids.fmod(decoder.output_size)
        #  print("topk ids", topk_ids)

        # Map beam_index to batch_index in the flat representation.
        batch_index = (
            topk_beam_index
            + beam_offset[:topk_beam_index.size(0)].unsqueeze(1))
        #  print("batch indx", batch_index)
        select_indices = batch_index.view(-1)
        #  print("select indx", select_indices)

        # Append last prediction.
        alive_seq = torch.cat(
            [alive_seq.index_select(0, select_indices),
             topk_ids.view(-1, 1)], -1)  # batch_size*k x hyp_len
        # print("alive", alive_seq)

        # TODO also keep track of attention

        is_finished = topk_ids.eq(eos_index)
        if step + 1 == max_output_length:
            is_finished.fill_(1)
        # End condition is top beam is finished.
        end_condition = is_finished[:, 0].eq(1)

        # Save finished hypotheses.
        if is_finished.any():
            # print("Finished a hyp.")
            predictions = alive_seq.view(-1, size, alive_seq.size(-1))
            for i in range(is_finished.size(0)):
                b = batch_offset[i]
                if end_condition[i]:
                    is_finished[i].fill_(1)
                finished_hyp = is_finished[i].nonzero().view(-1)
                # Store finished hypotheses for this batch.
                for j in finished_hyp:
                    hypotheses[b].append((
                        topk_scores[i, j],
                        predictions[i, j, 1:])  # Ignore start_token.
                    )
                # If the batch reached the end, save the n_best hypotheses.
                if end_condition[i]:
                    best_hyp = sorted(
                        hypotheses[b], key=lambda x: x[0], reverse=True)
                    for n, (score, pred) in enumerate(best_hyp):
                        if n >= n_best:
                            break
                        results["scores"][b].append(score)
                        results["predictions"][b].append(pred)
            non_finished = end_condition.eq(0).nonzero().view(-1)
            # If all sentences are translated, no need to go further.
            if len(non_finished) == 0:
                break
            # Remove finished batches for the next step.
            topk_log_probs = topk_log_probs.index_select(0, non_finished)
            batch_index = batch_index.index_select(0, non_finished)
            batch_offset = batch_offset.index_select(0, non_finished)
            alive_seq = predictions.index_select(0, non_finished) \
                .view(-1, alive_seq.size(-1))

            # Reorder states.
            select_indices = batch_index.view(-1)
            #   print("select indx", select_indices)
            #    print("encoder_output", encoder_output.size())
            encoder_output = encoder_output.index_select(0, select_indices)
            #     print("encoder output", encoder_output.size())
            src_mask = src_mask.index_select(0, select_indices)
            #      print("hidden", hidden.size())

            if isinstance(hidden, tuple):
                h, c = hidden
                h = h.index_select(1, select_indices)
                c = c.index_select(1, select_indices)
                hidden = (h, c)
            else:
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

    # TODO add attention scores
    return final_outputs, None