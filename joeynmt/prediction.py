# coding: utf-8
"""
This modules holds methods for generating predictions from a model.
"""
from typing import List
import numpy as np

import torch
from torchtext.data import Dataset

from joeynmt.constants import PAD_TOKEN
from joeynmt.helpers import bpe_postprocess, load_config, \
    get_latest_checkpoint, load_model_from_checkpoint, store_attention_plots
from joeynmt.metrics import bleu, chrf, token_accuracy, sequence_accuracy
from joeynmt.model import build_model, Model
from joeynmt.batch import Batch
from joeynmt.data import load_data, make_data_iter


# pylint: disable=too-many-arguments,too-many-locals,no-member
def validate_on_data(model: Model, data: Dataset, batch_size: int,
                     use_cuda: bool, max_output_length: int,
                     level: str, eval_metric: str, criterion: torch.nn.Module,
                     beam_size: int = 0, beam_alpha: int = -1) \
        -> (float, float, float, List[str], List[List[str]], List[str],
            List[str], List[List[str]], List[np.array]):
    """
    Generate translations for the given data.
    If `criterion` is not None and references are given, also compute the loss.

    :param model:
    :param data:
    :param batch_size:
    :param use_cuda:
    :param max_output_length:
    :param level:
    :param eval_metric:
    :param criterion:
    :param beam_size:
    :param beam_alpha:
    :return: current_valid_score: current validation score [eval_metric],
        valid_loss: validation loss,
        valid_ppl:, validation perplexity,
        valid_sources: validation sources,
        valid_sources_raw: raw validation sources (before post-processing),
        valid_references: validation references,
        valid_hypotheses: validation_hypotheses,
        decoded_valid: raw validation hypotheses (before post-processing),
        valid_attention_scores: attention scores for validation hypotheses
    """
    valid_iter = make_data_iter(dataset=data, batch_size=batch_size,
                                shuffle=False, train=False)
    valid_sources_raw = [s for s in data.src]
    pad_index = model.src_vocab.stoi[PAD_TOKEN]
    # disable dropout
    model.eval()
    # don't track gradients during validation
    with torch.no_grad():
        all_outputs = []
        valid_attention_scores = []
        total_loss = 0
        total_ntokens = 0
        for valid_batch in iter(valid_iter):
            # run as during training to get validation loss (e.g. xent)

            batch = Batch(valid_batch, pad_index, use_cuda=use_cuda)
            # sort batch now by src length and keep track of order
            sort_reverse_index = batch.sort_by_src_lengths()

            # run as during training with teacher forcing
            if criterion is not None and batch.trg is not None:
                batch_loss = model.get_loss_for_batch(
                    batch, criterion=criterion)
                total_loss += batch_loss
                total_ntokens += batch.ntokens

            # run as during inference to produce translations
            output, attention_scores = model.run_batch(
                batch=batch, beam_size=beam_size, beam_alpha=beam_alpha,
                max_output_length=max_output_length)

            # sort outputs back to original order
            all_outputs.extend(output[sort_reverse_index])
            valid_attention_scores.extend(
                attention_scores[sort_reverse_index]
                if attention_scores is not None else [])

        assert len(all_outputs) == len(data)

        if criterion is not None and total_ntokens > 0:
            # total validation loss
            valid_loss = total_loss
            # exponent of token-level negative log prob
            valid_ppl = torch.exp(total_loss / total_ntokens)
        else:
            valid_loss = -1
            valid_ppl = -1

        # decode back to symbols
        decoded_valid = model.trg_vocab.arrays_to_sentences(arrays=all_outputs,
                                                            cut_at_eos=True)

        # evaluate with metric on full dataset
        join_char = " " if level in ["word", "bpe"] else ""
        valid_sources = [join_char.join(s) for s in data.src]
        valid_references = [join_char.join(t) for t in data.trg]
        valid_hypotheses = [join_char.join(t) for t in decoded_valid]

        # post-process
        if level == "bpe":
            valid_sources = [bpe_postprocess(s) for s in valid_sources]
            valid_references = [bpe_postprocess(v)
                                for v in valid_references]
            valid_hypotheses = [bpe_postprocess(v) for
                                v in valid_hypotheses]

        # if references are given, evaluate against them
        if valid_references:
            assert len(valid_hypotheses) == len(valid_references)

            current_valid_score = 0
            if eval_metric.lower() == 'bleu':
                # this version does not use any tokenization
                current_valid_score = bleu(valid_hypotheses, valid_references)
            elif eval_metric.lower() == 'chrf':
                current_valid_score = chrf(valid_hypotheses, valid_references)
            elif eval_metric.lower() == 'token_accuracy':
                current_valid_score = token_accuracy(
                    valid_hypotheses, valid_references, level=level)
            elif eval_metric.lower() == 'sequence_accuracy':
                current_valid_score = sequence_accuracy(
                    valid_hypotheses, valid_references)
        else:
            current_valid_score = -1

    return current_valid_score, valid_loss, valid_ppl, valid_sources, \
        valid_sources_raw, valid_references, valid_hypotheses, \
        decoded_valid, valid_attention_scores


def test(cfg_file,
         ckpt: str = None,
         output_path: str = None,
         save_attention: bool = False) -> None:
    """
    Main test function. Handles loading a model from checkpoint, generating
    translations and storing them and attention plots.

    :param cfg_file: path to configuration file
    :param ckpt: path to checkpoint to load
    :param output_path: path to output
    :param save_attention: whether to save the computed attention weights
    """

    cfg = load_config(cfg_file)

    if "test" not in cfg["data"].keys():
        raise ValueError("Test data must be specified in config.")

    # when checkpoint is not specified, take oldest from model dir
    if ckpt is None:
        model_dir = cfg["training"]["model_dir"]
        ckpt = get_latest_checkpoint(model_dir)
        try:
            step = ckpt.split(model_dir+"/")[1].split(".ckpt")[0]
        except IndexError:
            step = "best"

    batch_size = cfg["training"]["batch_size"]
    use_cuda = cfg["training"].get("use_cuda", False)
    level = cfg["data"]["level"]
    eval_metric = cfg["training"]["eval_metric"]
    max_output_length = cfg["training"].get("max_output_length", None)

    # load the data
    _, dev_data, test_data, src_vocab, trg_vocab = \
        load_data(cfg=cfg)

    data_to_predict = {"dev": dev_data, "test": test_data}

    # load model state from disk
    model_checkpoint = load_model_from_checkpoint(ckpt, use_cuda=use_cuda)

    # build model and load parameters into it
    model = build_model(cfg["model"], src_vocab=src_vocab, trg_vocab=trg_vocab)
    model.load_state_dict(model_checkpoint["model_state"])

    if use_cuda:
        model.cuda()

    # whether to use beam search for decoding, 0: greedy decoding
    if "testing" in cfg.keys():
        beam_size = cfg["testing"].get("beam_size", 0)
        beam_alpha = cfg["testing"].get("alpha", -1)
    else:
        beam_size = 0
        beam_alpha = -1

    for data_set_name, data_set in data_to_predict.items():

        #pylint: disable=unused-variable
        score, loss, ppl, sources, sources_raw, references, hypotheses, \
        hypotheses_raw, attention_scores = validate_on_data(
            model, data=data_set, batch_size=batch_size, level=level,
            max_output_length=max_output_length, eval_metric=eval_metric,
            use_cuda=use_cuda, criterion=None, beam_size=beam_size,
            beam_alpha=beam_alpha)
        #pylint: enable=unused-variable

        if "trg" in data_set.fields:
            decoding_description = "Greedy decoding" if beam_size == 0 else \
                "Beam search decoding with beam size = {} and alpha = {}".\
                    format(beam_size, beam_alpha)
            print("{:4s} {}: {} [{}]".format(
                data_set_name, eval_metric, score, decoding_description))
        else:
            print("No references given for {} -> no evaluation.".format(
                data_set_name))

        if attention_scores is not None and save_attention:
            attention_path = "{}/{}.{}.att".format(model_dir, data_set_name,
                                                   step)
            print("Attention plots saved to: {}.xx".format(attention_path))
            store_attention_plots(attentions=attention_scores,
                                  targets=hypotheses_raw,
                                  sources=[s for s in data_set.src],
                                  idx=range(len(hypotheses)),
                                  output_prefix=attention_path)

        if output_path is not None:
            output_path_set = "{}.{}".format(output_path, data_set_name)
            with open(output_path_set, mode="w", encoding="utf-8") as out_file:
                for hyp in hypotheses:
                    out_file.write(hyp + "\n")
            print("Translations saved to: {}".format(output_path_set))
