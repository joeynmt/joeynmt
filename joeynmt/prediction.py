# coding: utf-8
"""
This module holds methods for generating predictions from a model.
"""
import logging
import math
import sys
import time
from itertools import zip_longest
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.nn import DataParallel as DP
from torch.utils.data import Dataset
from tqdm import tqdm

from joeynmt.config import BaseConfig, TestConfig, parse_global_args
from joeynmt.data import load_data
from joeynmt.datasets import StreamDataset
from joeynmt.helpers import (
    expand_reverse_index,
    load_checkpoint,
    resolve_ckpt_path,
    save_hypothese,
    set_seed,
    store_attention_plots,
    write_list_to_file,
)
from joeynmt.metrics import bleu, chrf, sequence_accuracy, token_accuracy
from joeynmt.model import Model, _DataParallel, build_model
from joeynmt.search import search

logger = logging.getLogger(__name__)


def predict(
    model: Model,
    data: Dataset,
    device: torch.device,
    n_gpu: int,
    compute_loss: bool = False,
    normalization: str = "batch",
    num_workers: int = 0,
    args: TestConfig = None,
    fp16: bool = False,
) -> Tuple[Dict[str, float], List[str], List[str], List[List[str]], List[np.ndarray],
           List[np.ndarray]]:
    """
    Generates translations for the given data.
    If `compute_loss` is True and references are given, also computes the loss.

    :param model: model module
    :param data: dataset for validation
    :param device: torch device
    :param n_gpu: number of GPUs
    :param compute_loss: whether to computes a scalar loss for given inputs and targets
    :param normalization: one of {`batch`, `tokens`, `none`}
    :param num_workers: number of workers for `collate_fn()` in data iterator
    :param args: configuration args
    :param fp16: whether to use fp16
    :return:
        - valid_scores: (dict) current validation scores,
        - valid_ref: (list) validation references,
        - valid_hyp: (list) validation hypotheses,
        - decoded_valid: (list) token-level validation hypotheses (before post-process),
        - valid_sequence_scores: (list) log probabilities for validation hypotheses
        - valid_attention_scores: (list) attention scores for validation hypotheses
    """
    # pylint: disable=too-many-branches,too-many-statements

    if args.return_prob == "ref":  # no decoding needed
        decoding_description = ""
    else:
        decoding_description = (  # write the decoding strategy in the log
            " (Greedy decoding with " if args.beam_size < 2 else
            f" (Beam search with beam_size={args.beam_size}, "
            f"beam_alpha={args.beam_alpha}, n_best={args.n_best}, ")
        decoding_description += (
            f"min_output_length={args.min_output_length}, "
            f"max_output_length={args.max_output_length}, "
            f"return_prob='{args.return_prob}', generate_unk={args.generate_unk}, "
            f"repetition_penalty={args.repetition_penalty}, "
            f"no_repeat_ngram_size={args.no_repeat_ngram_size})")
    logger.info("Predicting %d example(s)...%s", len(data), decoding_description)

    assert args.batch_size >= n_gpu, "`batch_size` must be bigger than `n_gpu`."
    # **CAUTION:** a batch will be expanded to batch.nseqs * beam_size, and it might
    # cause an out-of-memory error.
    # if batch_size > beam_size:
    #     batch_size //= beam_size

    valid_iter = data.make_iter(
        batch_size=args.batch_size,
        batch_type=args.batch_type,
        shuffle=False,
        num_workers=num_workers,
        pad_index=model.pad_index,
        device=device,
    )

    # disable dropout
    model.eval()

    # place holders for scores
    valid_scores = {"loss": float("nan"), "acc": float("nan"), "ppl": float("nan")}
    all_outputs = []
    valid_attention_scores = []
    valid_sequence_scores = []
    total_loss = 0
    total_nseqs = 0
    total_ntokens = 0
    total_n_correct = 0
    output, ref_scores, hyp_scores, attention_scores = None, None, None, None
    disable_tqdm = isinstance(data, StreamDataset)

    autocast = {"device_type": device.type, "enabled": fp16}
    if fp16:
        autocast["dtype"] = torch.float16 if device.type == "cuda" else torch.bfloat16

    gen_start_time = time.time()
    with tqdm(total=len(data), disable=disable_tqdm, desc="Predicting...") as pbar:
        for batch in valid_iter:
            total_nseqs += batch.nseqs  # number of sentences in the current batch

            # sort batch now by src length and keep track of order
            reverse_index = batch.sort_by_src_length()
            sort_reverse_index = expand_reverse_index(reverse_index, args.n_best)
            batch_size = len(sort_reverse_index)  # = batch.nseqs * args.n_best

            # run as during training to get validation loss (e.g. xent)
            if compute_loss and batch.has_trg:
                assert model.loss_function is not None

                # don't track gradients during validation
                with torch.autocast(**autocast):
                    with torch.no_grad():
                        batch_loss, log_probs, attn, n_correct = model(
                            return_type="loss",
                            return_attention=args.return_attention,
                            **vars(batch))

                # sum over multiple gpus
                batch_loss = batch.normalize(batch_loss, "sum", n_gpu=n_gpu)
                n_correct = batch.normalize(n_correct, "sum", n_gpu=n_gpu)
                if args.return_prob == "ref":
                    ref_scores = batch.score(log_probs)
                    attention_scores = attn.detach().cpu().float().numpy()
                    output = batch.trg

                total_loss += batch_loss.item()  # cast Tensor to float
                total_n_correct += n_correct.item()  # cast Tensor to int
                total_ntokens += batch.ntokens

            # if return_prob == "ref", then no search needed.
            # (just look up the prob of the ground truth.)
            if args.return_prob != "ref":
                # run search as during inference to produce translations
                output, hyp_scores, attention_scores = search(
                    model=model,
                    batch=batch,
                    beam_size=args.beam_size,
                    beam_alpha=args.beam_alpha,
                    max_output_length=args.max_output_length,
                    n_best=args.n_best,
                    return_attention=args.return_attention,
                    return_prob=args.return_prob,
                    generate_unk=args.generate_unk,
                    repetition_penalty=args.repetition_penalty,
                    no_repeat_ngram_size=args.no_repeat_ngram_size,
                    fp16=fp16,
                )

            # sort outputs back to original order
            all_outputs.extend(output[sort_reverse_index])  # either hyp or ref
            valid_attention_scores.extend(attention_scores[sort_reverse_index]
                                          if attention_scores is not None else [])
            valid_sequence_scores.extend(
                ref_scores[sort_reverse_index] \
                if ref_scores is not None and ref_scores.shape[0] == batch_size
                else hyp_scores[sort_reverse_index] \
                if hyp_scores is not None and hyp_scores.shape[0] == batch_size
                else [])

            pbar.update(batch.nseqs)

    gen_duration = time.time() - gen_start_time

    assert total_nseqs == len(data), (total_nseqs, len(data))
    assert len(all_outputs) == len(data) * args.n_best, (len(all_outputs), len(data))

    if compute_loss:
        if normalization == "batch":
            normalizer = total_nseqs
        elif normalization == "tokens":
            normalizer = total_ntokens
        elif normalization == "none":
            normalizer = 1

        # avoid zero division
        assert normalizer > 0
        assert total_ntokens > 0

        # normalized loss
        valid_scores["loss"] = total_loss / normalizer
        # accuracy before decoding
        valid_scores["acc"] = total_n_correct / total_ntokens
        # exponent of token-level negative log likelihood
        valid_scores["ppl"] = math.exp(total_loss / total_ntokens)

    # decode ids back to str symbols (cut-off AFTER eos; eos itself is included.)
    decoded_valid = model.trg_vocab.arrays_to_sentences(arrays=all_outputs,
                                                        cut_at_eos=True)
    # TODO: `valid_sequence_scores` should have the same seq length as `decoded_valid`
    #     -> needed to be cut-off at eos synchronously

    if args.return_prob == "ref":  # no evaluation needed
        logger.info(
            "Evaluation result (scoring) %s, duration: %.4f[sec]",
            ", ".join([
                f"{eval_metric}: {valid_scores[eval_metric]:6.2f}"
                for eval_metric in ["loss", "ppl", "acc"]
            ]),
            gen_duration,
        )
        return (
            valid_scores,
            None,  # valid_ref
            None,  # valid_hyp
            decoded_valid,
            valid_sequence_scores,
            valid_attention_scores,
        )

    # retrieve detokenized hypotheses and references
    valid_hyp = [
        data.tokenizer[data.trg_lang].post_process(s, generate_unk=args.generate_unk)
        for s in decoded_valid
    ]
    # references are not length-filtered, not duplicated for n_best > 1
    valid_ref = [data.tokenizer[data.trg_lang].post_process(s) for s in data.trg]

    # if references are given, evaluate 1best generation against them
    if data.has_trg:
        valid_hyp_1best = (valid_hyp if args.n_best == 1 else [
            valid_hyp[i] for i in range(0, len(valid_hyp), args.n_best)
        ])
        assert len(valid_hyp_1best) == len(valid_ref), (valid_hyp_1best, valid_ref)

        eval_start_time = time.time()

        # evaluate with metrics on full dataset
        for eval_metric in args.eval_metrics:
            if eval_metric == "bleu":
                valid_scores[eval_metric] = bleu(
                    valid_hyp_1best,
                    valid_ref,  # detokenized ref
                    **args.sacrebleu_cfg,
                )
            elif eval_metric == "chrf":
                valid_scores[eval_metric] = chrf(
                    valid_hyp_1best,
                    valid_ref,  # detokenized ref
                    **args.sacrebleu_cfg,
                )
            elif eval_metric == "token_accuracy":
                decoded_valid_1best = (decoded_valid if args.n_best == 1 else [
                    decoded_valid[i] for i in range(0, len(decoded_valid), args.n_best)
                ])
                valid_scores[eval_metric] = token_accuracy(
                    decoded_valid_1best,
                    data.get_list(lang=data.trg_lang, tokenized=True),  # tokenized ref
                )
            elif eval_metric == "sequence_accuracy":
                valid_scores[eval_metric] = sequence_accuracy(
                    valid_hyp_1best, valid_ref)

        eval_duration = time.time() - eval_start_time
        score_str = ", ".join([
            f"{eval_metric}: {valid_scores[eval_metric]:6.2f}"
            for eval_metric in args.eval_metrics + ["loss", "ppl", "acc"]
            if not math.isnan(valid_scores[eval_metric])
        ])
        logger.info(
            "Evaluation result (%s) %s, generation: %.4f[sec], evaluation: %.4f[sec]",
            "beam search" if args.beam_size > 1 else "greedy",
            score_str,
            gen_duration,
            eval_duration,
        )
    else:
        logger.info("Generation took %.4f[sec]. (No references given)", gen_duration)

    return (
        valid_scores,
        valid_ref,
        valid_hyp,
        decoded_valid,
        valid_sequence_scores,
        valid_attention_scores,
    )


def prepare(args: BaseConfig, mode: str) -> Tuple[Model, Dataset, Dataset, Dataset]:
    """
    Helper function for model and data loading.

    :param args: config args
    :param mode: execution mode
    """
    # load the data
    if mode == "train":
        datasets = ["train", "dev", "test"]
    if mode == "test":
        datasets = ["dev", "test"]
    if mode == "translate":
        datasets = ["stream"]

    if mode != "train":
        if "voc_file" not in args.data["src"] or not args.data["src"]["voc_file"]:
            args.data["src"]["voc_file"] = (args.model_dir / "src_vocab.txt").as_posix()
        if "voc_file" not in args.data["trg"] or not args.data["trg"]["voc_file"]:
            args.data["trg"]["voc_file"] = (args.model_dir / "trg_vocab.txt").as_posix()

    src_vocab, trg_vocab, train_data, dev_data, test_data = load_data(cfg=args.data,
                                                                      datasets=datasets)

    if mode == "train":
        # store the vocabs and tokenizers
        src_vocab.to_file(args.model_dir / "src_vocab.txt")
        if hasattr(train_data.tokenizer[train_data.src_lang], "copy_cfg_file"):
            train_data.tokenizer[train_data.src_lang].copy_cfg_file(args.model_dir)
        trg_vocab.to_file(args.model_dir / "trg_vocab.txt")
        if hasattr(train_data.tokenizer[train_data.trg_lang], "copy_cfg_file"):
            train_data.tokenizer[train_data.trg_lang].copy_cfg_file(args.model_dir)

    # build an encoder-decoder model
    model = build_model(args.model, src_vocab=src_vocab, trg_vocab=trg_vocab)
    model.log_parameters_list()
    # need to instantiate loss func after `build_model()`
    model.loss_function = (args.train.loss, args.train.label_smoothing)
    logger.info(model)

    if mode != "train":
        # when checkpoint is not specified, take latest (best) from model dir
        ckpt = resolve_ckpt_path(args.test.load_model, args.model_dir)

        # load model checkpoint
        logger.info("Loading model from %s", ckpt)
        model_checkpoint = load_checkpoint(ckpt, device=args.device)

        # restore model and optimizer parameters
        model.load_state_dict(model_checkpoint["model_state"])

    # CPU / GPU
    if args.device.type == "cuda":
        model.to(args.device)

    if args.n_gpu > 1:  # multi gpu training
        model = _DataParallel(DP(model))

    # set the random seed
    set_seed(seed=args.seed)

    return model, train_data, dev_data, test_data


def test(
    cfg: Dict,
    output_path: str = None,
    prepared: Dict = None,
    save_attention: bool = False,
    save_scores: bool = False,
) -> None:
    """
    Main test function. Handles loading a model from checkpoint, generating
    translations, storing them, and plotting attention.

    :param cfg: configuration dict
    :param output_path: path to output
    :param prepared: model and datasets passed from training
    :param save_attention: whether to save attention visualizations
    :param save_scores: whether to save scores
    """

    # parse args
    args = parse_global_args(cfg, mode="test")

    # load the data
    if prepared is None:
        model, _, dev_data, test_data = prepare(args, mode="test")
        data_to_predict = {"dev": dev_data, "test": test_data}

    else:  # avoid to load model and data again
        model = prepared["model"]
        data_to_predict = {"dev": prepared["dev"], "test": prepared["test"]}

    # check options
    if save_attention:
        if cfg["model"]["decoder"]["type"] == "transformer":
            assert cfg["testing"].get("beam_size", 1) == 1, (
                "Attention plots can be saved with greedy decoding only. Please set "
                "`beam_size: 1` in the config.")
        args = args._replace(test=args.test._replace(return_attention=True))
    if save_scores:
        assert output_path, "Please specify --output_path for saving scores."
        if args.test.return_prob == "none":
            logger.warning("Please specify prob type: {`ref` or `hyp`} in the config. "
                           "Scores will not be saved.")
            save_scores = False
        elif args.test.return_prob == "ref":
            assert cfg["testing"].get("beam_size", 1) == 1, (
                "Scores of given references can be computed with greedy decoding only. "
                "Please set `beam_size: 1` in the config.")

    # pediction loop over datasets
    for data_set_name, data_set in data_to_predict.items():
        if data_set is not None:
            data_set.reset_random_subset()  # no subsampling in evaluation

            logger.info(
                "%s on %s set...",
                "Scoring" if args.test.return_prob == "ref" else "Decoding",
                data_set_name,
            )
            _, _, hypotheses, hypotheses_raw, seq_scores, att_scores, = predict(
                model=model,
                data=data_set,
                compute_loss=args.test.return_prob == "ref",
                device=args.device,
                n_gpu=args.n_gpu,
                num_workers=args.num_workers,
                normalization=args.train.normalization,
                args=args.test,
                fp16=args.fp16,
            )

            if save_attention:
                if att_scores:
                    attention_file_name = f"{output_path}.{data_set_name}.att"
                    logger.info("Saving attention plots. This might take a while..")
                    store_attention_plots(
                        attentions=att_scores,
                        targets=hypotheses_raw,
                        sources=data_set.get_list(lang=data_set.src_lang,
                                                  tokenized=True),
                        indices=range(len(hypotheses)),
                        output_prefix=attention_file_name,
                    )
                    logger.info("Attention plots saved to: %s", attention_file_name)
                else:
                    logger.warning(
                        "Attention scores could not be saved. Note that attention "
                        "scores are not available when using beam search. "
                        "Set beam_size to 1 for greedy decoding.")

            if output_path is not None:
                if save_scores and seq_scores is not None:
                    # save scores
                    output_path_scores = Path(f"{output_path}.{data_set_name}.scores")
                    write_list_to_file(output_path_scores, seq_scores)
                    # save tokens
                    output_path_tokens = Path(f"{output_path}.{data_set_name}.tokens")
                    write_list_to_file(output_path_tokens, hypotheses_raw)
                    logger.info(
                        "Scores and corresponding tokens saved to: %s.{scores|tokens}",
                        f"{output_path}.{data_set_name}",
                    )
                if hypotheses is not None:
                    # save translations
                    output_path_set = Path(f"{output_path}.{data_set_name}")
                    save_hypothese(output_path_set, hypotheses, args.test.n_best)
                    logger.info("Translations saved to: %s.", output_path_set)


def translate(
    cfg: Dict,
    output_path: str = None,
) -> None:
    """
    Interactive translation function.
    Loads model from checkpoint and translates either the stdin input or asks for
    input to translate interactively. Translations and scores are printed to stdout.
    Note: The input sentences don't have to be pre-tokenized.

    :param cfg: configuration dict
    :param output_path: path to output file
    """

    # parse args
    args = parse_global_args(cfg)
    model, _, _, test_data = prepare(args, mode="translate")
    assert isinstance(test_data, StreamDataset)

    def _translate_data(test_data, args):
        """Translates given dataset, using parameters from outer scope."""
        _, _, hypotheses, trg_tokens, trg_scores, _ = predict(
            model=model,
            data=test_data,
            compute_loss=False,
            device=args.device,
            n_gpu=args.n_gpu,
            normalization="none",
            num_workers=args.num_workers,
            args=args.test,
            fp16=args.fp16,
        )
        return hypotheses, trg_tokens, trg_scores

    if not sys.stdin.isatty():  # pylint: disable=too-many-nested-blocks
        # input stream given
        for i, line in enumerate(sys.stdin.readlines()):
            if not line.strip():
                # skip empty lines and print warning
                logger.warning("The sentence in line %d is empty. Skip to load.", i)
                continue
            test_data.set_item(line.rstrip())
        all_hypotheses, tokens, scores = _translate_data(test_data, args)
        assert len(all_hypotheses) == len(test_data) * args.test.n_best

        if output_path is not None:
            # write to outputfile if given
            out_file = Path(output_path).expanduser()
            save_hypothese(out_file, all_hypotheses, args.n_best)
            logger.info("Translations saved to: %s.", out_file)

        else:
            # print to stdout
            for hyp in all_hypotheses:
                print(hyp)

    else:
        # CAUTION: this will raise an error if n_gpus > 1
        args = args._replace(
            test=args.test._replace(batch_size=1, batch_type="sentence"))
        # enter interactive mode
        np.set_printoptions(linewidth=sys.maxsize)  # for printing scores in stdout
        while True:
            try:
                src_input = input("\nPlease enter a source sentence:\n")
                if not src_input.strip():
                    break

                # every line has to be made into dataset
                test_data.set_item(src_input.rstrip())
                hypotheses, tokens, scores = _translate_data(test_data, args)

                print("JoeyNMT:")
                for i, (hyp, token,
                        score) in enumerate(zip_longest(hypotheses, tokens, scores)):
                    assert hyp is not None, (i, hyp, token, score)
                    print(f"#{i + 1}: {hyp}")
                    if args.test.return_prob in ["hyp"]:
                        if args.test.beam_size > 1:  # beam search: seq-level scores
                            print(f"\ttokens: {token}\n\tsequence score: {score[0]}")
                        else:  # greedy: token-level scores
                            assert len(token) == len(score), (token, score)
                            print(f"\ttokens: {token}\n\tscores: {score}")

                # reset cache
                test_data.cache = {}

            except (KeyboardInterrupt, EOFError):
                print("\nBye.")
                break
