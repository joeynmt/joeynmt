# coding: utf-8
"""
This modules holds methods for generating predictions from a model.
"""
import logging
import math
import sys
import time
from functools import partial
from itertools import zip_longest
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from joeynmt.data import load_data, make_data_iter
from joeynmt.datasets import build_dataset
from joeynmt.helpers import (
    expand_reverse_index,
    load_checkpoint,
    load_config,
    make_logger,
    parse_test_args,
    parse_train_args,
    resolve_ckpt_path,
    set_seed,
    store_attention_plots,
    write_list_to_file,
)
from joeynmt.metrics import bleu, chrf, sequence_accuracy, token_accuracy
from joeynmt.model import Model, _DataParallel, build_model
from joeynmt.search import search
from joeynmt.tokenizers import build_tokenizer
from joeynmt.vocabulary import build_vocab

logger = logging.getLogger(__name__)


def predict(
    model: Model,
    data: Dataset,
    device: torch.device,
    n_gpu: int,
    compute_loss: bool = False,
    normalization: str = "batch",
    num_workers: int = 0,
    cfg: Dict = None,
) -> Tuple[Dict[str, float], List[str], List[str], List[List[str]], List[np.ndarray],
           List[np.ndarray], ]:
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
    :param cfg: `testing` section in yaml config file
    :return:
        - valid_scores: (dict) current validation scores,
        - valid_ref: (list) validation references,
        - valid_hyp: (list) validation hypotheses,
        - decoded_valid: (list) token-level validation hypotheses (before post-process),
        - valid_sequence_scores: (list) log probabilities for validation hypotheses
        - valid_attention_scores: (list) attention scores for validation hypotheses
    """
    # pylint: disable=too-many-branches,too-many-statements
    # parse test cfg
    (
        eval_batch_size,
        eval_batch_type,
        max_output_length,
        min_output_length,
        eval_metrics,
        sacrebleu_cfg,
        beam_size,
        beam_alpha,
        n_best,
        return_attention,
        return_prob,
        generate_unk,
        repetition_penalty,
        no_repeat_ngram_size,
    ) = parse_test_args(cfg)

    if return_prob == "ref":  # no decoding needed
        decoding_description = ""
    else:
        decoding_description = (  # write the decoding strategy in the log
            " (Greedy decoding with " if beam_size < 2 else
            f" (Beam search with beam_size={beam_size}, beam_alpha={beam_alpha}, "
            f"n_best={n_best}, ")
        decoding_description += (
            f"min_output_length={min_output_length}, "
            f"max_output_length={max_output_length}, "
            f"return_prob='{return_prob}', generate_unk={generate_unk}, "
            f"repetition_penalty={repetition_penalty}, "
            f"no_repeat_ngram_size={no_repeat_ngram_size})")
    logger.info("Predicting %d example(s)...%s", len(data), decoding_description)

    assert eval_batch_size >= n_gpu, "`batch_size` must be bigger than `n_gpu`."
    # **CAUTION:** a batch will be expanded to batch.nseqs * beam_size, and it might
    # cause an out-of-memory error.
    # if batch_size > beam_size:
    #     batch_size //= beam_size

    valid_iter = make_data_iter(
        dataset=data,
        batch_size=eval_batch_size,
        batch_type=eval_batch_type,
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

    gen_start_time = time.time()
    for batch in valid_iter:
        total_nseqs += batch.nseqs  # number of sentences in the current batch

        # sort batch now by src length and keep track of order
        reverse_index = batch.sort_by_src_length()
        sort_reverse_index = expand_reverse_index(reverse_index, n_best)

        # run as during training to get validation loss (e.g. xent)
        if compute_loss and batch.has_trg:
            assert model.loss_function is not None

            # don't track gradients during validation
            with torch.no_grad():
                batch_loss, log_probs, _, n_correct = model(return_type="loss",
                                                            **vars(batch))
                # sum over multiple gpus
                batch_loss = batch.normalize(batch_loss, "sum", n_gpu=n_gpu)
                n_correct = batch.normalize(n_correct, "sum", n_gpu=n_gpu)
                if return_prob == "ref":
                    ref_scores = batch.score(log_probs)
                    output = batch.trg

            total_loss += batch_loss.item()  # cast Tensor to float
            total_n_correct += n_correct.item()  # cast Tensor to int
            total_ntokens += batch.ntokens

        # if return_prob == "ref", then no search needed.
        # (just look up the prob of the ground truth.)
        if return_prob != "ref":
            # run search as during inference to produce translations
            output, hyp_scores, attention_scores = search(
                model=model,
                batch=batch,
                beam_size=beam_size,
                beam_alpha=beam_alpha,
                max_output_length=max_output_length,
                n_best=n_best,
                return_attention=return_attention,
                return_prob=return_prob,
                generate_unk=generate_unk,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
            )

        # sort outputs back to original order
        all_outputs.extend(output[sort_reverse_index])  # either hyp or ref
        valid_attention_scores.extend(attention_scores[sort_reverse_index]
                                      if attention_scores is not None else [])
        valid_sequence_scores.extend(
            ref_scores[sort_reverse_index] if ref_scores is not None
            and ref_scores.shape[0] == len(sort_reverse_index) else
            hyp_scores[sort_reverse_index] if hyp_scores is not None and hyp_scores.
            shape[0] == len(sort_reverse_index) else [])
    gen_duration = time.time() - gen_start_time

    assert total_nseqs == len(data), (total_nseqs, len(data))
    assert len(all_outputs) == len(data) * n_best, (len(all_outputs), len(data), n_best)

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

    if return_prob == "ref":  # no evaluation needed
        logger.info(
            "Evaluation result (scoring) %s, duration: %.4f[sec]",
            ", ".join([
                f"{eval_metric}: {valid_scores[eval_metric]:6.2f}"
                for eval_metric in ["loss", "ppl", "acc"]
            ]),
            gen_duration,
        )
        return valid_scores, None, None, decoded_valid, valid_sequence_scores, None

    # retrieve detokenized hypotheses and references
    valid_hyp = [
        data.tokenizer[data.trg_lang].post_process(s, generate_unk=generate_unk)
        for s in decoded_valid
    ]
    valid_ref = data.trg  # not length-filtered, not duplicated for n_best > 1

    # if references are given, evaluate 1best generation against them
    if data.has_trg:
        valid_hyp_1best = (valid_hyp if n_best == 1 else
                           [valid_hyp[i] for i in range(0, len(valid_hyp), n_best)])
        assert len(valid_hyp_1best) == len(valid_ref), (valid_hyp_1best, valid_ref)

        eval_start_time = time.time()

        # evaluate with metrics on full dataset
        for eval_metric in eval_metrics:
            if eval_metric == "bleu":
                valid_scores[eval_metric] = bleu(
                    valid_hyp_1best,
                    valid_ref,  # detokenized ref
                    **sacrebleu_cfg,
                )
            elif eval_metric == "chrf":
                valid_scores[eval_metric] = chrf(
                    valid_hyp_1best,
                    valid_ref,  # detokenized ref
                    **sacrebleu_cfg,
                )
            elif eval_metric == "token_accuracy":
                decoded_valid_1best = (decoded_valid if n_best == 1 else [
                    decoded_valid[i] for i in range(0, len(decoded_valid), n_best)
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
            for eval_metric in eval_metrics + ["loss", "ppl", "acc"]
            if not math.isnan(valid_scores[eval_metric])
        ])
        logger.info(
            "Evaluation result (%s) %s, generation: %.4f[sec], evaluation: %.4f[sec]",
            "beam search" if beam_size > 1 else "greedy",
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


def test(
    cfg_file,
    ckpt: str,
    output_path: str = None,
    datasets: dict = None,
    save_attention: bool = False,
    save_scores: bool = False,
) -> None:
    """
    Main test function. Handles loading a model from checkpoint, generating
    translations, storing them, and plotting attention.

    :param cfg_file: path to configuration file
    :param ckpt: path to checkpoint to load
    :param output_path: path to output
    :param datasets: datasets to predict
    :param save_attention: whether to save attention visualizations
    :param save_scores: whether to save scores
    """
    # pylint: disable=too-many-branches
    cfg = load_config(Path(cfg_file))
    # parse train cfg
    model_dir, load_model, device, n_gpu, num_workers, normalization = parse_train_args(
        cfg["training"], mode="prediction")

    if len(logger.handlers) == 0:
        _ = make_logger(model_dir, mode="test")  # version string returned

    # load the data
    if datasets is None:
        src_vocab, trg_vocab, _, dev_data, test_data = load_data(
            data_cfg=cfg["data"], datasets=["dev", "test"])
        data_to_predict = {"dev": dev_data, "test": test_data}
    else:  # avoid to load data again
        data_to_predict = {"dev": datasets["dev"], "test": datasets["test"]}
        src_vocab = datasets["src_vocab"]
        trg_vocab = datasets["trg_vocab"]

    # build model and load parameters into it
    model = build_model(cfg["model"], src_vocab=src_vocab, trg_vocab=trg_vocab)
    model.log_parameters_list()

    # check options
    if save_attention:
        if cfg["model"]["decoder"]["type"] == "transformer":
            assert cfg["testing"].get("beam_size", 1) == 1, (
                "Attention plots can be saved with greedy decoding only. Please set "
                "`beam_size: 1` in the config.")
        cfg["testing"]["save_attention"] = True
    return_prob = cfg["testing"].get("return_prob", "none")
    if save_scores:
        assert output_path, "Please specify --output_path for saving scores."
        if return_prob == "none":
            logger.warning("Please specify prob type: {`ref` or `hyp`} in the config. "
                           "Scores will not be saved.")
            save_scores = False
        elif return_prob == "ref":
            assert cfg["testing"].get("beam_size", 1) == 1, (
                "Scores of given references can be computed with greedy decoding only."
                "Please set `beam_size: 1` in the config.")
            model.loss_function = (  # need to instantiate loss func to compute scores
                cfg["training"].get("loss_type", "crossentropy"),
                cfg["training"].get("label_smoothing", 0.1),
            )

    # when checkpoint is not specified, take latest (best) from model dir
    ckpt = resolve_ckpt_path(ckpt, load_model, model_dir)

    # load model checkpoint
    model_checkpoint = load_checkpoint(ckpt, device=device)

    # restore model and optimizer parameters
    model.load_state_dict(model_checkpoint["model_state"])
    if device.type == "cuda":
        model.to(device)

    # multi-gpu eval
    if n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
        model = _DataParallel(model)

    # set the random seed
    set_seed(seed=cfg["training"].get("random_seed", 42))

    for data_set_name, data_set in data_to_predict.items():
        if data_set is None:
            continue

        logger.info(
            "%s on %s set...",
            "Scoring" if return_prob == "ref" else "Decoding",
            data_set_name,
        )
        _, _, hypotheses, hypotheses_raw, sequence_scores, attention_scores, = predict(
            model=model,
            data=data_set,
            compute_loss=save_scores,
            device=device,
            n_gpu=n_gpu,
            num_workers=num_workers,
            normalization=normalization,
            cfg=cfg["testing"],
        )

        if save_attention:
            if attention_scores:
                attention_file_name = f"{data_set_name}.{ckpt.stem}.att"
                attention_file_path = (model_dir / attention_file_name).as_posix()
                logger.info("Saving attention plots. This might take a while..")
                store_attention_plots(
                    attentions=attention_scores,
                    targets=hypotheses_raw,
                    sources=data_set.get_list(lang=data_set.src_lang, tokenized=True),
                    indices=range(len(hypotheses)),
                    output_prefix=attention_file_path,
                )
                logger.info("Attention plots saved to: %s", attention_file_path)
            else:
                logger.warning(
                    "Attention scores could not be saved. Note that attention scores "
                    "are not available when using beam search. Set beam_size to 1 for "
                    "greedy decoding.")

        if output_path is not None:
            if sequence_scores is not None and save_scores:
                # save scores
                output_path_scores = Path(f"{output_path}.{data_set_name}.scores")
                write_list_to_file(output_path_scores, sequence_scores)
                # save tokens
                output_path_tokens = Path(f"{output_path}.{data_set_name}.tokens")
                write_list_to_file(output_path_tokens, hypotheses_raw)
                logger.info(
                    "Scores and corresponding tokens saved to: %s.{scores|tokens}",
                    f"{output_path}.{data_set_name}",
                )
            if hypotheses is not None:
                output_path_set = Path(f"{output_path}.{data_set_name}")
                write_list_to_file(output_path_set, hypotheses)
                logger.info("Translations saved to: %s.", output_path_set)


def translate(
    cfg_file: str,
    ckpt: str = None,
    output_path: str = None,
) -> None:
    """
    Interactive translation function.
    Loads model from checkpoint and translates either the stdin input or asks for
    input to translate interactively. Translations and scores are printed to stdout.
    Note: The input sentences don't have to be pre-tokenized.

    :param cfg_file: path to configuration file
    :param ckpt: path to checkpoint to load
    :param output_path: path to output file
    """

    # pylint: disable=too-many-branches
    def _translate_data(test_data, cfg):
        """Translates given dataset, using parameters from outer scope."""
        _, _, hypotheses, trg_tokens, trg_scores, _ = predict(
            model=model,
            data=test_data,
            compute_loss=False,
            device=device,
            n_gpu=n_gpu,
            normalization="none",
            num_workers=num_workers,
            cfg=cfg,
        )
        return hypotheses, trg_tokens, trg_scores

    cfg = load_config(Path(cfg_file))
    # parse and validate cfg
    model_dir, load_model, device, n_gpu, num_workers, _ = parse_train_args(
        cfg["training"], mode="prediction")
    test_cfg = cfg["testing"]
    src_cfg = cfg["data"]["src"]
    trg_cfg = cfg["data"]["trg"]

    _ = make_logger(model_dir, mode="translate")
    # version string returned

    # when checkpoint is not specified, take latest (best) from model dir
    ckpt = resolve_ckpt_path(ckpt, load_model, model_dir)

    # read vocabs
    src_vocab, trg_vocab = build_vocab(cfg["data"], model_dir=model_dir)

    # build model and load parameters into it
    model = build_model(cfg["model"], src_vocab=src_vocab, trg_vocab=trg_vocab)

    # load model state from disk
    model_checkpoint = load_checkpoint(ckpt, device=device)
    model.load_state_dict(model_checkpoint["model_state"])

    if device.type == "cuda":
        model.to(device)

    tokenizer = build_tokenizer(cfg["data"])
    sequence_encoder = {
        src_cfg["lang"]: partial(src_vocab.sentences_to_ids, bos=False, eos=True),
        trg_cfg["lang"]: None,
    }
    test_data = build_dataset(
        dataset_type="stream",
        path=None,
        src_lang=src_cfg["lang"],
        trg_lang=trg_cfg["lang"],
        split="test",
        tokenizer=tokenizer,
        sequence_encoder=sequence_encoder,
    )

    # set the random seed
    set_seed(seed=cfg["training"].get("random_seed", 42))

    n_best = test_cfg.get("n_best", 1)
    beam_size = test_cfg.get("beam_size", 1)
    return_prob = test_cfg.get("return_prob", "none")
    if not sys.stdin.isatty():  # pylint: disable=too-many-nested-blocks
        # input stream given
        for line in sys.stdin.readlines():
            test_data.set_item(line.rstrip())
        all_hypotheses, tokens, scores = _translate_data(test_data, test_cfg)
        assert len(all_hypotheses) == len(test_data) * n_best

        if output_path is not None:
            # write to outputfile if given
            out_file = Path(output_path).expanduser()

            if n_best > 1:
                for n in range(n_best):
                    write_list_to_file(
                        out_file.parent / f"{out_file.stem}-{n}.{out_file.suffix}",
                        [
                            all_hypotheses[i]
                            for i in range(n, len(all_hypotheses), n_best)
                        ],
                    )
            else:
                write_list_to_file(out_file, all_hypotheses)

            logger.info("Translations saved to: %s.", out_file)

        else:
            # print to stdout
            for hyp in all_hypotheses:
                print(hyp)

    else:
        # enter interactive mode
        test_cfg["batch_size"] = 1  # CAUTION: this will raise an error if n_gpus > 1
        test_cfg["batch_type"] = "sentence"
        np.set_printoptions(linewidth=sys.maxsize)  # for printing scores in stdout
        while True:
            try:
                src_input = input("\nPlease enter a source sentence:\n")
                if not src_input.strip():
                    break

                # every line has to be made into dataset
                test_data.set_item(src_input.rstrip())
                hypotheses, tokens, scores = _translate_data(test_data, test_cfg)

                print("JoeyNMT:")
                for i, (hyp, token,
                        score) in enumerate(zip_longest(hypotheses, tokens, scores)):
                    assert hyp is not None, (i, hyp, token, score)
                    print(f"#{i + 1}: {hyp}")
                    if return_prob in ["hyp"]:
                        if beam_size > 1:  # beam search: sequence-level scores
                            print(f"\ttokens: {token}\n\tsequence score: {score[0]}")
                        else:  # greedy: token-level scores
                            assert len(token) == len(score), (token, score)
                            print(f"\ttokens: {token}\n\tscores: {score}")

                # reset cache
                test_data.cache = {}

            except (KeyboardInterrupt, EOFError):
                print("\nBye.")
                break
