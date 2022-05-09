# coding: utf-8
"""
This modules holds methods for generating predictions from a model.
"""
import logging
import math
import sys
import time
from functools import partial
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from joeynmt.data import load_data, make_data_iter
from joeynmt.datasets import build_dataset
from joeynmt.helpers import (
    ConfigurationError,
    expand_reverse_index,
    load_checkpoint,
    load_config,
    make_logger,
    parse_train_args,
    resolve_ckpt_path,
    store_attention_plots,
    write_list_to_file,
)
from joeynmt.metrics import bleu, chrf, sequence_accuracy, token_accuracy
from joeynmt.model import Model, _DataParallel, build_model
from joeynmt.search import run_batch
from joeynmt.tokenizers import build_tokenizer
from joeynmt.vocabulary import build_vocab


logger = logging.getLogger(__name__)


def parse_test_args(cfg: Dict) -> Tuple:
    """Parse test args"""
    # batch options
    batch_size: int = cfg.get("batch_size", 64)
    batch_type: str = cfg.get("batch_type", "sentences")
    if batch_type not in ["sentence", "token"]:
        raise ConfigurationError(
            "Invalid `batch_type` option. Valid options: {`sentence`, `token`}."
        )

    # limit on generation length
    max_output_length = cfg.get("max_output_length", None)

    # eval metrics
    if "eval_metrics" in cfg:
        eval_metrics = [s.strip().lower() for s in cfg["eval_metrics"].split(",")]
    elif "eval_metric" in cfg:
        eval_metrics = [cfg["eval_metric"].strip().lower()]
        logger.warning(
            "`eval_metric` option is obsolete. Please use `eval_metrics`, instead."
        )
    else:
        eval_metrics = []
    for eval_metric in eval_metrics:
        if eval_metric not in ["bleu", "chrf", "token_accuracy", "sequence_accuracy"]:
            raise ConfigurationError(
                "Invalid setting for `eval_metrics`. "
                "Valid options: 'bleu', 'chrf', 'token_accuracy', 'sequence_accuracy'."
            )

    # sacrebleu cfg
    sacrebleu: Dict = cfg.get(
        "sacrebleu", {"remove_whitespace": True, "tokenize": "13a"}
    )

    # beam search options
    beam_size: int = cfg.get("beam_size", 1)
    beam_alpha: float = cfg.get("beam_alpha", -1)
    n_best: int = cfg.get("n_best", 1)
    assert beam_size > 0, "Beam size must be >0."
    assert n_best > 0, "N-best size must be >0."
    assert n_best <= beam_size, "`n_best` must be smaller than or equal to `beam_size`."

    return (
        batch_size,
        batch_type,
        max_output_length,
        eval_metrics,
        sacrebleu,
        beam_size,
        beam_alpha,
        n_best,
    )


def validate_on_data(
    model: Model,
    data: Dataset,
    device: torch.device,
    n_gpu: int,
    compute_loss: bool = False,
    normalization: str = "batch",
    num_workers: int = 0,
    cfg: Dict = None,
) -> Tuple[Dict[str, float], List[str], List[str], List[List[str]], List[np.ndarray]]:
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
        - valid_ref: validation references,
        - valid_hyp: validation hypotheses,
        - decoded_valid: token-level validation hypotheses (before post-processing),
        - valid_attention_scores: attention scores for validation hypotheses
    """
    # parse test cfg
    (
        eval_batch_size,
        eval_batch_type,
        max_output_length,
        eval_metrics,
        sacrebleu,
        beam_size,
        beam_alpha,
        n_best,
    ) = parse_test_args(cfg)

    decoding_description = (
        "Greedy decoding"
        if beam_size < 2
        else f"Beam search decoding with beam size={beam_size}, alpha={beam_alpha}"
    )
    logger.info("Validating on %d data points... (%s)", len(data), decoding_description)

    assert eval_batch_size >= n_gpu, "batch_size must be bigger than n_gpu."
    if eval_batch_size > 1000 and eval_batch_type == "sentence":
        logger.warning(
            "WARNING: Are you sure you meant to work on huge batches like this? "
            "'batch_size' is > 1000 for sentence-batching. Consider decreasing it "
            "or switching to 'batch_type: token'."
        )
    # CAUTION: a batch will be expanded to batch.nseqs * beam_size, and it might cause
    # an out-of-memory error.
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
    total_loss = 0
    total_nseqs = 0
    total_ntokens = 0
    total_n_correct = 0

    gen_start_time = time.time()
    for batch in valid_iter:
        total_nseqs += batch.nseqs  # number of sentences in the current batch

        # sort batch now by src length and keep track of order
        reverse_index = batch.sort_by_src_length()
        sort_reverse_index = expand_reverse_index(reverse_index, n_best)

        # run as during training to get validation loss (e.g. xent)
        if compute_loss and batch.has_trg:
            # don't track gradients during validation
            with torch.no_grad():
                batch_loss, _, _, n_correct = model(return_type="loss", **vars(batch))
                # sum over multiple gpus
                batch_loss = batch.normalize(batch_loss, "sum", n_gpu=n_gpu)
                n_correct = batch.normalize(n_correct, "sum", n_gpu=n_gpu)

            total_loss += batch_loss.item()  # float <- Tensor
            total_n_correct += n_correct.item()
            total_ntokens += batch.ntokens

        # run as during inference to produce translations
        output, attention_scores = run_batch(
            model=model,
            batch=batch,
            beam_size=beam_size,
            beam_alpha=beam_alpha,
            max_output_length=max_output_length,
            n_best=n_best,
            generate_unk=False,
        )

        # sort outputs back to original order
        all_outputs.extend(output[sort_reverse_index])
        valid_attention_scores.extend(
            attention_scores[sort_reverse_index] if attention_scores is not None else []
        )
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

    # decode back to symbols
    decoded_valid = model.trg_vocab.arrays_to_sentences(
        arrays=all_outputs, cut_at_eos=True
    )

    # retrieve detokenized hypotheses and references
    valid_hyp = [data.tokenizer[data.trg_lang].post_process(s) for s in decoded_valid]
    valid_ref = data.trg  # not length-filtered, not duplicated for n_best > 1

    # if references are given, evaluate 1best generation against them
    if data.has_trg:
        valid_hyp_1best = (
            valid_hyp
            if n_best == 1
            else [valid_hyp[i] for i in range(0, len(valid_hyp), n_best)]
        )
        assert len(valid_hyp_1best) == len(valid_ref), (valid_hyp_1best, valid_ref)

        eval_start_time = time.time()

        # evaluate with metric on full dataset
        for eval_metric in eval_metrics:
            if eval_metric == "bleu":
                valid_scores[eval_metric] = bleu(
                    valid_hyp_1best,
                    valid_ref,  # detokenized ref
                    tokenize=sacrebleu["tokenize"],
                )
            elif eval_metric == "chrf":
                valid_scores[eval_metric] = chrf(
                    valid_hyp_1best,
                    valid_ref,  # detokenized ref
                    remove_whitespace=sacrebleu["remove_whitespace"],
                )
            elif eval_metric == "token_accuracy":
                decoded_valid_1best = (
                    decoded_valid
                    if n_best == 1
                    else [
                        decoded_valid[i] for i in range(0, len(decoded_valid), n_best)
                    ]
                )
                valid_scores[eval_metric] = token_accuracy(
                    decoded_valid_1best,
                    data.get_list(lang=data.trg_lang, tokenized=True),  # tokenized ref
                )
            elif eval_metric == "sequence_accuracy":
                valid_scores[eval_metric] = sequence_accuracy(
                    valid_hyp_1best, valid_ref
                )

        eval_duration = time.time() - eval_start_time
        score_str = ", ".join(
            [
                f"{eval_metric}: {valid_scores[eval_metric]:6.2f}"
                for eval_metric in eval_metrics + ["loss", "ppl", "acc"]
                if not math.isnan(valid_scores[eval_metric])
            ]
        )
        logger.info(
            "Validation result (%s) %s, generation: %.4fs[sec], evaluation: %.4fs[sec]",
            "beam search" if beam_size > 1 else "greedy",
            score_str,
            gen_duration,
            eval_duration,
        )
    else:
        logger.info("Generation took %.4fs[sec]. (No references given)", gen_duration)

    return valid_scores, valid_ref, valid_hyp, decoded_valid, valid_attention_scores


def test(
    cfg_file,
    ckpt: str,
    output_path: str = None,
    datasets: dict = None,
    save_attention: bool = False,
) -> None:
    """
    Main test function. Handles loading a model from checkpoint, generating translations
    and storing them and attention plots.

    :param cfg_file: path to configuration file
    :param ckpt: path to checkpoint to load
    :param output_path: path to output
    :param datasets: datasets to predict
    :param save_attention: whether to save attention visualizations
    """
    cfg = load_config(Path(cfg_file))
    # parse train cfg
    model_dir, load_model, device, n_gpu, num_workers, normalization = parse_train_args(
        cfg["training"], mode="prediction"
    )

    if len(logger.handlers) == 0:
        _ = make_logger(model_dir, mode="test")  # version string returned

    # load the data
    if datasets is None:
        # load data
        src_vocab, trg_vocab, _, dev_data, test_data = load_data(
            data_cfg=cfg["data"], datasets=["dev", "test"]
        )
        data_to_predict = {"dev": dev_data, "test": test_data}
    else:  # avoid to load data again
        data_to_predict = {"dev": datasets["dev"], "test": datasets["test"]}
        src_vocab = datasets["src_vocab"]
        trg_vocab = datasets["trg_vocab"]

    # build model and load parameters into it
    model = build_model(cfg["model"], src_vocab=src_vocab, trg_vocab=trg_vocab)

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

    for data_set_name, data_set in data_to_predict.items():
        if data_set is None:
            continue

        logger.info("Decoding on %s set...", data_set_name)
        _, _, hypotheses, hypotheses_raw, attention_scores, = validate_on_data(
            model=model,
            data=data_set,
            compute_loss=False,
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
                    "greedy decoding."
                )

        if output_path is not None:
            output_path_set = Path(f"{output_path}.{data_set_name}")
            write_list_to_file(output_path_set, hypotheses)
            logger.info("Translations saved to: %s.", output_path_set)


def translate(cfg_file: str, ckpt: str = None, output_path: str = None) -> None:
    """
    Interactive translation function.
    Loads model from checkpoint and translates either the stdin input or asks for
    input to translate interactively. The input has to be pre-processed according to
    the data that the model was trained on, i.e. tokenized or split into subwords.
    Translations are printed to stdout.

    :param cfg_file: path to configuration file
    :param ckpt: path to checkpoint to load
    :param output_path: path to output file
    """

    def _translate_data(test_data, cfg):
        """Translates given dataset, using parameters from outer scope."""
        _, _, hypotheses, _, _ = validate_on_data(
            model=model,
            data=test_data,
            compute_loss=False,
            device=device,
            n_gpu=n_gpu,
            normalization="none",
            num_workers=num_workers,
            cfg=cfg,
        )
        return hypotheses

    cfg = load_config(Path(cfg_file))
    # parse and validate cfg
    model_dir, load_model, device, n_gpu, num_workers, _ = parse_train_args(
        cfg["training"], mode="prediction"
    )
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

    if not sys.stdin.isatty():
        # input stream given
        for line in sys.stdin.readlines():
            test_data.set_item(line.rstrip())
        all_hypotheses = _translate_data(test_data, test_cfg)
        _, _, _, _, _, _, _, n_best = parse_test_args(cfg)
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
        test_cfg["batch_size"] = 1
        test_cfg["batch_type"] = "sentence"
        while True:
            try:
                src_input = input("\nPlease enter a source sentence:\n")
                if not src_input.strip():
                    break

                # every line has to be made into dataset
                test_data.set_item(src_input.rstrip())
                hypotheses = _translate_data(test_data, test_cfg)

                print("JoeyNMT: Hypotheses ranked by score")
                for i, hyp in enumerate(hypotheses):
                    print(f"JoeyNMT #{i+1}: {hyp}")

                # reset cache
                test_data.cache = {}

            except (KeyboardInterrupt, EOFError):
                print("\nBye.")
                break
