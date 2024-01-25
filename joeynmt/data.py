# coding: utf-8
"""
Data module
"""
from typing import Dict, Optional, Tuple

from joeynmt.datasets import BaseDataset, build_dataset
from joeynmt.helpers_for_ddp import get_logger
from joeynmt.tokenizers import build_tokenizer
from joeynmt.vocabulary import Vocabulary, build_vocab

logger = get_logger(__name__)


def load_data(cfg: Dict, datasets: list = None) \
    -> Tuple[Vocabulary, Vocabulary, Optional[BaseDataset],
             Optional[BaseDataset], Optional[BaseDataset]]:
    """
    Load train, dev and optionally test data as specified in configuration.
    Vocabularies are created from the training set with a limit of `voc_limit` tokens
    and a minimum token frequency of `voc_min_freq` (specified in the configuration
    dictionary).

    The training data is filtered to include sentences up to `max_length` on source
    and target side.

    If you set `random_{train|dev}_subset`, a random selection of this size is used
    from the {train|development} set instead of the full {train|development} set.

    :param cfg: configuration dictionary for data ("data" part of config file)
    :param datasets: list of dataset names to load
    :returns:
        - src_vocab: source vocabulary
        - trg_vocab: target vocabulary
        - train_data: training dataset
        - dev_data: development dataset
        - test_data: test dataset if given, otherwise None
    """
    assert len(datasets) > 0, datasets

    src_cfg = cfg["src"]
    trg_cfg = cfg["trg"]

    # load data from files
    src_lang = src_cfg["lang"]
    trg_lang = trg_cfg["lang"]
    train_path = cfg.get("train", None)
    dev_path = cfg.get("dev", None)
    test_path = cfg.get("test", None)

    if train_path is None and dev_path is None and test_path is None:
        raise ValueError("Please specify at least one data source path.")

    # build tokenizer
    logger.info("Building tokenizer...")
    tokenizer = build_tokenizer(cfg)

    dataset_type = cfg.get("dataset_type", "plain")
    dataset_cfg = cfg.get("dataset_cfg", {})

    has_prompt = {
        src_lang: src_cfg.get("has_prompt", False),
        trg_lang: trg_cfg.get("has_prompt", False),
    }

    # train data
    train_data = None
    if "train" in datasets and train_path is not None:
        train_subset = cfg.get("sample_train_subset", -1)
        if "random_train_subset" in cfg:
            logger.warning(
                "`random_train_subset` option is obsolete. "
                "Please use `sample_train_subset` instead."
            )
            train_subset = cfg.get("random_train_subset", train_subset)
        logger.info("Loading train set...")
        train_data = build_dataset(
            dataset_type=dataset_type,
            path=train_path,
            src_lang=src_lang,
            trg_lang=trg_lang,
            split="train",
            tokenizer=tokenizer,
            has_prompt=has_prompt,
            random_subset=train_subset,
            **dataset_cfg,
        )

    # build vocab
    logger.info("Building vocabulary...")
    src_vocab, trg_vocab = build_vocab(cfg, dataset=train_data)

    # set vocab to tokenizer
    tokenizer[src_lang].set_vocab(src_vocab)
    tokenizer[trg_lang].set_vocab(trg_vocab)

    # encoding func
    sequence_encoder = {
        src_lang: src_vocab.sentences_to_ids,
        trg_lang: trg_vocab.sentences_to_ids,
    }
    if train_data is not None:
        train_data.sequence_encoder = sequence_encoder

    # dev data
    dev_data = None
    if "dev" in datasets and dev_path is not None:
        dev_subset = cfg.get("sample_dev_subset", -1)
        if "random_dev_subset" in cfg:
            logger.warning(
                "`random_dev_subset` option is obsolete. "
                "Please use `sample_dev_subset` instead."
            )
            dev_subset = cfg.get("random_dev_subset", dev_subset)
        logger.info("Loading dev set...")
        dev_data = build_dataset(
            dataset_type=dataset_type,
            path=dev_path,
            src_lang=src_lang,
            trg_lang=trg_lang,
            split="dev",
            tokenizer=tokenizer,
            sequence_encoder=sequence_encoder,
            has_prompt=has_prompt,
            random_subset=dev_subset,
            **dataset_cfg,
        )

    # test data
    test_data = None
    if "test" in datasets and test_path is not None:
        logger.info("Loading test set...")
        test_data = build_dataset(
            dataset_type=dataset_type,
            path=test_path,
            src_lang=src_lang,
            trg_lang=trg_lang,
            split="test",
            tokenizer=tokenizer,
            sequence_encoder=sequence_encoder,
            has_prompt=has_prompt,
            random_subset=-1,  # no subsampling for test
            **dataset_cfg,
        )

    if "stream" in datasets:
        test_data = build_dataset(
            dataset_type="stream",
            path=None,
            src_lang=src_lang,
            trg_lang=trg_lang,
            split="test",
            tokenizer=tokenizer,
            sequence_encoder=sequence_encoder,
            has_prompt=has_prompt,
        )
    logger.info("Data loaded.")

    # Log statistics of data and vocabulary
    logger.info("Train dataset: %s", train_data)
    logger.info("Valid dataset: %s", dev_data)
    logger.info(" Test dataset: %s", test_data)

    if train_data:
        src = "\n\t[SRC] " + " ".join(
            train_data.get_item(idx=0, lang=train_data.src_lang, is_train=False)
        )
        trg = "\n\t[TRG] " + " ".join(
            train_data.get_item(idx=0, lang=train_data.trg_lang, is_train=False)
        )
        logger.info("First training example:%s%s", src, trg)

    logger.info("First 10 Src tokens: %s", src_vocab.log_vocab(10))
    logger.info("First 10 Trg tokens: %s", trg_vocab.log_vocab(10))

    logger.info("Number of unique Src tokens (vocab_size): %d", len(src_vocab))
    logger.info("Number of unique Trg tokens (vocab_size): %d", len(trg_vocab))

    return src_vocab, trg_vocab, train_data, dev_data, test_data
