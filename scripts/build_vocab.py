#!/usr/bin/env python
# coding: utf-8
"""
Build a vocab file
"""
import argparse
import sys
import tempfile
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List

import numpy as np
import sentencepiece as sp
from subword_nmt import apply_bpe, learn_bpe

from joeynmt.config import ConfigurationError, _check_special_symbols, load_config
from joeynmt.datasets import BaseDataset, build_dataset
from joeynmt.helpers import flatten, write_list_to_file
from joeynmt.tokenizers import BasicTokenizer
from joeynmt.vocabulary import sort_and_cut

# default definitions
SPECIAL_SYMBOLS = {
    "unk_token": "<unk>",
    "pad_token": "<pad>",
    "bos_token": "<s>",
    "eos_token": "</s>",
    "sep_token": None,  # "<sep>",
    "unk_id": 0,
    "pad_id": 1,
    "bos_id": 2,
    "eos_id": 3,
    "sep_id": None,  # 4,
    "lang_tags": [],  # ["<de>", "<en>"],
}


def build_vocab_from_sents(
    tokens: List[List[str]],
    vocab_file: Path,
    min_freq: int,
    max_size: int = sys.maxsize,
) -> None:
    """
    Builds char/word vocabulary from sentences.
    """
    print("### Building vocab...")

    # newly create unique token list
    counter = Counter(flatten(tokens))
    unique_tokens = sort_and_cut(counter, max_size, min_freq)
    write_list_to_file(vocab_file, unique_tokens)


def train_spm(
    sents: List[str],
    langs: List[str],
    max_size: int,
    model_file: str,
    random_subset: int,
    vocab_file: Path,
    cfg: SimpleNamespace,
    character_coverage: float = 1.0,
    model_type: str = 'unigram',
) -> None:
    """
    Train SentencePiece Model
    See: https://github.com/google/sentencepiece/blob/master/doc/options.md

    Note: model_file and vocab_file should not exist before sentencepiece training,
        will be overwritten if exist!

    :param sents: sentence list from training set
    :param langs: list of language codes, i.e ['en', 'de']
    :param max_size: same as vocab_limit in config
    :param model_file: sentencepiece model file (with ".model" extension)
    :param random_subset: subset size to train sentencepiece
    :param vocab_file: path to vocab file (one token per line)
    :param cfg: special symbols defined in config file
    :param character_coverage: amount of characters covered by the model,
        good defaults are: 0.9995 for languages with rich character set like Japanese
        or Chinese and 1.0 for other languages with small character set.
    :param model_type: model type. Choose from unigram (default), bpe, char, or word.
        The input sentence must be pretokenized when using word type.
    """
    model_file = Path(model_file)
    model_file.parent.mkdir(exist_ok=True)
    if model_file.is_file():
        print(f"Model file '{model_file}' will be overwritten.")

    with tempfile.NamedTemporaryFile(prefix="sentencepiece_", suffix=".txt") as temp:
        txt_file = Path(temp.name)
        write_list_to_file(txt_file, sents)

        model_prefix = model_file.parent / model_file.stem
        arguments = [
            f"--input={txt_file}",
            f"--model_prefix={model_prefix.as_posix()}",
            f"--model_type={model_type}",
            f"--vocab_size={max_size}",
            f"--character_coverage={character_coverage}",
            f"--accept_language={','.join(langs)}",
            f"--unk_piece={cfg.unk_token}",
            f"--bos_piece={cfg.bos_token}",
            f"--eos_piece={cfg.eos_token}",
            f"--pad_piece={cfg.pad_token}",
            f"--unk_id={cfg.unk_id}",
            f"--bos_id={cfg.bos_id}",
            f"--eos_id={cfg.eos_id}",
            f"--pad_id={cfg.pad_id}",
            "--vocabulary_output_piece_score=false",
        ]
        if cfg.sep_token:
            arguments.append(f"--control_symbols={cfg.sep_token}")
        if cfg.lang_tags:
            arguments.append(f"--user_defined_symbols={','.join(cfg.lang_tags)}")
        if len(sents) >= random_subset:  # subsample
            arguments.append(f"--input_sentence_size={random_subset}")
            arguments.append("--shuffle_input_sentence=true")
            arguments.append("--train_extremely_large_corpus=true")

        print("### Training sentencepiece...")

        # Train sentencepiece model
        sp.SentencePieceTrainer.Train(" ".join(arguments))

        # Rename vocab file
        orig_vocab_file = model_prefix.with_suffix(".vocab")
        if orig_vocab_file.as_posix() != vocab_file.as_posix():
            print(f"### Copying {orig_vocab_file} to {vocab_file} ...")
            orig_vocab_file.rename(vocab_file)


def train_bpe(
    sents: List[str],
    num_merges: int,
    min_freq: int,
    codes: str,
) -> None:
    """
    Train BPE Model
    See: https://github.com/rsennrich/subword-nmt/blob/master/subword_nmt/learn_bpe.py

    :param sents: sentence list from training set
    :param num_merges: number of merges.
        Resulting vocabulary size can be slightly smaller or larger.
    :param min_freq: minimum frequency for a token to become part of the vocabulary
    :param codes: codes file. should not exist before bpe training, will be overwritten!
    """
    codes = Path(codes)
    codes.parent.mkdir(exist_ok=True)
    if codes.is_file():
        print(f"### Codes file '{codes}' will be overwitten.")

    with tempfile.NamedTemporaryFile(prefix="subword-nmt_", suffix=".txt") as temp:
        txt_file = Path(temp.name)
        write_list_to_file(txt_file, sents)

        bpe_parser = learn_bpe.create_parser()
        bpe_args = bpe_parser.parse_args([
            f"--input={txt_file}",
            f"--output={codes}",
            f"--symbols={num_merges}",
            f"--min-frequency={min_freq}",
        ])
        print("### Training bpe...")
        learn_bpe.learn_bpe(
            bpe_args.input,
            bpe_args.output,
            bpe_args.symbols,
            bpe_args.min_frequency,
            bpe_args.verbose,
            is_dict=False,
            total_symbols=False,
        )


def save_bpe(
    sents: List[str],
    vocab_file: str,
    codes: str,
    max_size: int,
    min_freq: int,
    separator: str = "@@",
    **kwargs,  # pylint: disable=unused-argument
) -> None:
    # pylint: disable=unused-argument
    bpe_parser = apply_bpe.create_parser()
    bpe_args = bpe_parser.parse_args([
        f"--codes={codes}",
        f"--separator={separator}",
    ])
    print("### Applying bpe...")
    bpe = apply_bpe.BPE(
        bpe_args.codes,
        bpe_args.merges,
        bpe_args.separator,
        None,
        bpe_args.glossaries,
    )
    tokens = [bpe.process_line(sent).strip().split() for sent in sents]
    # No max_size to include all vocab items generated by merges.
    build_vocab_from_sents(
        tokens=tokens, vocab_file=vocab_file, min_freq=min_freq, max_size=max_size
    )


def run(
    args,
    train_data: BaseDataset,
    langs: List[str],
    level: str,
    min_freq: int,
    max_size: int,
    vocab_file: Path,
    tokenizer_type: str,
    tokenizer_cfg: Dict,
    special_symbols: Dict = SPECIAL_SYMBOLS,
):
    # pylint: disable=redefined-outer-name, dangerous-default-value
    # Warn overwriting
    if vocab_file.is_file():
        print(f"### Vocab file '{vocab_file}' will be overwritten.")

    def _get_sents(args, dataset: BaseDataset, langs: List[str], tokenized: bool):
        assert len(langs) in [1, 2], langs
        if len(dataset) > args.random_subset:
            # sample subset
            n = args.random_subset if len(langs) == 1 else args.random_subset // 2
            np.random.seed(args.seed)
            indices = np.random.choice(range(len(dataset)), n, replace=False)
            indices.sort()
            dataset.indices = indices.tolist()

        sents = []
        for lang in langs:
            sents.extend(dataset.get_list(lang=lang, tokenized=tokenized))
        assert len(sents) <= args.random_subset, (len(sents), len(dataset))
        return sents

    if level in ["char", "word"]:
        # Get preprocessed tokenized sentences
        tokens = _get_sents(args, train_data, langs, tokenized=True)

        build_vocab_from_sents(
            tokens=tokens,
            vocab_file=vocab_file,
            min_freq=min_freq,
            max_size=max_size,
        )

    elif level == "bpe":
        # Get preprocessed sentences
        sents = _get_sents(args, train_data, langs, tokenized=False)

        if tokenizer_type == "sentencepiece":
            train_spm(
                sents=sents,
                langs=langs,
                max_size=max_size,
                model_file=tokenizer_cfg["model_file"],
                random_subset=args.random_subset,
                vocab_file=vocab_file,
                cfg=SimpleNamespace(**special_symbols),
                character_coverage=tokenizer_cfg.get("character_coverage", 1.0),
                model_type=tokenizer_cfg.get("model_type", "unigram"),
            )

        elif tokenizer_type == "subword-nmt":
            train_bpe(
                sents=sents,
                num_merges=tokenizer_cfg["num_merges"],
                min_freq=min_freq,
                codes=tokenizer_cfg["codes"],
            )
            save_bpe(
                sents=sents,
                vocab_file=vocab_file,
                max_size=max_size,
                min_freq=min_freq,
                **tokenizer_cfg,
            )
        else:
            raise ConfigurationError(f"{tokenizer_type}: Unknown tokenizer type.")
            # TODO: support fastBPE training! https://github.com/glample/fastBPE

    print("### Done.")


def main(args) -> None:  # pylint: disable=redefined-outer-name
    cfg = load_config(Path(args.config_path))
    src_cfg = cfg["data"]["src"]
    trg_cfg = cfg["data"]["trg"]
    special_symbols = _check_special_symbols(
        cfg["data"].get("special_symbols", SPECIAL_SYMBOLS)
    )

    # build basic tokenizer just for preprocessing purpose
    tokenizer = {
        src_cfg["lang"]: BasicTokenizer(
            level=src_cfg["level"],
            lowercase=src_cfg.get("lowercase", False),
            normalize=src_cfg.get("normalize", False),
            pretokenizer=src_cfg["tokenizer_cfg"].get("pretokenizer", "none").lower()
            if "tokenizer_cfg" in src_cfg else "none",
        ),
        trg_cfg["lang"]: BasicTokenizer(
            level=trg_cfg["level"],
            lowercase=trg_cfg.get("lowercase", False),
            normalize=trg_cfg.get("normalize", False),
            pretokenizer=trg_cfg["tokenizer_cfg"].get("pretokenizer", "none").lower()
            if "tokenizer_cfg" in trg_cfg else "none",
        ),
    }

    train_data = build_dataset(
        dataset_type=cfg["data"]["dataset_type"],
        path=cfg["data"]["train"],
        src_lang=src_cfg["lang"],
        trg_lang=trg_cfg["lang"],
        split="train",
        tokenizer=tokenizer,
        **cfg["data"].get("dataset_cfg", {}),
    )

    def _parse_cfg(cfg):
        lang = cfg["lang"]
        level = cfg["level"]
        min_freq = cfg.get("voc_min_freq", 1)
        max_size = int(cfg.get("voc_limit", sys.maxsize))

        voc_file = cfg.get("voc_file", 'vocab.txt')
        tok_type = cfg.get("tokenizer_type", "sentencepiece")
        tok_cfg = cfg.get("tokenizer_cfg", {})
        return lang, level, min_freq, max_size, voc_file, tok_type, tok_cfg

    src_tuple = _parse_cfg(src_cfg)
    trg_tuple = _parse_cfg(trg_cfg)

    if args.joint:
        for s, t in zip(src_tuple[1:], trg_tuple[1:]):
            assert s == t

        run(
            args,
            train_data=train_data,
            langs=[src_tuple[0], trg_tuple[0]],
            level=src_tuple[1],
            min_freq=src_tuple[2],
            max_size=src_tuple[3],
            vocab_file=Path(src_tuple[4]),
            tokenizer_type=src_tuple[5],
            tokenizer_cfg=src_tuple[6],
            special_symbols=special_symbols,
        )

    else:
        for lang, level, min_freq, max_size, voc_file, tok_type, tok_cfg in [
            src_tuple,
            trg_tuple,
        ]:
            run(
                args,
                train_data=train_data,
                langs=[lang],
                level=level,
                min_freq=min_freq,
                max_size=max_size,
                vocab_file=Path(
                    f'{lang}_vocab.txt' if voc_file == 'vocab.txt' else voc_file
                ),
                tokenizer_type=tok_type,
                tokenizer_cfg=tok_cfg,
                special_symbols=special_symbols,
            )


if __name__ == "__main__":

    ap = argparse.ArgumentParser(description="Builds a vocabulary from training data.")

    ap.add_argument("config_path", type=str, help="path to YAML config file")
    ap.add_argument(
        "--joint", action="store_true", help="Jointly train src and trg vocab"
    )
    ap.add_argument(
        "--random-subset",
        type=int,
        default=1000000,
        help="Take this many examples randomly to train subwords.",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed to select the train subset. "
        "used only if len(dataset) > args.random_subset.",
    )
    args = ap.parse_args()

    main(args)
