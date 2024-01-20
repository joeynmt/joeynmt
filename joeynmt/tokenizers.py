# coding: utf-8
"""
Tokenizer module
"""
import argparse
import shutil
from pathlib import Path
from typing import Dict, List, Union

import sentencepiece as sp
from subword_nmt import apply_bpe

from joeynmt.config import ConfigurationError
from joeynmt.helpers import remove_extra_spaces, unicode_normalize
from joeynmt.helpers_for_ddp import get_logger

logger = get_logger(__name__)


class BasicTokenizer:
    # pylint: disable=too-many-instance-attributes
    SPACE = chr(32)  # ' ': half-width white space (ascii)
    SPACE_ESCAPE = chr(9601)  # '▁': sentencepiece default

    def __init__(
        self,
        level: str = "word",
        lowercase: bool = False,
        normalize: bool = False,
        max_length: int = -1,
        min_length: int = -1,
        **kwargs,
    ):
        # pylint: disable=unused-argument
        self.level = level
        self.lowercase = lowercase
        self.normalize = normalize

        # filter by length
        self.max_length = max_length
        self.min_length = min_length

        # pretokenizer
        self.pretokenizer = kwargs.get("pretokenizer", "none").lower()
        assert self.pretokenizer in ["none", "moses"], \
            "Currently, we support moses tokenizer only."
        # sacremoses
        if self.pretokenizer == "moses":
            try:
                from sacremoses import (  # pylint: disable=import-outside-toplevel
                    MosesDetokenizer,
                    MosesPunctNormalizer,
                    MosesTokenizer,
                )

                # sacremoses package has to be installed.
                # https://github.com/alvations/sacremoses
            except ImportError as e:
                logger.error(e)
                raise ImportError from e

            self.lang = kwargs.get("lang", "en")
            self.moses_tokenizer = MosesTokenizer(lang=self.lang)
            self.moses_detokenizer = MosesDetokenizer(lang=self.lang)
            if self.normalize:
                self.moses_normalizer = MosesPunctNormalizer()

    def pre_process(self, raw_input: str, allow_empty: bool = False) -> str:
        """
        Pre-process text
            - ex.) Lowercase, Normalize, Remove emojis,
                Pre-tokenize(add extra white space before punc) etc.
            - applied for all inputs both in training and inference.

        :param raw_input: raw input string
        :param allow_empty: whether to allow empty string
        :return: preprocessed input string
        """
        if not allow_empty:
            assert isinstance(raw_input, str) and raw_input.strip() != "", \
                "The input sentence is empty! Please make sure " \
                "that you are feeding a valid input."

        if self.normalize:
            raw_input = remove_extra_spaces(unicode_normalize(raw_input))

        if self.pretokenizer == "moses":
            if self.normalize:
                raw_input = self.moses_normalizer.normalize(raw_input)
            raw_input = self.moses_tokenizer.tokenize(raw_input, return_str=True)

        if self.lowercase:
            raw_input = raw_input.lower()

        if not allow_empty:
            # ensure the string is not empty.
            assert raw_input is not None and len(raw_input) > 0, raw_input
        return raw_input

    def __call__(self, raw_input: str, is_train: bool = False) -> List[str]:
        """Tokenize single sentence"""
        if raw_input is None:
            return None

        if self.level == "word":
            sequence = raw_input.split(self.SPACE)
        elif self.level == "char":
            sequence = list(raw_input.replace(self.SPACE, self.SPACE_ESCAPE))

        if is_train and self._filter_by_length(len(sequence)):
            return None
        return sequence

    def _filter_by_length(self, length: int) -> bool:
        """
        Check if the given seq length is out of the valid range.

        :param length: (int) number of tokens
        :return: True if the length is invalid(= to be filtered out), False if valid.
        """
        return length > self.max_length > 0 or self.min_length > length > 0

    def _remove_special(self, sequence: List[str], generate_unk: bool = False):
        specials = self.specials if generate_unk else self.specials + [self.unk_token]
        valid = [token for token in sequence if token not in specials]
        if len(valid) == 0:  # if empty, return <unk>
            valid = [self.unk_token]
        return valid

    def post_process(
        self,
        sequence: Union[List[str], str],
        generate_unk: bool = True,
        cut_at_sep: bool = True
    ) -> str:
        """Detokenize"""

        if isinstance(sequence, list):
            if cut_at_sep:
                try:
                    sep_pos = sequence.index(self.sep_token)  # cut off prompt
                    sequence = sequence[sep_pos + 1:]

                except ValueError as e:  # pylint: disable=unused-variable # noqa: F841
                    pass
            sequence = self._remove_special(sequence, generate_unk=generate_unk)
            if self.level == "word":
                if self.pretokenizer == "moses":
                    sequence = self.moses_detokenizer.detokenize(sequence)
                else:
                    sequence = self.SPACE.join(sequence)
            elif self.level == "char":
                sequence = "".join(sequence).replace(self.SPACE_ESCAPE, self.SPACE)

        # Remove extra spaces
        if self.normalize:
            sequence = remove_extra_spaces(sequence)

        # ensure the string is not empty.
        assert sequence is not None and len(sequence) > 0, sequence
        return sequence

    def set_vocab(self, vocab) -> None:
        """
        Set vocab
        :param vocab: (Vocabulary)
        """
        # pylint: disable=attribute-defined-outside-init
        self.unk_token = vocab.specials[vocab.unk_index]
        self.eos_token = vocab.specials[vocab.eos_index]
        self.sep_token = vocab.specials[vocab.sep_index] if vocab.sep_index else None
        specials = vocab.specials + vocab.lang_tags
        self.specials = [token for token in specials if token != self.unk_token]
        self.lang_tags = vocab.lang_tags

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(level={self.level}, "
            f"lowercase={self.lowercase}, normalize={self.normalize}, "
            f"filter_by_length=({self.min_length}, {self.max_length}), "
            f"pretokenizer={self.pretokenizer})"
        )


class SentencePieceTokenizer(BasicTokenizer):

    def __init__(
        self,
        level: str = "bpe",
        lowercase: bool = False,
        normalize: bool = False,
        max_length: int = -1,
        min_length: int = -1,
        **kwargs,
    ):
        super().__init__(level, lowercase, normalize, max_length, min_length, **kwargs)
        assert self.level == "bpe"

        self.model_file: Path = Path(kwargs["model_file"])
        assert self.model_file.is_file(), f"model file {self.model_file} not found."

        self.spm = sp.SentencePieceProcessor()
        self.spm.load(kwargs["model_file"])

        self.nbest_size: int = kwargs.get("nbest_size", 5)
        self.alpha: float = kwargs.get("alpha", 0.0)

    def __call__(self, raw_input: str, is_train: bool = False) -> List[str]:
        """Tokenize"""
        if raw_input is None:
            return None

        if is_train and self.alpha > 0:
            tokenized = self.spm.sample_encode_as_pieces(
                raw_input,
                nbest_size=self.nbest_size,
                alpha=self.alpha,
            )
        else:
            tokenized = self.spm.encode(raw_input, out_type=str)

        if is_train and self._filter_by_length(len(tokenized)):
            return None
        return tokenized

    def post_process(
        self,
        sequence: Union[List[str], str],
        generate_unk: bool = True,
        cut_at_sep: bool = True
    ) -> str:
        """Detokenize"""
        if isinstance(sequence, list):
            if cut_at_sep:
                try:
                    sep_pos = sequence.index(self.sep_token)  # cut off prompt
                    sequence = sequence[sep_pos:]
                except ValueError as e:  # pylint: disable=unused-variable # noqa: F841
                    pass
            sequence = self._remove_special(sequence, generate_unk=generate_unk)

            # Decode back to str
            sequence = self.spm.decode(sequence)
            sequence = sequence.replace(self.SPACE_ESCAPE, self.SPACE).strip()

        # Apply moses detokenizer
        if self.pretokenizer == "moses":
            sequence = self.moses_detokenizer.detokenize(sequence.split())

        # Remove extra spaces
        if self.normalize:
            sequence = remove_extra_spaces(sequence)

        # ensure the string is not empty.
        assert sequence is not None and len(sequence) > 0, sequence
        return sequence

    def set_vocab(self, vocab) -> None:
        """Set vocab"""
        super().set_vocab(vocab)
        self.spm.SetVocabulary(vocab._itos)  # pylint: disable=protected-access

    def copy_cfg_file(self, model_dir: Path) -> None:
        """Copy config file to model_dir"""
        if (model_dir / self.model_file.name).is_file():
            logger.warning(
                "%s already exists. Stop copying.",
                (model_dir / self.model_file.name).as_posix(),
            )
        shutil.copy2(self.model_file, (model_dir / self.model_file.name).as_posix())

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(level={self.level}, "
            f"lowercase={self.lowercase}, normalize={self.normalize}, "
            f"filter_by_length=({self.min_length}, {self.max_length}), "
            f"pretokenizer={self.pretokenizer}, "
            f"tokenizer={self.spm.__class__.__name__}, "
            f"nbest_size={self.nbest_size}, alpha={self.alpha})"
        )


class SubwordNMTTokenizer(BasicTokenizer):

    def __init__(
        self,
        level: str = "bpe",
        lowercase: bool = False,
        normalize: bool = False,
        max_length: int = -1,
        min_length: int = -1,
        **kwargs,
    ):
        super().__init__(level, lowercase, normalize, max_length, min_length, **kwargs)
        assert self.level == "bpe"

        codes_file = Path(kwargs["codes"])
        assert codes_file.is_file(), f"codes file {codes_file} not found."

        self.separator: str = kwargs.get("separator", "@@")
        self.dropout: float = kwargs.get("dropout", 0.0)

        bpe_parser = apply_bpe.create_parser()
        for action in bpe_parser._actions:  # workaround to ensure utf8 encoding
            if action.dest == "codes":
                action.type = argparse.FileType('r', encoding='utf8')
        bpe_args = bpe_parser.parse_args([
            "--codes", codes_file.as_posix(), "--separator", self.separator
        ])
        self.bpe = apply_bpe.BPE(
            bpe_args.codes,
            bpe_args.merges,
            bpe_args.separator,
            None,
            bpe_args.glossaries,
        )
        self.codes: Path = codes_file

    def __call__(self, raw_input: str, is_train: bool = False) -> List[str]:
        """Tokenize"""
        if raw_input is None:
            return None

        dropout = self.dropout if is_train else 0.0
        tokenized = self.bpe.process_line(raw_input, dropout).strip().split()
        if is_train and self._filter_by_length(len(tokenized)):
            return None
        return tokenized

    def post_process(
        self,
        sequence: Union[List[str], str],
        generate_unk: bool = True,
        cut_at_sep: bool = True
    ) -> str:
        """Detokenize"""
        if isinstance(sequence, list):
            if cut_at_sep:
                try:
                    sep_pos = sequence.index(self.sep_token)  # cut off prompt
                    sequence = sequence[sep_pos:]
                except ValueError as e:  # pylint: disable=unused-variable # noqa: F841
                    pass
            sequence = self._remove_special(sequence, generate_unk=generate_unk)

            # Remove separators, join with spaces
            sequence = self.SPACE.join(sequence
                                       ).replace(self.separator + self.SPACE, "")
            # Remove final merge marker.
            if sequence.endswith(self.separator):
                sequence = sequence[:-len(self.separator)]

        # Moses detokenizer
        if self.pretokenizer == "moses":
            sequence = self.moses_detokenizer.detokenize(sequence.split())

        # Remove extra spaces
        if self.normalize:
            sequence = remove_extra_spaces(sequence)

        # ensure the string is not empty.
        assert sequence is not None and len(sequence) > 0, sequence
        return sequence

    def set_vocab(self, vocab) -> None:
        """Set vocab"""
        # pylint: disable=protected-access
        super().set_vocab(vocab)
        self.bpe.vocab = set(vocab._itos) - set(vocab.specials) - set(vocab.lang_tags)

    def copy_cfg_file(self, model_dir: Path) -> None:
        """Copy config file to model_dir"""
        shutil.copy2(self.codes, (model_dir / self.codes.name).as_posix())

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(level={self.level}, "
            f"lowercase={self.lowercase}, normalize={self.normalize}, "
            f"filter_by_length=({self.min_length}, {self.max_length}), "
            f"pretokenizer={self.pretokenizer}, "
            f"tokenizer={self.bpe.__class__.__name__}, "
            f"separator={self.separator}, dropout={self.dropout})"
        )


def _build_tokenizer(cfg: Dict) -> BasicTokenizer:
    """Builds tokenizer."""
    tokenizer = None
    tokenizer_cfg = cfg.get("tokenizer_cfg", {})

    # assign lang for moses tokenizer
    if tokenizer_cfg.get("pretokenizer", "none") == "moses":
        tokenizer_cfg["lang"] = cfg["lang"]

    if cfg["level"] in ["word", "char"]:
        tokenizer = BasicTokenizer(
            level=cfg["level"],
            lowercase=cfg.get("lowercase", False),
            normalize=cfg.get("normalize", False),
            max_length=cfg.get("max_length", -1),
            min_length=cfg.get("min_length", -1),
            **tokenizer_cfg,
        )
    elif cfg["level"] == "bpe":
        tokenizer_type = cfg.get("tokenizer_type", cfg.get("bpe_type", "sentencepiece"))
        if tokenizer_type == "sentencepiece":
            assert "model_file" in tokenizer_cfg
            tokenizer = SentencePieceTokenizer(
                level=cfg["level"],
                lowercase=cfg.get("lowercase", False),
                normalize=cfg.get("normalize", False),
                max_length=cfg.get("max_length", -1),
                min_length=cfg.get("min_length", -1),
                **tokenizer_cfg,
            )
        elif tokenizer_type == "subword-nmt":
            assert "codes" in tokenizer_cfg
            tokenizer = SubwordNMTTokenizer(
                level=cfg["level"],
                lowercase=cfg.get("lowercase", False),
                normalize=cfg.get("normalize", False),
                max_length=cfg.get("max_length", -1),
                min_length=cfg.get("min_length", -1),
                **tokenizer_cfg,
            )
        else:
            raise ConfigurationError(
                f"{tokenizer_type}: Unknown tokenizer type. "
                "Valid options: {'sentencepiece', 'subword-nmt'}."
            )
    else:
        raise ConfigurationError(
            f"{cfg['level']}: Unknown tokenization level. "
            "Valid options: {'word', 'bpe', 'char'}."
        )
    return tokenizer


def build_tokenizer(cfg: Dict) -> Dict[str, BasicTokenizer]:
    src_lang = cfg["src"]["lang"]
    trg_lang = cfg["trg"]["lang"]
    tokenizer = {
        src_lang: _build_tokenizer(cfg["src"]),
        trg_lang: _build_tokenizer(cfg["trg"]),
    }
    logger.info("%s tokenizer: %s", src_lang, tokenizer[src_lang])
    logger.info("%s tokenizer: %s", trg_lang, tokenizer[trg_lang])
    return tokenizer
