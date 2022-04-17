# coding: utf-8
"""
Tokenizer module
"""

import logging
import re
import shutil
import sys
import unicodedata
from pathlib import Path
from typing import Dict, List

import sentencepiece as sp
from subword_nmt import apply_bpe

from joeynmt.helpers import ConfigurationError


logger = logging.getLogger(__name__)


def _unicode_normalize(cls, s):
    pt = re.compile("([{}]+)".format(cls))

    def norm(c):
        return unicodedata.normalize("NFKC", c) if pt.match(c) else c

    s = "".join(norm(x) for x in re.split(pt, s))
    s = re.sub("－", "-", s)
    return s


def _remove_extra_spaces(s):
    s = re.sub("\u200b", "", s)
    s = re.sub("[ 　]+", " ", s)
    blocks = "".join(
        (
            "\u4E00-\u9FFF",  # CJK UNIFIED IDEOGRAPHS
            "\u3040-\u309F",  # HIRAGANA
            "\u30A0-\u30FF",  # KATAKANA
            "\u3000-\u303F",  # CJK SYMBOLS AND PUNCTUATION
            "\uFF00-\uFFEF",  # HALFWIDTH AND FULLWIDTH FORMS
        )
    )
    # latin = ''.join(('\u0000-\u007F',   # Basic Latin[g]
    #                 '\u0080-\u00FF',   # Latin-1 Supplement[h]
    # ))

    def _remove_space_between(cls1, cls2, s):
        # pylint: disable=consider-using-f-string
        p = re.compile("([{}]) ([{}])".format(cls1, cls2))
        while p.search(s):
            s = p.sub(r"\1\2", s)
        return s

    s = _remove_space_between(blocks, blocks, s)
    # s = _remove_space_between(blocks, latin, s)
    # s = _remove_space_between(latin, blocks, s)

    s = re.sub(" ,", ",", s)
    s = re.sub(" .", ".", s)
    s = re.sub(" ?", "?", s)
    s = re.sub(" !", "!", s)
    return s.strip()


def _normalize(s):
    """Normalize unicode strings
    cf.)
    https://github.com/neologd/mecab-ipadic-neologd/wiki/Regexp.ja
    http://lotus.kuee.kyoto-u.ac.jp/WAT/Timely_Disclosure_Documents_Corpus/specifications.html
    """
    s = re.sub("\t", " ", s)
    s = _unicode_normalize("０-９Ａ-Ｚａ-ｚ｡-ﾟ", s)

    def _maketrans(f, t):
        return {ord(x): ord(y) for x, y in zip(f, t)}

    s = re.sub("[˗֊‐‑‒–⁃⁻₋−]+", "-", s)  # normalize hyphens
    s = re.sub("[﹣－ｰ—―─━ー]+", "ー", s)  # normalize choonpus
    s = re.sub("[~∼∾〜〰～]+", "〜", s)  # normalize tildes (modified by Isao Sonobe)
    s = s.translate(
        _maketrans(
            "!\"#$%&'()*+,-./:;<=>?@[¥]^_`{|}~｡､･｢｣",
            "！”＃＄％＆’（）＊＋，－．／：；＜＝＞？＠［￥］＾＿｀｛｜｝〜。、・「」",
        )
    )

    s = _remove_extra_spaces(s)
    s = _unicode_normalize("！”＃＄％＆’（）＊＋，－．／：；＜＞？＠［￥］＾＿｀｛｜｝〜", s)  # keep ＝,・,「,」
    s = re.sub("[’]", "'", s)
    s = re.sub("[”“]", '"', s)
    return s


class BasicTokenizer:
    SPACE = chr(32)  # ' ': half-width white space (ascii)
    SPACE_ESCAPE = chr(9601)  # '▁': sentencepiece default

    def __init__(
        self,
        level: str = "word",
        lowercase: bool = False,
        normalize: bool = False,
        max_len: int = -1,
        min_len: int = -1,
        **kwargs,
    ):
        self.level = level
        self.lowercase = lowercase
        self.normalize = normalize

        # filter by length
        self.max_len = max_len
        self.min_len = min_len

    def pre_process(self, raw_input: str) -> str:
        """
        Pre-process text
            - ex.) Lowercase, Normalize, Remove emojis,
                Pre-tokenize(add extra white space before punc) etc.
            - applied for all inputs both in training and inference.
        """
        if self.lowercase:
            raw_input = raw_input.lower()
        if self.normalize:
            raw_input = _normalize(raw_input)
            # TODO: support other normalization(?)
        return raw_input

    def __call__(
        self, raw_input: str, sample: bool = False, filter_by_length: bool = False
    ) -> List[str]:
        """Tokenize single sentence"""
        sequence = self.pre_process(raw_input)
        if self.level == "word":
            sequence = sequence.split(self.SPACE)
        elif self.level == "char":
            sequence = list(sequence.replace(self.SPACE, self.SPACE_ESCAPE))

        if filter_by_length and self._filter_by_length(len(sequence)):
            return None
        return sequence

    def _filter_by_length(self, length: int):
        if length > self.max_len > 0 or self.min_len > length > 0:
            return True
        else:
            return False

    def post_process(self, output: List[str]) -> str:
        """Detokenize"""
        if self.level == "word":
            detokenized = self.SPACE.join(output)
        elif self.level == "char":
            detokenized = "".join(output).replace(self.SPACE_ESCAPE, self.SPACE)

        # Remove extra spaces
        if self.normalize:
            detokenized = _remove_extra_spaces(detokenized)
        return detokenized

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(level={self.level}, "
            f"lowercase={self.lowercase}, normalize={self.normalize}, "
            f"filter_by_length=({self.min_len}, {self.max_len}))"
        )


class SentencePieceTokenizer(BasicTokenizer):
    def __init__(
        self,
        level: str = "bpe",
        lowercase: bool = False,
        normalize: bool = False,
        max_len: int = -1,
        min_len: int = -1,
        **kwargs,
    ):
        super().__init__(level, lowercase, normalize, max_len, min_len)
        assert self.level == "bpe"

        self.model_file: Path = Path(kwargs["model_file"])
        assert self.model_file.is_file(), f"model file {self.model_file} not found."

        self.spm = sp.SentencePieceProcessor()
        self.spm.load(kwargs["model_file"])

        self.enable_sampling: bool = kwargs.get("enable_sampling", False)
        self.alpha: float = kwargs.get("alpha", 0.0)

    def __call__(
        self, raw_input: str, sample: bool = False, filter_by_length: bool = False
    ) -> List[str]:
        if sample:
            tokenized = self.spm.encode(
                raw_input,
                out_type=str,
                enable_sampling=self.enable_sampling,
                alpha=self.alpha,
            )
        else:
            tokenized = self.spm.encode(raw_input, out_type=str)

        if filter_by_length and self._filter_by_length(len(tokenized)):
            return None
        return tokenized

    def post_process(self, output: List[str]) -> str:
        detokenized = self.spm.decode(output)

        # Remove extra spaces
        if self.normalize:
            detokenized = _remove_extra_spaces(detokenized)
        return detokenized

    def set_vocab(self, itos: List[str]) -> None:
        """
        Set vocab
        :param itos: (list) indices-to-symbols mapping
        """
        self.spm.SetVocabulary(itos)

    def copy_cfg_file(self, model_dir: Path) -> None:
        """Copy confg file to model_dir"""
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
            f"filter_by_length=({self.min_len}, {self.max_len}), "
            f"tokenizer={self.spm.__class__.__name__}, "
            f"enable_sampling={self.enable_sampling}, alpha={self.alpha})"
        )


class SubwordNMTTokenizer(BasicTokenizer):
    def __init__(
        self,
        level: str = "bpe",
        lowercase: bool = False,
        normalize: bool = False,
        max_len: int = -1,
        min_len: int = -1,
        **kwargs,
    ):
        super().__init__(level, lowercase, normalize, max_len, min_len)
        assert self.level == "bpe"

        self.codes: Path = Path(kwargs["codes"])
        assert self.codes.is_file(), f"codes file {self.codes} not found."

        bpe_parser = apply_bpe.create_parser()
        bpe_args = bpe_parser.parse_args(
            ["--codes", kwargs["codes"], "--separator", kwargs.get("separator", "@@")]
        )
        self.bpe = apply_bpe.BPE(
            bpe_args.codes,
            bpe_args.merges,
            bpe_args.separator,
            None,
            bpe_args.glossaries,
        )
        self.separator: str = kwargs.get("separator", "@@")
        self.dropout: float = kwargs.get("dropout", 0.0)

    def __call__(
        self, raw_input: str, sample: bool = False, filter_by_length: bool = False
    ) -> List[str]:
        dropout = self.dropout if sample else 0.0
        tokenized = self.bpe.process_line(raw_input, dropout).strip().split()
        if filter_by_length and self._filter_by_length(len(tokenized)):
            return None
        return tokenized

    def post_process(self, output: List[str]) -> str:
        # Remove merge markers within the sentence.
        detokenized = " ".join(output).replace(self.separator + " ", "")
        # Remove final merge marker.
        if detokenized.endswith(self.separator):
            detokenized = detokenized[:-2]

        # Remove extra spaces
        if self.normalize:
            detokenized = _remove_extra_spaces(detokenized)
        return detokenized

    def copy_cfg_file(self, model_dir: Path) -> None:
        shutil.copy2(self.codes, (model_dir / self.codes.name).as_posix())

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(level={self.level}, "
            f"lowercase={self.lowercase}, normalize={self.normalize}, "
            f"filter_by_length=({self.min_len}, {self.max_len}), "
            f"tokenizer={self.bpe.__class__.__name__}, "
            f"separator={self.separator}, dropout={self.dropout})"
        )


def _build_tokenizer(cfg: Dict) -> BasicTokenizer:
    """Builds tokenizer."""
    tokenizer = None
    if cfg["level"] in ["word", "char"]:
        tokenizer = BasicTokenizer(
            level=cfg["level"],
            lowercase=cfg.get("lowercase", False),
            normalize=cfg.get("normalize", False),
            max_len=cfg.get("max_length", -1),
            min_len=cfg.get("min_length", -1),
        )
    elif cfg["level"] == "bpe":
        tokenizer_type = cfg.get("tokenizer_type", cfg.get("bpe_type", "sentencepiece"))
        if tokenizer_type == "sentencepiece":
            assert "tokenizer_cfg" in cfg and "model_file" in cfg["tokenizer_cfg"]
            tokenizer = SentencePieceTokenizer(
                level=cfg["level"],
                lowercase=cfg.get("lowercase", False),
                normalize=cfg.get("normalize", False),
                max_len=cfg.get("max_length", -1),
                min_len=cfg.get("min_length", -1),
                **cfg["tokenizer_cfg"],
            )
        elif tokenizer_type == "subword-nmt":
            assert "tokenizer_cfg" in cfg and "codes" in cfg["tokenizer_cfg"]
            tokenizer = SubwordNMTTokenizer(
                level=cfg["level"],
                lowercase=cfg.get("lowercase", False),
                normalize=cfg.get("normalize", False),
                max_len=cfg.get("max_length", -1),
                min_len=cfg.get("min_length", -1),
                **cfg["tokenizer_cfg"],
            )
        else:
            raise ConfigurationError(f"{tokenizer_type}: Unknown tokenizer type.")
    else:
        raise ConfigurationError(f"{cfg['level']}: Unknown tokenization level.")
    return tokenizer


def build_tokenizer(data_cfg: Dict) -> Dict[str, BasicTokenizer]:
    src_lang = data_cfg["src"]["lang"]
    trg_lang = data_cfg["trg"]["lang"]
    tokenizer = {
        src_lang: _build_tokenizer(data_cfg["src"]),
        trg_lang: _build_tokenizer(data_cfg["trg"]),
    }
    logger.info("%s tokenizer: %s", src_lang, tokenizer[src_lang])
    logger.info("%s tokenizer: %s", trg_lang, tokenizer[trg_lang])
    return tokenizer
