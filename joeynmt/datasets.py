# coding: utf-8
"""
Dataset module
"""
import logging
import random
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple, Union

from torch.utils.data import Dataset

from joeynmt.helpers import ConfigurationError, read_list_from_file
from joeynmt.tokenizers import BasicTokenizer

logger = logging.getLogger(__name__)


class BaseDataset(Dataset):
    """
    BaseDataset which loads and looks up data.
    - holds pointer to tokenizers, encoding functions.

    :param path: path to data directory
    :param src_lang: source language code, i.e. `en`
    :param trg_lang: target language code, i.e. `de`
    :param has_trg: bool indicator if trg exists
    :param split: bool indicator for train set or not
    :param tokenizer: tokenizer objects
    :param sequence_encoder: encoding functions
    """

    def __init__(
        self,
        path: str,
        src_lang: str,
        trg_lang: str,
        split: int = "train",
        has_trg: bool = True,
        tokenizer: Dict[str, BasicTokenizer] = None,
        sequence_encoder: Dict[str, Callable] = None,
        random_subset: int = -1,
    ):
        self.path = path
        self.src_lang = src_lang
        self.trg_lang = trg_lang
        self.has_trg = has_trg
        self.split = split
        if self.split == "train":
            assert self.has_trg

        _place_holder = {self.src_lang: None, self.trg_lang: None}
        self.tokenizer = _place_holder if tokenizer is None else tokenizer
        self.sequence_encoder = (_place_holder
                                 if sequence_encoder is None else sequence_encoder)

        # for ransom subsampling
        self.random_subset = random_subset

    def sample_random_subset(self, seed: int = 42) -> None:
        # pylint: disable=unused-argument
        assert (
            self.split != "test" and self.__len__() > self.random_subset > 0
        ), f"Can only subsample from train or dev set larger than {self.random_subset}."

    def reset_random_subset(self):
        raise NotImplementedError

    def load_data(self, path: Path, **kwargs) -> Any:
        """
        load data
            - preprocessing (lowercasing etc) is applied here.
        """
        raise NotImplementedError

    def get_item(self, idx: int, lang: str) -> List[str]:
        """
        seek one src/trg item of given index.
            - tokenization is applied here.
            - length-filtering, bpe-dropout etc also triggered if self.split == "train"
        """
        raise NotImplementedError

    def __getitem__(self, idx: Union[int, str]) -> Tuple[List[str], List[str]]:
        """lookup one item pair of given index."""
        src, trg = None, None
        src = self.get_item(idx=idx, lang=self.src_lang)
        if self.has_trg:
            trg = self.get_item(idx=idx, lang=self.trg_lang)
            if trg is None:
                src = None
        return src, trg

    def get_list(self,
                 lang: str,
                 tokenized: bool = False) -> Union[List[str], List[List[str]]]:
        """get data column-wise."""
        raise NotImplementedError

    @property
    def src(self) -> List[str]:
        """get detokenized preprocessed data in src language."""
        return self.get_list(self.src_lang, tokenized=False)

    @property
    def trg(self) -> List[str]:
        """get detokenized preprocessed data in trg language."""
        return self.get_list(self.trg_lang, tokenized=False) if self.has_trg else []

    def __len__(self) -> int:
        raise NotImplementedError

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}(split={self.split}, len={self.__len__()}, "
                f"src_lang={self.src_lang}, trg_lang={self.trg_lang}, "
                f"has_trg={self.has_trg}, random_subset={self.random_subset})")


class PlaintextDataset(BaseDataset):
    """
    PlaintextDataset which stores plain text pairs.
    - used for text file data in the format of one sentence per line.
    """

    def __init__(
        self,
        path: str,
        src_lang: str,
        trg_lang: str,
        split: int = "train",
        has_trg: bool = True,
        tokenizer: Dict[str, BasicTokenizer] = None,
        sequence_encoder: Dict[str, Callable] = None,
        random_subset: int = -1,
        **kwargs,
    ):
        super().__init__(
            path=path,
            src_lang=src_lang,
            trg_lang=trg_lang,
            split=split,
            has_trg=has_trg,
            tokenizer=tokenizer,
            sequence_encoder=sequence_encoder,
            random_subset=random_subset,
        )

        # load data
        self.data = self.load_data(path, **kwargs)
        self._initial_len = len(self.data[self.src_lang])

        # for random subsampling
        self.idx_map = []

    def load_data(self, path: str, **kwargs) -> Any:

        def _pre_process(seq, lang):
            if self.tokenizer[lang] is not None:
                seq = [self.tokenizer[lang].pre_process(s) for s in seq if len(s) > 0]
            return seq

        path = Path(path)
        src_file = path.with_suffix(f"{path.suffix}.{self.src_lang}")
        assert src_file.is_file(), f"{src_file} not found. Abort."

        src_list = read_list_from_file(src_file)
        data = {self.src_lang: _pre_process(src_list, self.src_lang)}

        if self.has_trg:
            trg_file = path.with_suffix(f"{path.suffix}.{self.trg_lang}")
            assert trg_file.is_file(), f"{trg_file} not found. Abort."

            trg_list = read_list_from_file(trg_file)
            data[self.trg_lang] = _pre_process(trg_list, self.trg_lang)
            assert len(src_list) == len(trg_list)
        return data

    def sample_random_subset(self, seed: int = 42) -> None:
        super().sample_random_subset(seed)  # check validity

        random.seed(seed)  # resample every epoch: seed += epoch_no
        self.idx_map = list(random.sample(range(self._initial_len), self.random_subset))

    def reset_random_subset(self):
        self.idx_map = []

    def get_item(self, idx: int, lang: str, is_train: bool = None) -> List[str]:
        line = self._look_up_item(idx, lang)
        is_train = self.split == "train" if is_train is None else is_train
        item = self.tokenizer[lang](line, is_train=is_train)
        return item

    def _look_up_item(self, idx: int, lang: str) -> str:
        try:
            if len(self.idx_map) > 0:
                idx = self.idx_map[idx]
            line = self.data[lang][idx]
            return line
        except Exception as e:
            print(idx, self._initial_len)
            raise Exception from e

    def get_list(self,
                 lang: str,
                 tokenized: bool = False) -> Union[List[str], List[List[str]]]:
        """
        Return list of preprocessed sentences in the given language.
        (not length-filtered, no bpe-dropout)
        """
        item_list = []
        for idx in range(self.__len__()):
            item = self._look_up_item(idx, lang)
            if tokenized:
                item = self.tokenizer[lang](self._look_up_item(idx, lang),
                                            is_train=False)
            item_list.append(item)
        return item_list

    def __len__(self) -> int:
        if len(self.idx_map) > 0:
            return len(self.idx_map)
        return self._initial_len


class TsvDataset(BaseDataset):
    """
    TsvDataset which handles data in tsv format.
    - file_name should be specified without extention `.tsv`
    - needs src_lang and trg_lang (i.e. `en`, `de`) in header.
    """

    def __init__(
        self,
        path: str,
        src_lang: str,
        trg_lang: str,
        split: int = "train",
        has_trg: bool = True,
        tokenizer: Dict[str, BasicTokenizer] = None,
        sequence_encoder: Dict[str, Callable] = None,
        random_subset: int = -1,
        **kwargs,
    ):
        super().__init__(
            path=path,
            src_lang=src_lang,
            trg_lang=trg_lang,
            split=split,
            has_trg=has_trg,
            tokenizer=tokenizer,
            sequence_encoder=sequence_encoder,
            random_subset=random_subset,
        )

        # load tsv file
        self.df = self.load_data(path, **kwargs)

        # for random subsampling
        self._initial_df = None

    def load_data(self, path: str, **kwargs) -> Any:
        path = Path(path)
        file_path = path.with_suffix(f"{path.suffix}.tsv")
        assert file_path.is_file(), f"{file_path} not found. Abort."

        # read tsv data
        try:
            import pandas as pd  # pylint: disable=import-outside-toplevel

            df = pd.read_csv(
                file_path.as_posix(),
                sep="\t",
                header=0,
                encoding="utf-8",
                escapechar="\\",
                quoting=3,
                na_filter=False,
            )
            # TODO: use `chunksize` for online data loading.
            assert self.src_lang in df.columns
            df[self.src_lang] = df[self.src_lang].apply(
                self.tokenizer[self.src_lang].pre_process)

            if self.trg_lang not in df.columns:
                self.has_trg = False
                assert self.split == "test"
            if self.has_trg:
                df[self.trg_lang] = df[self.trg_lang].apply(
                    self.tokenizer[self.trg_lang].pre_process)
            return df

        except ImportError as e:
            logger.error(e)
            raise ImportError from e

    def sample_random_subset(self, seed: int = 42) -> None:
        super().sample_random_subset(seed)  # check validity

        if self._initial_df is None:
            self._initial_df = self.df.copy(deep=True)

        self.df = self._initial_df.sample(
            n=self.random_subset,
            replace=False,
            random_state=seed,  # resample every epoch: seed += epoch_no
        ).reset_index()

    def reset_random_subset(self):
        assert self._initial_df is not None
        self.df = self._initial_df
        self._initial_df = None

    def get_item(self, idx: int, lang: str, is_train: bool = None) -> List[str]:
        line = self.df.iloc[idx][lang]
        is_train = self.split == "train" if is_train is None else is_train
        item = self.tokenizer[lang](line, is_train=is_train)
        return item

    def get_list(self,
                 lang: str,
                 tokenized: bool = False) -> Union[List[str], List[List[str]]]:
        return (self.df[lang].apply(self.tokenizer[lang]).to_list()
                if tokenized else self.df[lang].to_list())

    def __len__(self) -> int:
        return len(self.df)


class StreamDataset(BaseDataset):
    """
    StreamDataset which interacts with stream inputs.
    - called by `translate()` func in `prediction.py`.
    """

    def __init__(
        self,
        path: str,
        src_lang: str,
        trg_lang: str,
        split: int = "test",
        has_trg: bool = False,
        tokenizer: Dict[str, BasicTokenizer] = None,
        sequence_encoder: Dict[str, Callable] = None,
        random_subset: int = -1,
        **kwargs,
    ):
        # pylint: disable=unused-argument
        super().__init__(
            path=path,
            src_lang=src_lang,
            trg_lang=trg_lang,
            split=split,
            has_trg=has_trg,
            tokenizer=tokenizer,
            sequence_encoder=sequence_encoder,
            random_subset=random_subset,
        )
        # place holder
        self.cache = {}

    def set_item(self, line: str) -> None:
        """
        takes source sentence string (i.e. `this is a test.`)
            - tokenizer specified in config will be applied in this func.

        :param line: (str)
        """
        idx = len(self.cache)
        line = self.tokenizer[self.src_lang].pre_process(line)
        self.cache[idx] = (line, None)

    def get_item(self, idx: int, lang: str, is_train: bool = None) -> List[str]:
        # pylint: disable=unused-argument
        assert idx in self.cache, (idx, self.cache)
        assert lang == self.src_lang, (lang, self.src_lang)
        line, _ = self.cache[idx]
        item = self.tokenizer[lang](line, is_train=False)
        return item

    def __len__(self) -> int:
        return len(self.cache)

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}(split={self.split}, len={len(self.cache)}, "
                f"src_lang={self.src_lang}, trg_lang={self.trg_lang}, "
                f"has_trg={self.has_trg}, random_subset={self.random_subset})")


class BaseHuggingfaceDataset(BaseDataset):
    """
    Wrapper for Huggingface's dataset object
    cf.) https://huggingface.co/docs/datasets
    """

    def __init__(
        self,
        path: str,
        src_lang: str,
        trg_lang: str,
        has_trg: bool = True,
        tokenizer: Dict[str, BasicTokenizer] = None,
        sequence_encoder: Dict[str, Callable] = None,
        random_subset: int = -1,
        **kwargs,
    ):
        super().__init__(
            path=path,
            src_lang=src_lang,
            trg_lang=trg_lang,
            split=kwargs["split"],
            has_trg=has_trg,
            tokenizer=tokenizer,
            sequence_encoder=sequence_encoder,
            random_subset=random_subset,
        )
        # load data
        self.dataset = self.load_data(path, **kwargs)
        self._kwargs = kwargs  # should contain arguments passed to `load_dataset()`
        self._kwargs["path"] = path

    def load_data(self, path: str, **kwargs) -> Any:
        # pylint: disable=import-outside-toplevel
        try:
            from datasets import load_dataset
            return load_dataset(path, **kwargs)

        except ImportError as e:
            logger.error(e)
            raise ImportError from e

    def sample_random_subset(self, seed: int = 42) -> None:
        super().sample_random_subset(seed)  # check validity

        # resample every epoch: seed += epoch_no
        self.dataset = self.dataset.shuffle(seed=seed).select(range(self.random_subset))

    def reset_random_subset(self) -> None:
        # reload from cache
        self.dataset = self.load_data(**self._kwargs)

    def get_item(self, idx: int, lang: str, is_train: bool = None) -> List[str]:
        # lookup
        line = self.dataset[idx]
        assert lang in line, (line, lang)

        # tokenize
        is_train = self.split == "train" if is_train is None else is_train
        item = self.tokenizer[lang](line[lang], is_train=is_train)
        return item

    def get_list(self, lang: str, tokenized: bool = False) -> List[str]:
        if tokenized:
            return self.dataset.map(
                lambda item: {f"tok_{lang}": self.tokenizer[lang](item[lang])},
                desc=f"Tokenizing {lang}...",
            )[f"tok_{lang}"]
        return self.dataset[lang]

    def __len__(self) -> int:
        return self.dataset.num_rows

    def __repr__(self) -> str:
        ret = (f"{self.__class__.__name__}(len={self.__len__()}, "
               f"src_lang={self.src_lang}, trg_lang={self.trg_lang}, "
               f"has_trg={self.has_trg}, random_subset={self.random_subset}")
        for k, v in self._kwargs.items():
            ret += f", {k}={v}"
        ret += ")"
        return ret


class HuggingfaceDataset(BaseHuggingfaceDataset):
    """
    Wrapper for Huggingface's `datasets.features.Translation` class
    cf.) https://github.com/huggingface/datasets/blob/master/src/datasets/features/translation.py
    """  # noqa

    def load_data(self, path: str, **kwargs) -> Any:
        dataset = super().load_data(path=path, **kwargs)

        lang_pair = dataset.features["translation"].languages
        assert self.src_lang in lang_pair, (self.src_lang, lang_pair)
        if self.has_trg:
            assert self.trg_lang in lang_pair, (self.trg_lang, lang_pair)

        def _pre_process(item):
            sl = self.src_lang
            tl = self.trg_lang
            ret = {sl: self.tokenizer[sl].pre_process(item[sl])}
            if self.has_trg:
                ret[tl] = self.tokenizer[tl].pre_process(item[tl])
            return ret

        columns = {
            f"translation.{self.src_lang}": self.src_lang,
            f"translation.{self.trg_lang}": self.trg_lang,
        }
        return (dataset.flatten().rename_columns(columns).map(_pre_process,
                                                              desc="Preprocessing..."))


def build_dataset(
    dataset_type: str,
    path: str,
    src_lang: str,
    trg_lang: str,
    split: str,
    tokenizer: Dict = None,
    sequence_encoder: Dict = None,
    random_subset: int = -1,
    **kwargs,
):
    """
    Builds a dataset.

    :param dataset_type: (str) one of {`plain`, `tsv`, `stream`, `huggingface`}
    :param path: (str) either a local file name or
        dataset name to download from remote
    :param src_lang: (str) language code for source
    :param trg_lang: (str) language code for target
    :param split: (str) one of {`train`, `dev`, `test`}
    :param tokenizer: tokenizer objects for both source and target
    :param sequence_encoder: encoding functions for both source and target
    :param random_subset: (int) number of random subset; -1 means no subsampling
    :return: loaded Dataset
    """
    dataset = None
    has_trg = True  # by default, we expect src-trg pairs

    if dataset_type == "plain":
        if not Path(path).with_suffix(f"{Path(path).suffix}.{trg_lang}").is_file():
            # no target is given -> create dataset from src only
            has_trg = False
        dataset = PlaintextDataset(
            path=path,
            src_lang=src_lang,
            trg_lang=trg_lang,
            split=split,
            has_trg=has_trg,
            tokenizer=tokenizer,
            sequence_encoder=sequence_encoder,
            random_subset=random_subset,
            **kwargs,
        )
    elif dataset_type == "tsv":
        dataset = TsvDataset(
            path=path,
            src_lang=src_lang,
            trg_lang=trg_lang,
            split=split,
            has_trg=has_trg,
            tokenizer=tokenizer,
            sequence_encoder=sequence_encoder,
            random_subset=random_subset,
            **kwargs,
        )
    elif dataset_type == "stream":
        dataset = StreamDataset(
            path=path,
            src_lang=src_lang,
            trg_lang=trg_lang,
            split="test",
            has_trg=False,
            tokenizer=tokenizer,
            sequence_encoder=sequence_encoder,
            random_subset=-1,
            **kwargs,
        )
    elif dataset_type == "huggingface":
        # "split" should be specified in kwargs
        kwargs["split"] = "validation" if split == "dev" else split
        dataset = HuggingfaceDataset(
            path=path,
            src_lang=src_lang,
            trg_lang=trg_lang,
            has_trg=has_trg,
            tokenizer=tokenizer,
            sequence_encoder=sequence_encoder,
            random_subset=random_subset,
            **kwargs,
        )
    else:
        ConfigurationError(f"{dataset_type}: Unknown dataset type.")
    return dataset
