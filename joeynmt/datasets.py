# coding: utf-8
"""
Dataset module
"""
import logging
import random
import sys
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from torch.utils.data import (
    BatchSampler,
    DataLoader,
    Dataset,
    RandomSampler,
    Sampler,
    SequentialSampler,
)
from torch.utils.data.distributed import DistributedSampler

from joeynmt.batch import Batch
from joeynmt.config import ConfigurationError
from joeynmt.constants import EOS_ID, PAD_ID, SEP_TOKEN
from joeynmt.helpers import read_list_from_file
from joeynmt.tokenizers import BasicTokenizer

logger = logging.getLogger(__name__)
CPU_DEVICE = torch.device("cpu")


class DataStats:

    def __init__(self):
        self.epoch = 0
        self.num_valid = 0
        self.num_invalid = 0
        self.max_length = 0
        self.min_length = sys.maxsize

    def reset(self):
        self.num_valid = 0
        self.num_invalid = 0
        self.max_length = 0
        self.min_length = sys.maxsize

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}(epoch={self.epoch}, "
                f"num_valid={self.num_valid}, num_invalid={self.num_invalid}, "
                f"max_length={self.max_length}, min_length={self.min_length})")


class BaseDataset(Dataset):
    """
    BaseDataset which loads and looks up data.
    - holds pointer to tokenizers, encoding functions.

    :param path: path to data directory
    :param src_lang: source language code, i.e. `en`
    :param trg_lang: target language code, i.e. `de`
    :param has_trg: bool indicator if trg exists
    :param has_prompt: bool indicator if prompt exists
    :param split: bool indicator for train set or not
    :param tokenizer: tokenizer objects
    :param sequence_encoder: encoding functions
    """

    def __init__(
        self,
        path: str,
        src_lang: str,
        trg_lang: str,
        split: str = "train",
        has_trg: bool = True,
        has_prompt: Dict[str, bool] = None,
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
        self.has_prompt = _place_holder if has_prompt is None else has_prompt

        # for random subsampling
        self.random_subset = random_subset
        self.stats = DataStats()  # (random seed) called by sampler.set_epoch()
        self.indices = None  # range(self.__len__())

    def load_data(self, path: Path, **kwargs) -> Any:
        """
        load data
            - preprocessing (lowercasing etc) is applied here.
        """
        raise NotImplementedError

    def get_item(self, idx: int, lang: str, is_train: bool = None) -> List[str]:
        """
        seek one src/trg item of given index.
            - tokenization is applied here.
            - length-filtering, bpe-dropout etc also triggered if self.split == "train"
        """
        line, prompt = self.lookup_item(idx, lang)
        is_train = self.split == "train" if is_train is None else is_train
        item = self.tokenizer[lang](line, is_train=is_train)

        if self.has_prompt[lang] and prompt is not None:
            prompt = self.tokenizer[lang](prompt, is_train=False)
            item = prompt + [SEP_TOKEN] + item
            max_length = self.tokenizer[lang].max_length
            if 0 < max_length < len(item):
                item = item[-max_length:]  # truncate prompt
        return item

    def lookup_item(self, idx: int, lang: str) -> Tuple[str, str]:
        raise NotImplementedError

    def __getitem__(self, idx: Union[int, str]) -> Tuple[List[str], List[str]]:
        """lookup one item pair of given index."""
        src, trg = None, None
        src = self.get_item(idx=idx, lang=self.src_lang)
        if self.has_trg or self.has_prompt[self.trg_lang]:
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

    def collate_fn(
        self,
        batch: List[Tuple],
        eos_index: int = EOS_ID,
        pad_index: int = PAD_ID,
        device: torch.device = CPU_DEVICE,
    ) -> Batch:
        """
        Custom collate function.
        See https://pytorch.org/docs/stable/data.html#dataloader-collate-fn for details.
        Please override the batch class here. (not in TrainManager)

        :param batch:
        :param eos_index:
        :param pad_index:
        :param device:
        :return: joeynmt batch object
        """

        def _is_valid(s, t):
            # pylint: disable=no-else-return
            if self.has_trg or self.has_prompt[self.trg_lang]:
                return s is not None and t is not None
            else:
                return s is not None

        batch = [(s, t) for s, t in batch if _is_valid(s, t)]
        src_list, trg_list = zip(*batch)
        assert len(batch) == len(src_list), (len(batch), len(src_list))
        assert all(s is not None for s in src_list), src_list
        src, src_length, src_prompt_mask = self.sequence_encoder[self.src_lang](
            src_list, bos=False, eos=True)

        if self.has_trg or self.has_prompt[self.trg_lang]:
            assert all(t is not None for t in trg_list), trg_list
            trg, trg_length, trg_prompt_mask = self.sequence_encoder[self.trg_lang](
                trg_list, bos=True, eos=self.has_trg)  # no EOS if not self.has_trg
        else:
            assert all(t is None for t in trg_list)
            trg, trg_length, trg_prompt_mask = None, None, None

        return Batch(
            src=torch.tensor(src).long(),
            src_length=torch.tensor(src_length).long(),
            src_prompt_mask=torch.tensor(src_prompt_mask).long() if self.has_prompt[self.src_lang] else None,
            trg=torch.tensor(trg).long() if trg else None,
            trg_prompt_mask=torch.tensor(trg_prompt_mask).long() if self.has_prompt[self.trg_lang] else None,
            device=device,
            eos_index=eos_index,
            pad_index=pad_index,
            is_train=self.split == "train",
        )

    def make_iter(
        self,
        batch_size: int,
        batch_type: str = "sentence",
        seed: int = 42,
        shuffle: bool = False,
        num_workers: int = 0,
        eos_index: int = EOS_ID,
        pad_index: int = PAD_ID,
        device: torch.device = CPU_DEVICE,
        use_ddp: bool = False,
    ) -> DataLoader:
        """
        Returns a torch DataLoader for a torch Dataset. (no bucketing)

        :param batch_size: size of the batches the iterator prepares
        :param batch_type: measure batch size by sentence count or by token count
        :param seed: random seed for shuffling
        :param shuffle: whether to shuffle the data before each epoch
            (for testing, no effect even if set to True)
        :param num_workers: number of cpus for multiprocessing
        :param eos_index:
        :param pad_index:
        :param device:
        :return: torch DataLoader
        """
        # sampler
        sampler: Sampler[int]  # will raise an integer
        if use_ddp:
            sampler = DistributedSampler(self,
                                         shuffle=(shuffle and self.split == "train"),
                                         seed=seed)
        else:
            if shuffle and self.split == "train":
                generator = torch.Generator()
                generator.manual_seed(seed)
                sampler = RandomSampler(self, generator=generator)
            else:
                if 0 < self.random_subset:
                    generator = torch.Generator()
                    generator.manual_seed(seed)
                    sampler = RandomSubsetSampler(self, generator=generator, seed=seed)
                else:
                    sampler = SequentialSampler(self)

        # batch generator
        if batch_type == "sentence":
            batch_sampler = SentenceBatchSampler(sampler,
                                                 batch_size=batch_size,
                                                 drop_last=False,
                                                 seed=seed)
        elif batch_type == "token":
            batch_sampler = TokenBatchSampler(sampler,
                                              batch_size=batch_size,
                                              drop_last=False,
                                              seed=seed)
        else:
            raise ConfigurationError(f"{batch_type}: Unknown batch type")

        # initialize epoch no
        if 0 < batch_sampler.sampler.data_source.random_subset:
            batch_sampler.set_epoch(self.stats.epoch)
        elif hasattr(batch_sampler.sampler, 'reset'):
            batch_sampler.sampler.data_source.reset()
            batch_sampler.sampler.reset()

        assert self.sequence_encoder[self.src_lang] is not None
        if self.has_trg:
            assert self.sequence_encoder[self.trg_lang] is not None
        # else:
        #    self.sequence_encoder[self.trg_lang] = None

        # data iterator
        return DataLoader(
            dataset=self,
            batch_sampler=batch_sampler,
            collate_fn=partial(self.collate_fn, eos_index=eos_index, pad_index=pad_index, device=device),
            num_workers=num_workers,
        )

    def __len__(self) -> int:
        raise NotImplementedError

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}(split={self.split}, len={self.__len__()}, "
                f"src_lang={self.src_lang}, trg_lang={self.trg_lang}, "
                f"has_trg={self.has_trg}, random_subset={self.random_subset}, "
                f"has_src_prompt={self.has_prompt[self.src_lang]}, "
                f"has_trg_prompt={self.has_prompt[self.trg_lang]})")


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
        has_prompt: Dict[str, bool] = None,
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
            has_prompt=has_prompt,
            tokenizer=tokenizer,
            sequence_encoder=sequence_encoder,
            random_subset=random_subset,
        )

        # load data
        self.data = self.load_data(path, **kwargs)
        self.indices = range(self.__len__())

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

    def lookup_item(self, idx: int, lang: str) -> Tuple[str, str]:
        try:
            line = self.data[lang][idx]
            prompt = (self.data[f"{lang}_prompt"][idx]
                      if f"{lang}_prompt" in self.data else None)
            return line, prompt
        except Exception as e:
            logger.error(idx, e)
            raise ValueError from e

    def get_list(self,
                 lang: str,
                 tokenized: bool = False) -> Union[List[str], List[List[str]]]:
        """
        Return list of preprocessed sentences in the given language.
        (not length-filtered, no bpe-dropout)
        """
        item_list = []
        for idx in self.indices:
            item, _ = self.lookup_item(idx, lang)
            if tokenized:
                item = self.tokenizer[lang](item, is_train=False)
            item_list.append(item)
        assert len(self.indices) == len(item_list), (len(self.indices), len(item_list))
        return item_list

    def __len__(self) -> int:
        return len(self.data[self.src_lang])


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
        has_prompt: Dict[str, bool] = None,
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
            has_prompt=has_prompt,
            tokenizer=tokenizer,
            sequence_encoder=sequence_encoder,
            random_subset=random_subset,
        )

        # load tsv file
        self.df = self.load_data(path, **kwargs)
        self.indices = range(self.__len__())

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
                index_col=None,
            )
            df = df.dropna()
            df = df.reset_index()

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
            if f"{self.src_lang}_prompt" in df.columns:
                self.has_prompt[self.src_lang] = True
                df[f"{self.src_lang}_prompt"] = df[f"{self.src_lang}_prompt"].apply(
                    self.tokenizer[self.src_lang].pre_process, allow_empty=True)
            if f"{self.trg_lang}_prompt" in df.columns:
                self.has_prompt[self.trg_lang] = True
                df[f"{self.trg_lang}_prompt"] = df[f"{self.trg_lang}_prompt"].apply(
                    self.tokenizer[self.trg_lang].pre_process, allow_empty=True)
            return df

        except ImportError as e:
            logger.error(e)
            raise ImportError from e

    def lookup_item(self, idx: int, lang: str) -> Tuple[str, str]:
        try:
            row = self.df.iloc[idx]
            line = row[lang]
            prompt = row.get(f"{lang}_prompt", None)
            return line, prompt
        except Exception as e:
            logger.error(idx, e)
            raise ValueError from e

    def get_list(self,
                 lang: str,
                 tokenized: bool = False) -> Union[List[str], List[List[str]]]:
        df = self.df.iloc[self.indices]
        return (df[lang].apply(self.tokenizer[lang]).to_list()
                if tokenized else df[lang].to_list())

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
        has_prompt: Dict[str, bool] = None,
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
            has_prompt=has_prompt,
            tokenizer=tokenizer,
            sequence_encoder=sequence_encoder,
            random_subset=random_subset,
        )
        # place holder
        self.cache = {}

    def set_item(self, src_line: str,
                 trg_line: Optional[str] = None,
                 src_prompt: Optional[str] = None,
                 trg_prompt: Optional[str] = None) -> None:
        """
        Set input text to the cache.

        :param src_line: (non-empty) str
        :param trg_line: Optional[str]
        :param src_prompt: Optional[str]
        :param trg_prompt: Optional[str]
        """
        assert isinstance(src_line, str) and src_line.strip() != "", \
            "The input sentence is empty! Please make sure " \
            "that you are feeding a valid input."

        idx = len(self.cache)
        src_line = self.tokenizer[self.src_lang].pre_process(src_line)

        if self.has_trg:
            trg_line = self.tokenizer[self.trg_lang].pre_process(trg_line)

        if src_prompt:
            self.has_prompt[self.src_lang] = True
            src_prompt = self.tokenizer[self.src_lang].pre_process(src_prompt, allow_empty=True)

        if trg_prompt:
            self.has_prompt[self.trg_lang] = True
            trg_prompt = self.tokenizer[self.trg_lang].pre_process(trg_prompt, allow_empty=True)

        self.cache[idx] = (src_line, trg_line, src_prompt, trg_prompt)

    def lookup_item(self, idx: int, lang: str) -> Tuple[str, str]:
        try:
            assert idx in self.cache, (idx, self.cache)
            assert lang in [self.src_lang, self.trg_lang]
            if lang == self.trg_lang:
                assert self.has_trg

            src_line, trg_line, src_prompt, trg_prompt = self.cache[idx]
            if lang == self.src_lang:
                return src_line, src_prompt
            elif lang == self.trg_lang:
                return trg_line, trg_prompt
        except Exception as e:
            logger.error(idx, e)
            raise ValueError from e

    def __len__(self) -> int:
        return len(self.cache)

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}(split={self.split}, len={len(self.cache)}, "
                f"src_lang={self.src_lang}, trg_lang={self.trg_lang}, "
                f"has_trg={self.has_trg}, random_subset={self.random_subset}, "
                f"has_src_prompt={self.has_prompt[self.src_lang]}, "
                f"has_trg_prompt={self.has_prompt[self.trg_lang]})")


class BaseHuggingfaceDataset(BaseDataset):
    """
    Wrapper for Huggingface's dataset object
    cf.) https://huggingface.co/docs/datasets
    """
    COLUMN_NAME = "sentence"  # dummy column name. should be overriden.

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

        self.indices = range(self.__len__())

    def load_data(self, path: str, **kwargs) -> Any:
        # pylint: disable=import-outside-toplevel
        try:
            from datasets import Dataset as Dataset_hf
            from datasets import DatasetDict, config, load_dataset, load_from_disk
            if Path(path, config.DATASET_STATE_JSON_FILENAME).exists() \
                    or Path(path, config.DATASETDICT_JSON_FILENAME).exists():
                hf_dataset = load_from_disk(path)
                if isinstance(hf_dataset, DatasetDict):
                    assert kwargs["split"] in hf_dataset
                    hf_dataset = hf_dataset[kwargs["split"]]
            else:
                hf_dataset = load_dataset(path, **kwargs)
            assert isinstance(hf_dataset, Dataset_hf)
            assert self.COLUMN_NAME in hf_dataset.features
            return hf_dataset

        except ImportError as e:
            logger.error(e)
            raise ImportError from e

    def lookup_item(self, idx: int, lang: str) -> Tuple[str, str]:
        try:
            line = self.dataset[idx]
            assert lang in line[self.COLUMN_NAME], (line, lang)
            prompt = line[f"{lang}_prompt"] if f"{lang}_prompt" in line else None
            return line[self.COLUMN_NAME][lang], prompt
        except Exception as e:
            logger.error(idx, e)
            raise ValueError from e

    def get_list(self, lang: str, tokenized: bool = False) -> List[str]:
        dataset = self.dataset.filter(lambda x, idx: idx in self.indices,
                                      with_indices=True)
        assert len(dataset) == len(self.indices), (len(dataset), len(self.indices))
        if tokenized:

            def _tok(item):
                item[f'tok_{lang}'] = self.tokenizer[lang](item[self.COLUMN_NAME][lang])
                return item

            return dataset.map(_tok, desc=f"Tokenizing {lang}...")[f'tok_{lang}']
        return dataset.flatten()[f'{self.COLUMN_NAME}.{lang}']

    def __len__(self) -> int:
        return self.dataset.num_rows

    def __repr__(self) -> str:
        ret = (f"{self.__class__.__name__}(len={self.__len__()}, "
               f"src_lang={self.src_lang}, trg_lang={self.trg_lang}, "
               f"has_trg={self.has_trg}, random_subset={self.random_subset}, "
               f"has_src_prompt={self.has_prompt[self.src_lang]}, "
               f"has_trg_prompt={self.has_prompt[self.trg_lang]}")
        for k, v in self._kwargs.items():
            ret += f", {k}={v}"
        ret += ")"
        return ret


class HuggingfaceTranslationDataset(BaseHuggingfaceDataset):
    """
    Wrapper for Huggingface's `datasets.features.Translation` class
    cf.) https://github.com/huggingface/datasets/blob/master/src/datasets/features/translation.py
    """  # noqa
    COLUMN_NAME = "translation"

    def load_data(self, path: str, **kwargs) -> Any:
        dataset = super().load_data(path=path, **kwargs)
        # pylint: disable=import-outside-toplevel
        try:
            from datasets.features import Translation as Translation_hf
            assert isinstance(dataset.features[self.COLUMN_NAME], Translation_hf), \
                f"Data type mismatch. Please cast `{self.COLUMN_NAME}` column to " \
                "datasets.features.Translation class."
            assert self.src_lang in dataset.features[self.COLUMN_NAME].languages
            if self.has_trg:
                assert self.trg_lang in dataset.features[self.COLUMN_NAME].languages

        except ImportError as e:
            logger.error(e)
            raise ImportError from e

        # preprocess (lowercase, pretokenize, etc.) + validity check
        def _pre_process(item):
            sl = self.src_lang
            tl = self.trg_lang
            item[self.COLUMN_NAME][sl] = self.tokenizer[sl].pre_process(
                item[self.COLUMN_NAME][sl])
            if self.has_trg:
                item[self.COLUMN_NAME][tl] = self.tokenizer[tl].pre_process(
                    item[self.COLUMN_NAME][tl])
            if self.has_prompt[sl]:
                item[f"{sl}_prompt"] = self.tokenizer[sl].pre_process(
                    item[f"{sl}_prompt"], allow_empty=True)
            if self.has_prompt[tl]:
                item[f"{tl}_prompt"] = self.tokenizer[tl].pre_process(
                    item[f"{tl}_prompt"], allow_empty=True)
            return item

        def _drop_nan(item):
            src_item = item[self.COLUMN_NAME][self.src_lang]
            is_src_valid = src_item is not None and len(src_item) > 0
            if self.has_trg:
                trg_item = item[self.COLUMN_NAME][self.trg_lang]
                is_trg_valid = trg_item is not None and len(trg_item) > 0
                return is_src_valid and is_trg_valid
            return is_src_valid

        dataset = dataset.filter(_drop_nan, desc="Dropping NaN...")
        dataset = dataset.map(_pre_process, desc="Preprocessing...")
        return dataset


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
        if "split" not in kwargs:
            kwargs["split"] = "validation" if split == "dev" else split
        dataset = HuggingfaceTranslationDataset(
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
        raise ConfigurationError(f"{dataset_type}: Unknown dataset type.")
    return dataset


class SentenceBatchSampler(BatchSampler):
    """
    Wraps another sampler to yield a mini-batch of indices based on num of instances.
    An instance longer than dataset.max_len will be filtered out.

    :param sampler: Base sampler. Can be any iterable object
    :param batch_size: Size of mini-batch.
    :param drop_last: If `True`, the sampler will drop the last batch if its size
        would be less than `batch_size`
    """

    def __init__(self, sampler: Sampler, batch_size: int, drop_last: bool, seed: int):
        super().__init__(sampler, batch_size, drop_last)
        self.seed = seed

    @property
    def num_samples(self) -> int:
        d = self.sampler.data_source
        if 0 < d.random_subset:
            assert (
                d.split != "test" and d.random_subset < len(d)
            ), f"Can only subsample from train or dev set larger than {d.random_subset}."
            return d.random_subset
        return len(d)

    def __iter__(self):
        batch = []
        d = self.sampler.data_source
        # d.indices = [] for dev dataset (RandomSubsetSampler) only

        for counter, idx in enumerate(self.sampler):
            # subsampling
            if (isinstance(self.sampler, RandomSampler) and 0 < d.random_subset < len(d)
                    and d.random_subset < counter):
                break

            src, trg = d[idx]  # pylint: disable=unused-variable
            if src is not None:  # otherwise drop instance
                batch.append(idx)

                # data stats
                d.stats.num_valid += 1
                src_len = 0 if src is None else len(src)
                trg_len = 0 if trg is None else len(trg)
                if src_len + trg_len > d.stats.max_length:
                    d.stats.max_length = src_len + trg_len
                if src_len + trg_len < d.stats.min_length:
                    d.stats.min_length = src_len + trg_len

                if len(batch) >= self.batch_size:
                    # d.indices.extend(batch)
                    yield batch
                    batch = []
            else:
                d.stats.num_invalid += 1
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self) -> int:
        if self.drop_last:
            return self.num_samples // self.batch_size
        else:
            return (self.num_samples + self.batch_size - 1) // self.batch_size

    def set_epoch(self, epoch: int) -> None:
        self.sampler.data_source.stats.epoch = epoch
        if hasattr(self.sampler, 'set_epoch'):
            self.sampler.set_epoch(epoch)
        elif hasattr(self.sampler, 'generator'):
            self.sampler.generator.manual_seed(self.seed + epoch)

        if self.num_samples < len(self.sampler.data_source):
            logger.info("Sample random subset from %s set: n=%d, seed=%d",
                        self.sampler.data_source.split, self.num_samples,
                        self.seed + epoch)
        self.sampler.data_source.stats.reset()

    def reset(self) -> None:
        if hasattr(self.sampler, 'reset'):
            self.sampler.reset()
        self.sampler.data_source.random_subset = -1


class TokenBatchSampler(SentenceBatchSampler):
    """
    Wraps another sampler to yield a mini-batch of indices based on num of tokens
    (incl. padding). An instance longer than dataset.max_len or shorter than
    dataset.min_len will be filtered out.
    * no bucketing implemented

    :param sampler: Base sampler. Can be any iterable object
    :param batch_size: Size of mini-batch.
    :param drop_last: If `True`, the sampler will drop the last batch if
            its size would be less than `batch_size`
    """

    def __iter__(self):
        batch = []
        max_tokens = 0
        d = self.sampler.data_source

        for counter, idx in enumerate(self.sampler):
            # subsampling
            if 0 < d.random_subset < len(d) and d.random_subset < counter:
                break
            src, trg = d[idx]  # call __getitem__()
            if src is not None:  # otherwise drop instance
                src_len = 0 if src is None else len(src)
                trg_len = 0 if trg is None else len(trg)
                n_tokens = 0 if src_len == 0 else max(src_len + 1, trg_len + 1)
                batch.append(idx)

                # data stats
                d.stats.num_valid += 1
                if src_len + trg_len > d.stats.max_length:
                    d.stats.max_length = src_len + trg_len
                if src_len + trg_len < d.stats.min_length:
                    d.stats.min_length = src_len + trg_len

                if n_tokens > max_tokens:
                    max_tokens = n_tokens
                if max_tokens * len(batch) >= self.batch_size:
                    yield batch
                    batch = []
                    max_tokens = 0
            else:
                d.stats.num_invalid += 1
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        raise NotImplementedError


class RandomSubsetSampler(SequentialSampler):
    """Samples subset randomly from a given data_source without replacement,
       and yields subset elements sequentially.

    :param data_source (Dataset): dataset to sample from
    :param generator (Generator): Generator used in sampling.
    """

    def __init__(self, data_source: Dataset, generator: torch.Generator, seed: int):
        super().__init__(data_source)
        self.generator = generator
        self.seed = seed
        # self._subsample() should be called in set_epoch()

    @property
    def num_samples(self) -> int:
        d = self.data_source
        if 0 < d.random_subset:
            assert (
                d.split != "test" and d.random_subset < len(d)
            ), f"Can only subsample from train or dev set larger than {d.random_subset}."
            return d.random_subset
        return len(self.data_source.indices)

    def __iter__(self):
        return iter(self.data_source.indices)

    def __len__(self) -> int:
        return self.num_samples

    def _subsample(self):
        orig_len = len(self.data_source)
        subset_len = self.data_source.random_subset
        if 0 < subset_len < orig_len:
            subset = torch.randperm(n=orig_len,
                                    generator=self.generator).tolist()[:subset_len]
            self.data_source.indices = sorted(subset)
            assert len(subset) == self.num_samples

    def reset(self):
        self.data_source.indices = range(len(self.data_source))

    def set_epoch(self, epoch: int) -> None:
        self.generator.manual_seed(self.seed + epoch)
        self._subsample()
