# coding: utf-8
"""
Dataset module
"""
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from torch.utils.data import BatchSampler, DataLoader, Dataset, Sampler

from joeynmt.batch import Batch
from joeynmt.config import ConfigurationError
from joeynmt.helpers import read_list_from_file
from joeynmt.helpers_for_ddp import (
    DistributedSubsetSampler,
    RandomSubsetSampler,
    get_logger,
    use_ddp,
)
from joeynmt.tokenizers import BasicTokenizer

logger = get_logger(__name__)
CPU_DEVICE = torch.device("cpu")


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

    # pylint: disable=too-many-instance-attributes
    def __init__(
        self,
        path: str,
        src_lang: str,
        trg_lang: str,
        split: str = "train",
        has_trg: bool = False,
        has_prompt: Dict[str, bool] = None,
        tokenizer: Dict[str, BasicTokenizer] = None,
        sequence_encoder: Dict[str, Callable] = None,
        random_subset: int = -1
    ):

        self.path = path
        self.src_lang = src_lang
        self.trg_lang = trg_lang
        self.has_trg = has_trg
        self.split = split
        if self.split == "train":
            assert self.has_trg

        self.tokenizer = tokenizer
        self.sequence_encoder = sequence_encoder
        self.has_prompt = has_prompt

        # for random subsampling
        self.random_subset = random_subset
        self.indices = None  # range(self.__len__())
        # Note: self.indices is kept sorted, even if shuffle = True in make_iter()
        # (Sampler yields permuted indices)
        self.seed = 1  # random seed for generator

    def reset_indices(self, random_subset: int = None):
        # should be called after data are loaded.
        # otherwise self.__len__() is undefined.
        self.indices = list(range(self.__len__())) if self.__len__() > 0 else []
        if random_subset is not None:
            self.random_subset = random_subset

        if 0 < self.random_subset:
            assert (self.split != "test" and self.random_subset < self.__len__()), \
                ("Can only subsample from train or dev set "
                 f"larger than {self.random_subset}.")

    def load_data(self, path: Path, **kwargs) -> Any:
        """
        load data
            - preprocessing (lowercasing etc) is applied here.
        """
        raise NotImplementedError

    def get_item(self, idx: int, lang: str, is_train: bool = None) -> List[str]:
        """
        seek one src/trg item of the given index.
            - tokenization is applied here.
            - length-filtering, bpe-dropout etc also triggered if self.split == "train"
        """

        # workaround if tokenizer prepends an extra escape symbol before lang_tang ...
        def _remove_escape(item):
            if (
                item is not None and self.tokenizer[lang] is not None
                and item[0] == self.tokenizer[lang].SPACE_ESCAPE
                and item[1] in self.tokenizer[lang].lang_tags
            ):
                return item[1:]
            return item

        line, prompt = self.lookup_item(idx, lang)
        is_train = self.split == "train" if is_train is None else is_train
        item = _remove_escape(self.tokenizer[lang](line, is_train=is_train))

        if self.has_prompt[lang] and prompt is not None:
            prompt = _remove_escape(self.tokenizer[lang](prompt, is_train=False))
            item = item if item is not None else []

            max_length = self.tokenizer[lang].max_length
            if 0 < max_length < len(prompt) + len(item) + 1:
                # truncate prompt
                offset = max_length - len(item) - 1
                if prompt[0] in self.tokenizer[lang].lang_tags:
                    prompt = [prompt[0]] + prompt[-(offset - 1):]
                else:
                    prompt = prompt[-offset:]

            item = prompt + [self.tokenizer[lang].sep_token] + item
        return item

    def lookup_item(self, idx: int, lang: str) -> Tuple[str, str]:
        raise NotImplementedError

    def __getitem__(self, idx: Union[int, str]) -> Tuple[int, List[str], List[str]]:
        """
        lookup one item pair of the given index.

        :param idx: index of the instance to lookup
        :return:
            - index  # needed to recover the original order
            - tokenized src sentences
            - tokenized trg sentences
        """
        if idx > self.__len__():
            raise KeyError

        src, trg = None, None
        src = self.get_item(idx=idx, lang=self.src_lang)
        if self.has_trg or self.has_prompt[self.trg_lang]:
            trg = self.get_item(idx=idx, lang=self.trg_lang)
            if trg is None:
                src = None
        return idx, src, trg

    def get_list(self,
                 lang: str,
                 tokenized: bool = False,
                 subsampled: bool = True) -> Union[List[str], List[List[str]]]:
        """get data column-wise."""
        raise NotImplementedError

    @property
    def src(self) -> List[str]:
        """get detokenized preprocessed data in src language."""
        return self.get_list(self.src_lang, tokenized=False, subsampled=True)

    @property
    def trg(self) -> List[str]:
        """get detokenized preprocessed data in trg language."""
        return (
            self.get_list(self.trg_lang, tokenized=False, subsampled=True)
            if self.has_trg else []
        )

    def collate_fn(
        self,
        batch: List[Tuple],
        pad_index: int,
        eos_index: int,
        device: torch.device = CPU_DEVICE,
    ) -> Batch:
        """
        Custom collate function.
        See https://pytorch.org/docs/stable/data.html#dataloader-collate-fn for details.
        Please override the batch class here. (not in TrainManager)

        :param batch:
        :param pad_index:
        :param eos_index:
        :param device:
        :return: joeynmt batch object
        """
        idx, src_list, trg_list = zip(*batch)
        assert len(batch) == len(src_list) == len(trg_list), (len(batch), len(src_list))
        assert all(s is not None for s in src_list), src_list
        src, src_length, src_prompt_mask = self.sequence_encoder[
            self.src_lang](src_list, bos=False, eos=True)

        if self.has_trg or self.has_prompt[self.trg_lang]:
            if self.has_trg:
                assert all(t is not None for t in trg_list), trg_list
            trg, _, trg_prompt_mask = self.sequence_encoder[self.trg_lang](
                trg_list, bos=True, eos=self.has_trg
            )  # no EOS if not self.has_trg
        else:
            assert all(t is None for t in trg_list)
            trg, trg_prompt_mask = None, None  # Note: we don't need trg_length!

        return Batch(
            src=torch.tensor(src).long(),
            src_length=torch.tensor(src_length).long(),
            src_prompt_mask=(
                torch.tensor(src_prompt_mask).long()
                if self.has_prompt[self.src_lang] else None
            ),
            trg=torch.tensor(trg).long() if trg else None,
            trg_prompt_mask=(
                torch.tensor(trg_prompt_mask).long()
                if self.has_prompt[self.trg_lang] else None
            ),
            indices=torch.tensor(idx).long(),
            device=device,
            pad_index=pad_index,
            eos_index=eos_index,
            is_train=self.split == "train",
        )

    def make_iter(
        self,
        batch_size: int,
        batch_type: str = "sentence",
        seed: int = 42,
        shuffle: bool = False,
        num_workers: int = 0,
        pad_index: int = 1,
        eos_index: int = 3,
        device: torch.device = CPU_DEVICE,
        generator_state: torch.Tensor = None,
    ) -> DataLoader:
        """
        Returns a torch DataLoader for a torch Dataset. (no bucketing)

        :param batch_size: size of the batches the iterator prepares
        :param batch_type: measure batch size by sentence count or by token count
        :param seed: random seed for shuffling
        :param shuffle: whether to shuffle the order of sequences before each epoch
                        (for testing, no effect even if set to True; generator is
                        still used for random subsampling, but not for permutation!)
        :param num_workers: number of cpus for multiprocessing
        :param pad_index:
        :param eos_index:
        :param device:
        :param generator_state:
        :return: torch DataLoader
        """
        shuffle = shuffle and self.split == "train"

        # for decoding in DDP, we cannot use TokenBatchSampler
        if use_ddp() and self.split != "train":
            assert batch_type == "sentence", self

        generator = torch.Generator()
        generator.manual_seed(seed)
        if generator_state is not None:
            generator.set_state(generator_state)

        # define sampler which yields an integer
        sampler: Sampler[int]
        if use_ddp():  # use ddp
            sampler = DistributedSubsetSampler(
                self, shuffle=shuffle, drop_last=True, generator=generator
            )
        else:
            sampler = RandomSubsetSampler(self, shuffle=shuffle, generator=generator)

        # batch sampler which yields a list of integers
        if batch_type == "sentence":
            batch_sampler = SentenceBatchSampler(
                sampler, batch_size=batch_size, drop_last=False, seed=seed
            )
        elif batch_type == "token":
            batch_sampler = TokenBatchSampler(
                sampler, batch_size=batch_size, drop_last=False, seed=seed
            )
        else:
            raise ConfigurationError(f"{batch_type}: Unknown batch type")

        # initialize generator seed
        batch_sampler.set_seed(seed)  # set seed and resample

        # ensure that sequence_encoder (padding func) exists
        assert self.sequence_encoder[self.src_lang] is not None
        if self.has_trg:
            assert self.sequence_encoder[self.trg_lang] is not None

        # data iterator
        return DataLoader(
            dataset=self,
            batch_sampler=batch_sampler,
            collate_fn=partial(
                self.collate_fn,
                eos_index=eos_index,
                pad_index=pad_index,
                device=device
            ),
            num_workers=num_workers
        )

    def __len__(self) -> int:
        raise NotImplementedError

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(split={self.split}, len={self.__len__()}, "
            f'src_lang="{self.src_lang}", trg_lang="{self.trg_lang}", '
            f"has_trg={self.has_trg}, random_subset={self.random_subset}, "
            f"has_src_prompt={self.has_prompt[self.src_lang]}, "
            f"has_trg_prompt={self.has_prompt[self.trg_lang]})"
        )


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
        split: str = "train",
        has_trg: bool = False,
        has_prompt: Dict[str, bool] = None,
        tokenizer: Dict[str, BasicTokenizer] = None,
        sequence_encoder: Dict[str, Callable] = None,
        random_subset: int = -1,
        **kwargs
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
            random_subset=random_subset
        )

        # load data
        self.data = self.load_data(path, **kwargs)
        self.reset_indices()

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
            prompt = (
                self.data[f"{lang}_prompt"][idx]
                if f"{lang}_prompt" in self.data else None
            )
            return line, prompt
        except Exception as e:
            logger.error(idx, e)
            raise ValueError from e

    def get_list(self,
                 lang: str,
                 tokenized: bool = False,
                 subsampled: bool = True) -> Union[List[str], List[List[str]]]:
        """
        Return list of preprocessed sentences in the given language.
        (not length-filtered, no bpe-dropout)
        """
        indices = self.indices if subsampled else range(self.__len__())
        item_list = []
        for idx in indices:
            item, _ = self.lookup_item(idx, lang)
            if tokenized:
                item = self.tokenizer[lang](item, is_train=False)
            item_list.append(item)
        assert len(indices) == len(item_list), (len(indices), len(item_list))
        return item_list

    def __len__(self) -> int:
        return len(self.data[self.src_lang])


class TsvDataset(BaseDataset):
    """
    TsvDataset which handles data in tsv format.
    - file_name should be specified without extension `.tsv`
    - needs src_lang and trg_lang (i.e. `en`, `de`) in header.
    see: test/data/toy/dev.tsv
    """

    def __init__(
        self,
        path: str,
        src_lang: str,
        trg_lang: str,
        split: str = "train",
        has_trg: bool = False,
        has_prompt: Dict[str, bool] = None,
        tokenizer: Dict[str, BasicTokenizer] = None,
        sequence_encoder: Dict[str, Callable] = None,
        random_subset: int = -1,
        **kwargs
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
            random_subset=random_subset
        )

        # load tsv file
        self.df = self.load_data(path, **kwargs)
        self.reset_indices()

    def load_data(self, path: str, **kwargs) -> Any:
        path = Path(path)
        file_path = path.with_suffix(f"{path.suffix}.tsv")
        assert file_path.is_file(), f"{file_path} not found. Abort."

        try:
            import pandas as pd  # pylint: disable=import-outside-toplevel

            # TODO: use `chunksize` for online data loading.
            df = pd.read_csv(
                file_path.as_posix(),
                sep="\t",
                header=0,
                encoding="utf-8",
                index_col=None
            )
            df = df.dropna()
            df = df.reset_index()

            assert self.src_lang in df.columns
            df[self.src_lang
               ] = df[self.src_lang].apply(self.tokenizer[self.src_lang].pre_process)

            if self.trg_lang not in df.columns:
                self.has_trg = False
                assert self.split == "test"
            if self.has_trg:
                df[self.trg_lang] = df[self.trg_lang].apply(
                    self.tokenizer[self.trg_lang].pre_process
                )
            if f"{self.src_lang}_prompt" in df.columns:
                self.has_prompt[self.src_lang] = True
                df[f"{self.src_lang}_prompt"] = df[f"{self.src_lang}_prompt"].apply(
                    self.tokenizer[self.src_lang].pre_process, allow_empty=True
                )
            if f"{self.trg_lang}_prompt" in df.columns:
                self.has_prompt[self.trg_lang] = True
                df[f"{self.trg_lang}_prompt"] = df[f"{self.trg_lang}_prompt"].apply(
                    self.tokenizer[self.trg_lang].pre_process, allow_empty=True
                )
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
                 tokenized: bool = False,
                 subsampled: bool = True) -> Union[List[str], List[List[str]]]:
        indices = self.indices if subsampled else range(self.__len__())
        df = self.df.iloc[indices]
        return (
            df[lang].apply(self.tokenizer[lang]).to_list()
            if tokenized else df[lang].to_list()
        )

    def __len__(self) -> int:
        return len(self.df)


class StreamDataset(BaseDataset):
    """
    StreamDataset which interacts with stream inputs.
    - called by `translate()` func in `prediction.py`.
    """

    # pylint: disable=unused-argument
    def __init__(
        self,
        path: str,
        src_lang: str,
        trg_lang: str,
        split: str = "test",
        has_trg: bool = False,
        has_prompt: Dict[str, bool] = None,
        tokenizer: Dict[str, BasicTokenizer] = None,
        sequence_encoder: Dict[str, Callable] = None,
        random_subset: int = -1,
        **kwargs
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
            random_subset=random_subset
        )

        # place holder
        self.cache = []

    def _split_at_sep(self, line: str, prompt: str, lang: str, sep_token: str):
        """
        Split string at sep_token

        :param line: (non-empty) input string
        :param prompt: input prompt
        :param lang:
        :param sep_token:
        """
        if (
            sep_token is not None and line is not None and sep_token in line
            and prompt is None
        ):
            line, prompt = line.split(sep_token)

        if line:
            line = self.tokenizer[lang].pre_process(line, allow_empty=False)
        if prompt:
            prompt = self.tokenizer[lang].pre_process(prompt, allow_empty=True)
            self.has_prompt[lang] = True

        return line, prompt

    def set_item(
        self,
        src_line: str,
        trg_line: Optional[str] = None,
        src_prompt: Optional[str] = None,
        trg_prompt: Optional[str] = None
    ) -> None:
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

        src_line, src_prompt = self._split_at_sep(
            src_line, src_prompt, self.src_lang, self.tokenizer[self.src_lang].sep_token
        )
        assert src_line is not None

        trg_line, trg_prompt = self._split_at_sep(
            trg_line, trg_prompt, self.trg_lang, self.tokenizer[self.trg_lang].sep_token
        )
        if self.has_trg:
            assert trg_line is not None

        self.cache.append((src_line, trg_line, src_prompt, trg_prompt))
        self.reset_indices()

    def lookup_item(self, idx: int, lang: str) -> Tuple[str, str]:
        # pylint: disable=no-else-return
        try:
            assert lang in [self.src_lang, self.trg_lang]
            if lang == self.trg_lang:
                assert self.has_trg or self.has_prompt[lang]

            src_line, trg_line, src_prompt, trg_prompt = self.cache[idx]
            if lang == self.src_lang:
                return src_line, src_prompt
            elif lang == self.trg_lang:
                return trg_line, trg_prompt
            else:
                raise ValueError
        except Exception as e:
            logger.error(idx, e)
            raise ValueError from e

    def reset_cache(self):
        self.cache = []
        self.reset_indices()

    def __len__(self) -> int:
        return len(self.cache)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(split={self.split}, len={len(self.cache)}, "
            f'src_lang="{self.src_lang}", trg_lang="{self.trg_lang}", '
            f"has_trg={self.has_trg}, random_subset={self.random_subset}, "
            f"has_src_prompt={self.has_prompt[self.src_lang]}, "
            f"has_trg_prompt={self.has_prompt[self.trg_lang]})"
        )


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
            split=kwargs["split"],
            has_trg=has_trg,
            has_prompt=has_prompt,
            tokenizer=tokenizer,
            sequence_encoder=sequence_encoder,
            random_subset=random_subset,
        )
        # load data
        self.dataset = self.load_data(path, **kwargs)
        self._kwargs = kwargs  # should contain arguments passed to `load_dataset()`

        self.reset_indices()

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
            prompt = line.get(f"{lang}_prompt", None)
            return line[self.COLUMN_NAME][lang], prompt
        except Exception as e:
            logger.error(idx, e)
            raise ValueError from e

    def get_list(self,
                 lang: str,
                 tokenized: bool = False,
                 subsampled: bool = True) -> Union[List[str], List[List[str]]]:
        dataset = self.dataset
        if subsampled:
            dataset = dataset.filter(
                lambda x, idx: idx in self.indices, with_indices=True
            )
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
        ret = (
            f"{self.__class__.__name__}(len={self.__len__()}, "
            f'src_lang="{self.src_lang}", trg_lang="{self.trg_lang}", '
            f"has_trg={self.has_trg}, random_subset={self.random_subset}, "
            f"has_src_prompt={self.has_prompt[self.src_lang]}, "
            f"has_trg_prompt={self.has_prompt[self.trg_lang]}"
        )
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
                item[self.COLUMN_NAME][sl]
            )
            if self.has_trg:
                item[self.COLUMN_NAME][tl] = self.tokenizer[tl].pre_process(
                    item[self.COLUMN_NAME][tl]
                )
            if self.has_prompt[sl]:
                item[f"{sl}_prompt"] = self.tokenizer[sl].pre_process(
                    item[f"{sl}_prompt"], allow_empty=True
                )
            if self.has_prompt[tl]:
                item[f"{tl}_prompt"] = self.tokenizer[tl].pre_process(
                    item[f"{tl}_prompt"], allow_empty=True
                )
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
    has_prompt: Dict = None,
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
    :param has_prompt: prompt indicators
    :param random_subset: (int) number of random subset; -1 means no subsampling
    :return: loaded Dataset
    """
    dataset = None
    has_trg = True  # by default, we expect src-trg pairs
    _placeholder = {src_lang: None, trg_lang: None}
    tokenizer = _placeholder if tokenizer is None else tokenizer
    sequence_encoder = _placeholder if sequence_encoder is None else sequence_encoder
    has_prompt = _placeholder if has_prompt is None else has_prompt

    if dataset_type == "plain":
        if not Path(path).with_suffix(f"{Path(path).suffix}.{trg_lang}").is_file():
            has_trg = False  # no target is given -> create dataset from src only
        dataset = PlaintextDataset(
            path=path,
            src_lang=src_lang,
            trg_lang=trg_lang,
            split=split,
            has_trg=has_trg,
            has_prompt=has_prompt,
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
            has_prompt=has_prompt,
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
            has_prompt=has_prompt,
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
            has_prompt=has_prompt,
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
        """
        Returns number of samples in the dataset.
        This may change during sampling.

        Note: len(dataset) won't change during sampling.
              Use len(dataset) instead, to retrieve the original dataset length.
        """
        assert self.sampler.data_source.indices is not None
        try:
            return len(self.sampler)
        except NotImplementedError as e:  # pylint: disable=unused-variable # noqa: F841
            return len(self.sampler.data_source.indices)

    def __iter__(self):
        batch = []
        d = self.sampler.data_source

        for idx in self.sampler:
            _, src, trg = d[idx]  # pylint: disable=unused-variable
            if src is not None:  # otherwise drop instance
                batch.append(idx)

                if len(batch) >= self.batch_size:
                    yield batch
                    batch = []

        if len(batch) > 0:
            if not self.drop_last:
                yield batch
            else:
                logger.warning(f"Drop indices {batch}.")

    def __len__(self) -> int:
        # pylint: disable=no-else-return
        if self.drop_last:
            return self.num_samples // self.batch_size
        else:
            return (self.num_samples + self.batch_size - 1) // self.batch_size

    def set_seed(self, seed: int) -> None:
        assert seed is not None, seed
        self.sampler.data_source.seed = seed

        if hasattr(self.sampler, 'set_seed'):
            self.sampler.set_seed(seed)  # set seed and resample
        elif hasattr(self.sampler, 'generator'):
            self.sampler.generator.manual_seed(seed)

        if self.num_samples < len(self.sampler.data_source):
            logger.info(
                "Sample random subset from %s data: n=%d, seed=%d",
                self.sampler.data_source.split, self.num_samples, seed
            )

    def reset(self) -> None:
        if hasattr(self.sampler, 'reset'):
            self.sampler.reset()

    def get_state(self):
        if hasattr(self.sampler, 'generator'):
            return self.sampler.generator.get_state()
        return None

    def set_state(self, state) -> None:
        if hasattr(self.sampler, 'generator'):
            self.sampler.generator.set_state(state)


class TokenBatchSampler(SentenceBatchSampler):
    """
    Wraps another sampler to yield a mini-batch of indices based on num of tokens
    (incl. padding). An instance longer than dataset.max_len or shorter than
    dataset.min_len will be filtered out.
    * no bucketing implemented

    .. warning::
        In DDP, we shouldn't use TokenBatchSampler for prediction, because we cannot
        ensure that the data points will be distributed evenly across devices.
        `ddp_merge()` (`dist.all_gather()`) called in `predict()` can get stuck.

    :param sampler: Base sampler. Can be any iterable object
    :param batch_size: Size of mini-batch.
    :param drop_last: If `True`, the sampler will drop the last batch if
            its size would be less than `batch_size`
    """

    def __iter__(self):
        """yields list of indices"""
        batch = []
        max_tokens = 0
        d = self.sampler.data_source

        for idx in self.sampler:
            _, src, trg = d[idx]  # call __getitem__()
            if src is not None:  # otherwise drop instance
                src_len = 0 if src is None else len(src)
                trg_len = 0 if trg is None else len(trg)
                n_tokens = 0 if src_len == 0 else max(src_len + 1, trg_len + 1)
                batch.append(idx)

                if n_tokens > max_tokens:
                    max_tokens = n_tokens
                if max_tokens * len(batch) >= self.batch_size:
                    yield batch
                    batch = []
                    max_tokens = 0

        if len(batch) > 0:
            if not self.drop_last:
                yield batch
            else:
                logger.warning(f"Drop indices {batch}.")

    def __len__(self):
        raise NotImplementedError
