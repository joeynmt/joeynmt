# coding: utf-8
"""
Data module
"""
import logging
from functools import partial
from typing import Callable, List, Tuple

import torch
from torch.utils.data import (
    BatchSampler,
    DataLoader,
    Dataset,
    RandomSampler,
    Sampler,
    SequentialSampler,
)

from joeynmt.batch import Batch
from joeynmt.constants import PAD_ID

logger = logging.getLogger(__name__)
CPU_DEVICE = torch.device("cpu")


def collate_fn(
    batch: List[Tuple],
    src_process: Callable,
    trg_process: Callable,
    pad_index: int = PAD_ID,
    device: torch.device = CPU_DEVICE,
    has_trg: bool = True,
    is_train: bool = True,
) -> Batch:
    """
    Custom collate function.
    See https://pytorch.org/docs/stable/data.html#dataloader-collate-fn for details.
    Note: you might need another collate_fn() if you switch to a different batch class.
    Please override the batch class here. (not in TrainManager)

    :param batch:
    :param src_process:
    :param trg_process:
    :param pad_index:
    :param device:
    :param has_trg:
    :param is_train:
    :return: joeynmt batch object
    """

    def _is_valid(s, t):
        # pylint: disable=no-else-return
        if has_trg:
            return s is not None and t is not None
        else:
            return s is not None

    batch = [(s, t) for s, t in batch if _is_valid(s, t)]
    src_list, trg_list = zip(*batch)
    assert len(batch) == len(src_list), (len(batch), len(src_list))
    assert all(s is not None for s in src_list), src_list
    src, src_length = src_process(src_list)

    if has_trg:
        assert all(t is not None for t in trg_list), trg_list
        assert trg_process is not None
        trg, trg_length = trg_process(trg_list)
    else:
        assert all(t is None for t in trg_list)
        trg, trg_length = None, None
    return Batch(
        src=torch.tensor(src).long(),
        src_length=torch.tensor(src_length).long(),
        trg=torch.tensor(trg).long() if trg else None,
        trg_length=torch.tensor(trg_length).long() if trg_length else None,
        device=device,
        pad_index=pad_index,
        has_trg=has_trg,
        is_train=is_train,
    )


def make_data_iter(
    dataset: Dataset,
    batch_size: int,
    batch_type: str = "sentence",
    seed: int = 42,
    shuffle: bool = False,
    num_workers: int = 0,
    pad_index: int = PAD_ID,
    device: torch.device = CPU_DEVICE,
) -> DataLoader:
    """
    Returns a torch DataLoader for a torch Dataset. (no bucketing)

    :param dataset: torch dataset containing src and optionally trg
    :param batch_size: size of the batches the iterator prepares
    :param batch_type: measure batch size by sentence count or by token count
    :param seed: random seed for shuffling
    :param shuffle: whether to shuffle the data before each epoch
        (for testing, no effect even if set to True)
    :param num_workers: number of cpus for multiprocessing
    :param pad_index:
    :param device:
    :return: torch DataLoader
    """
    assert isinstance(dataset, Dataset), dataset
    # sampler
    sampler: Sampler[int]  # (type annotation)
    if shuffle and dataset.split == "train":
        generator = torch.Generator()
        generator.manual_seed(seed)
        sampler = RandomSampler(dataset, generator=generator)
    else:
        sampler = SequentialSampler(dataset)

    # batch generator
    if batch_type == "sentence":
        batch_sampler = SentenceBatchSampler(sampler,
                                             batch_size=batch_size,
                                             drop_last=False)
    elif batch_type == "token":
        batch_sampler = TokenBatchSampler(sampler,
                                          batch_size=batch_size,
                                          drop_last=False)

    assert dataset.sequence_encoder[dataset.src_lang] is not None
    if dataset.has_trg:
        assert dataset.sequence_encoder[dataset.trg_lang] is not None
    else:
        dataset.sequence_encoder[dataset.trg_lang] = None

    # data iterator
    return DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        collate_fn=partial(
            collate_fn,
            src_process=dataset.sequence_encoder[dataset.src_lang],
            trg_process=dataset.sequence_encoder[dataset.trg_lang],
            pad_index=pad_index,
            device=device,
            has_trg=dataset.has_trg,
            is_train=dataset.split == "train",
        ),
        num_workers=num_workers,
    )


class SentenceBatchSampler(BatchSampler):
    """
    Wraps another sampler to yield a mini-batch of indices based on num of instances.
    An instance longer than dataset.max_len will be filtered out.

    :param sampler: Base sampler. Can be any iterable object
    :param batch_size: Size of mini-batch.
    :param drop_last: If `True`, the sampler will drop the last batch if its size
        would be less than `batch_size`
    """

    def __init__(self, sampler: Sampler, batch_size: int, drop_last: bool):
        super().__init__(sampler, batch_size, drop_last)

    def __iter__(self):
        batch = []
        d = self.sampler.data_source
        for idx in self.sampler:
            src, trg = d[idx]  # pylint: disable=unused-variable
            if src is not None:  # otherwise drop instance
                batch.append(idx)
                if len(batch) >= self.batch_size:
                    yield batch
                    batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch


class TokenBatchSampler(BatchSampler):
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

    def __init__(self, sampler: Sampler, batch_size: int, drop_last: bool):
        super().__init__(sampler, batch_size, drop_last)

    def __iter__(self):
        batch = []
        max_tokens = 0
        d = self.sampler.data_source
        for idx in self.sampler:
            src, trg = d[idx]  # call __getitem__()
            if src is not None:  # otherwise drop instance
                src_len = 0 if src is None else len(src)
                trg_len = 0 if trg is None else len(trg)
                n_tokens = 0 if src_len == 0 else max(src_len + 1, trg_len + 2)
                batch.append(idx)
                if n_tokens > max_tokens:
                    max_tokens = n_tokens
                if max_tokens * len(batch) >= self.batch_size:
                    yield batch
                    batch = []
                    max_tokens = 0
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        raise NotImplementedError
