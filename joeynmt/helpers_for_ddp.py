# coding: utf-8
"""
helper functions for DDP
"""
import logging
import math
import os
from typing import Optional, Union

import torch
import torch.distributed as dist
from torch import Tensor
from torch.utils.data import Dataset, SequentialSampler
from torch.utils.data.distributed import DistributedSampler


def ddp_setup(
    rank: int,
    world_size: int,
    master_addr: str = "localhost",
    master_port: int = 12355,
) -> None:
    """
    Setup distributed environment

    :param rank: Unique identifier of each process
    :param world_size: Total number of processes
    :param master_addr:
    :param master_port:
    """
    if dist.is_available():
        if "MASTER_ADDR" not in os.environ:
            os.environ["MASTER_ADDR"] = master_addr
        if "MASTER_PORT" not in os.environ:
            os.environ["MASTER_PORT"] = str(master_port)

        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)


def use_ddp() -> bool:
    """Check if DDP environment is available"""
    return dist.is_available() and dist.is_initialized()


def ddp_cleanup() -> None:
    """Cleanup distributed environment"""
    if use_ddp():
        dist.destroy_process_group()


def ddp_synchronize() -> None:
    """Synchronize distributed processes"""
    if use_ddp():
        dist.barrier()


def ddp_merge(data: Tensor, pad_index: int = 1) -> Tensor:
    """
    Merge tensors from multiple devices

    :param data: tensor to merge
    :param pad_index:
    :return: merged tensor
    """
    if data is None:
        return None
    assert torch.is_tensor(data), data

    if use_ddp():
        dim = len(data.size())
        if dim == 2:
            batch_size, seq_length = data.size()
        elif dim == 3:
            batch_size, seq_length, vocab_size = data.size()
        else:
            raise ValueError

        world_size = dist.get_world_size()

        # check tensor size in each device
        local_size = torch.tensor(data.size(), device=data.device)
        all_sizes = [torch.zeros_like(local_size) for _ in range(world_size)]
        dist.all_gather(all_sizes, local_size)

        # resolve length differences
        # `dist.all_gather()` requires the same tensor dim across devices
        #
        # ex) world_size = 2;
        #   rank 0: data = [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]  in shape (2, 5)
        #   rank 1: data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]    in shape (3, 3)
        # -> max length along 0th dim = 3, max length along 1st dim = 5
        #
        # after padding (pad_index = -1): all tensors in shape (3, 5)
        #   rank 0: padding = [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [-1, -1, -1, -1, -1]]
        #   rank 1: padding = [[1, 2, 3, -1, -1], [4, 5, 6, -1, -1], [7, 8, 9, -1, -1]]
        #
        # merged tensor in shape (3 * world_size, 5):
        #   tmp = [
        #      [1, 2, 3, 4, 5],      # rank 0
        #      [6, 7, 8, 9, 10],     # rank 0
        #      [-1, -1, -1, -1, -1], # rank 0
        #      [1, 2, 3, -1, -1],    # rank 1
        #      [4, 5, 6, -1, -1],    # rank 1
        #      [7, 8, 9, -1, -1],    # rank 1
        #   ]
        #
        # will be returned a tensor in shape (5, 5):
        #   data = [
        #      [1, 2, 3, 4, 5],
        #      [6, 7, 8, 9, 10],
        #      # [-1, -1, -1, -1, -1], # the padding-only row should be omitted!
        #      [1, 2, 3, -1, -1],
        #      [4, 5, 6, -1, -1],
        #      [7, 8, 9, -1, -1],
        #   ]

        # check max lengths
        max_length0 = max(size[0] for size in all_sizes).item()
        max_length1 = max(size[1] for size in all_sizes).item()
        max_length2 = max(size[2] for size in all_sizes).item() if dim == 3 else 0

        # padding
        if dim == 2:
            pad_size = (max_length0, max_length1)
        elif dim == 3:
            pad_size = (max_length0, max_length1, max_length2)

        padding = torch.full(pad_size, pad_index, device=data.device, dtype=data.dtype)

        if dim == 2:
            padding[:batch_size, :seq_length] = data
        elif dim == 3:
            padding[:batch_size, :seq_length, :vocab_size] = data

        # gather
        tmp = [torch.zeros_like(padding) for _ in range(world_size)]
        dist.all_gather(tmp, padding)
        data = torch.cat(tmp, dim=0)

        # omit padding-only rows
        valid = []
        valid_batch_size = 0
        for i, size in enumerate(all_sizes):
            valid_batch_size += size[0].item()
            offset = i * max_length0
            end = offset + size[0].item()
            valid.append(data[offset:end])
            if end < offset + max_length0:
                cutoff = data[end:offset + max_length0]
                assert torch.all(cutoff == pad_index).item()
        data = torch.cat(valid, dim=0)
        assert data.size(0) == valid_batch_size, (data.size(), valid_batch_size)
    return data


def ddp_reduce(data: Union[Tensor, int], device=None, dtype=None) -> Tensor:
    """
    Reduce tensors from multiple devices

    :param data: tensor to reduce
    :return: reduced tensor
    """
    if data is None:
        return None
    if not torch.is_tensor(data):
        assert device is not None and dtype is not None
        data = torch.tensor(data, device=device, dtype=dtype)

    if use_ddp():
        if len(data.size()) < 1:
            data = data.unsqueeze(0)
        dist.all_reduce(data, op=dist.ReduceOp.SUM)
    return data


class MultiProcessAdapter(logging.LoggerAdapter):
    """
    An adapter to assist with logging in multiprocess.

    taken from Huggingface's Accelerate logger
    """

    def log(self, level, msg, *args, **kwargs):
        """
        Delegates logger call after checking if we should log.
        """
        flag = False
        master_only = kwargs.pop("master_only", True)

        if master_only:
            rank = dist.get_rank() if use_ddp() else 0
            flag = rank == 0

        if self.isEnabledFor(level) and flag:
            msg, kwargs = self.process(msg, kwargs)
            self.logger.log(level, msg, *args, **kwargs)


def get_logger(name: str = "", log_file: str = None) -> logging.Logger:
    """
    Create a logger for logging the training/testing process.

    :param name: logger name.
    :param log_file: path to file where log is stored as well
    :return: logging.Logger
    """
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    )

    def _add_filehandler(logger, log_file):
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(level=logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    def _add_streamhandler(logger):
        sh = logging.StreamHandler()
        sh.setLevel(logging.INFO)
        sh.setFormatter(formatter)
        logger.addHandler(sh)

    # assign file handler whenever `log_file` arg is provided
    if log_file is not None:
        for logger_name in logging.root.manager.loggerDict:
            if logger_name.startswith("joeynmt."):
                logger = logging.getLogger(logger_name)
                if len(logger.handlers) < 2:
                    _add_filehandler(logger, log_file)

    current_logger = logging.getLogger(name)
    if len(current_logger.handlers) == 0:
        current_logger.setLevel(level=logging.DEBUG)
        _add_streamhandler(current_logger)
        if log_file is not None:
            _add_filehandler(current_logger, log_file)

    current_logger.propagate = False  # otherwise root logger prints things again

    return MultiProcessAdapter(current_logger, {})


class DistributedSubsetSampler(DistributedSampler):
    """
    DistributedSampler with random subsampling.
    `drop_last` logic is simplified; raise error if `len(dataset)` is not divisible
    by `world_size` and cut off leftovers.

    .. warning::
        Token-based batch sampling is not supported in distributed learning.

    :param data_source (Dataset): dataset to sample from
    :param num_replicas (int): ddp world size
    :param rank (int): ddp local rank
    :param shuffle (bool): whether to permute or not
    :param drop_last (bool): must be true!
    :param generator (Generator): Generator used in sampling.
    """

    def __init__(
        self,
        dataset: Dataset,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        drop_last: bool = True,
        generator: torch.Generator = None
    ):
        # pylint: disable=super-init-not-called
        # super().__init__(dataset, num_replicas, rank, shuffle, seed, drop_last)

        if num_replicas is None:
            if not use_ddp():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not use_ddp():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                f"Invalid rank {rank}, rank should be in the interval"
                f" [0, {num_replicas - 1}]"
            )
        self.data_source = dataset  # alias
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.generator = generator

    @property
    def num_samples(self) -> int:
        """total size"""
        return len(self.data_source.indices)

    def __iter__(self):
        indices = self.data_source.indices

        if self.shuffle:  # permute
            perm = torch.randperm(len(indices), generator=self.generator).tolist()
            indices = [indices[i] for i in perm]
            # don't assign permuted indices to self.data_source.indices

        if len(indices) % self.num_replicas != 0 and not self.drop_last:
            raise RuntimeError("`len(dataset)` must be divisible by `world_size`.")
            # set `random_subset` with a divisible value or enable drop_last

        # remove tail of data to make it evenly divisible.
        total_samples = (self.num_samples // self.num_replicas) * self.num_replicas
        indices = indices[:total_samples]
        assert len(indices) % self.num_replicas == 0, (
            len(indices), self.num_samples, self.num_replicas
        )
        self.data_source.indices = indices  # reset indices after dropping leftovers

        # distribute samples
        indices_per_replica = indices[self.rank:self.num_samples:self.num_replicas]
        assert len(indices_per_replica) == math.ceil(
            self.num_samples / self.num_replicas
        )

        return iter(indices_per_replica)

    def _subsample(self):
        """get random subset; indices are still sorted (no permutation!)"""
        orig_len = len(self.data_source)
        subset_len = self.data_source.random_subset
        if 0 < subset_len < orig_len:
            subset = torch.randperm(n=orig_len,
                                    generator=self.generator).tolist()[:subset_len]
            self.data_source.indices = sorted(subset)
            assert len(subset) == self.num_samples

    def reset(self):
        self.data_source.reset_indices()

    def set_seed(self, seed: int) -> None:
        """set seed and resample"""
        self.generator.manual_seed(seed)
        self._subsample()


class RandomSubsetSampler(SequentialSampler):
    """Samples subset randomly from a given data_source without replacement.
       If shuffle = False, yields subset elements sequentially.

    :param data_source (Dataset): dataset to sample from
    :param shuffle (bool): whether to permute or not
    :param generator (Generator): Generator used in sampling.
    """

    def __init__(self, data_source: Dataset, shuffle: bool, generator: torch.Generator):
        super().__init__(data_source)
        self.generator = generator
        self.shuffle = shuffle

    @property
    def num_samples(self) -> int:
        return len(self.data_source.indices)

    def __iter__(self):
        indices = self.data_source.indices

        if self.shuffle:  # permute
            perm = torch.randperm(n=len(indices), generator=self.generator).tolist()
            return iter([indices[i] for i in perm])

        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples

    def _subsample(self):
        """get random subset; indices are still sorted (no permutation!)"""
        orig_len = len(self.data_source)
        subset_len = self.data_source.random_subset
        if 0 < subset_len < orig_len:
            subset = torch.randperm(n=orig_len,
                                    generator=self.generator).tolist()[:subset_len]
            self.data_source.indices = sorted(subset)
            assert len(subset) == self.num_samples

    def reset(self):
        self.data_source.reset_indices()

    def set_seed(self, seed: int) -> None:
        """set seed and resample"""
        self.generator.manual_seed(seed)
        self._subsample()
