# coding: utf-8
"""
Implementation of a mini-batch.
"""
import logging
from typing import List, Optional

import numpy as np
import torch
from torch import Tensor

from joeynmt.constants import PAD_ID

logger = logging.getLogger(__name__)


class Batch:
    """
    Object for holding a batch of data with mask during training.
    Input is yielded from `collate_fn()` called by torch.data.utils.DataLoader.
    """

    # pylint: disable=too-many-instance-attributes

    def __init__(
        self,
        src: Tensor,
        src_length: Tensor,
        trg: Optional[Tensor],
        trg_length: Optional[Tensor],
        device: torch.device,
        pad_index: int = PAD_ID,
        has_trg: bool = True,
        is_train: bool = True,
    ):
        """
        Creates a new joey batch. This batch supports attributes with src and trg
        length, masks, number of non-padded tokens in trg. Furthermore, it can be
        sorted by src length.

        :param src:
        :param src_length:
        :param trg:
        :param trg_length:
        :param device:
        :param pad_index: *must be the same for both src and trg
        :param is_train: *can be used for online data augmentation, subsampling etc.
        """
        self.src: Tensor = src
        self.src_length: Tensor = src_length
        self.src_mask: Tensor = (self.src != pad_index).unsqueeze(1)
        self.trg_input: Optional[Tensor] = None
        self.trg: Optional[Tensor] = None
        self.trg_mask: Optional[Tensor] = None
        self.trg_length: Optional[Tensor] = None

        self.nseqs: int = self.src.size(0)
        self.ntokens: Optional[int] = None
        self.has_trg: bool = has_trg
        self.is_train: bool = is_train

        if self.has_trg:
            assert trg is not None and trg_length is not None
            # trg_input is used for teacher forcing, last one is cut off
            self.trg_input: Tensor = trg[:, :-1]  # shape (batch_size, seq_length)
            self.trg_length: Tensor = trg_length - 1
            # trg is used for loss computation, shifted by one since BOS
            self.trg: Tensor = trg[:, 1:]  # shape (batch_size, seq_length)
            # we exclude the padded areas (and blank areas) from the loss computation
            self.trg_mask: Tensor = (self.trg != pad_index).unsqueeze(1)
            self.ntokens: int = (self.trg != pad_index).data.sum().item()

        if device.type == "cuda":
            self._make_cuda(device)

        # a batch has to contain more than one src sentence
        assert self.nseqs > 0, self.nseqs

    def _make_cuda(self, device: torch.device) -> None:
        """Move the batch to GPU"""
        self.src = self.src.to(device)
        self.src_length = self.src_length.to(device)
        self.src_mask = self.src_mask.to(device)

        if self.has_trg:
            self.trg_input = self.trg_input.to(device)
            self.trg = self.trg.to(device)
            self.trg_mask = self.trg_mask.to(device)

    def normalize(
        self,
        tensor: Tensor,
        normalization: str = "none",
        n_gpu: int = 1,
        n_accumulation: int = 1,
    ) -> Tensor:
        """
        Normalizes batch tensor (i.e. loss). Takes sum over multiple gpus, divides by
        nseqs or ntokens, divide by n_gpu, then divide by n_accumulation.

        :param tensor: (Tensor) tensor to normalize, i.e. batch loss
        :param normalization: (str) one of {`batch`, `tokens`, `none`}
        :param n_gpu: (int) the number of gpus
        :param n_accumulation: (int) the number of gradient accumulation
        :return: normalized tensor
        """
        if n_gpu > 1:
            tensor = tensor.sum()

        if normalization == "sum":  # pylint: disable=no-else-return
            return tensor
        elif normalization == "batch":
            normalizer = self.nseqs
        elif normalization == "tokens":
            normalizer = self.ntokens
        elif normalization == "none":
            normalizer = 1

        norm_tensor = tensor / normalizer

        if n_gpu > 1:
            norm_tensor = norm_tensor / n_gpu

        if n_accumulation > 1:
            norm_tensor = norm_tensor / n_accumulation
        return norm_tensor

    def sort_by_src_length(self) -> List[int]:
        """
        Sort by src length (descending) and return index to revert sort

        :return: list of indices
        """
        _, perm_index = self.src_length.sort(0, descending=True)
        rev_index = [0] * perm_index.size(0)
        for new_pos, old_pos in enumerate(perm_index.cpu().numpy()):
            rev_index[old_pos] = new_pos

        sorted_src_length = self.src_length[perm_index]
        sorted_src = self.src[perm_index]
        sorted_src_mask = self.src_mask[perm_index]
        if self.has_trg:
            sorted_trg_input = self.trg_input[perm_index]
            sorted_trg_length = self.trg_length[perm_index]
            sorted_trg_mask = self.trg_mask[perm_index]
            sorted_trg = self.trg[perm_index]

        self.src = sorted_src
        self.src_length = sorted_src_length
        self.src_mask = sorted_src_mask

        if self.has_trg:
            self.trg_input = sorted_trg_input
            self.trg_mask = sorted_trg_mask
            self.trg_length = sorted_trg_length
            self.trg = sorted_trg

        assert max(rev_index) < len(rev_index), rev_index
        return rev_index

    def score(self, log_probs: Tensor) -> np.ndarray:
        """Look up the score of the trg token (ground truth) in the batch"""
        scores = []
        for i in range(self.nseqs):
            scores.append(
                np.array([
                    log_probs[i, j, ind].item() for j, ind in enumerate(self.trg[i])
                    if ind != PAD_ID
                ]))
        # Note: each element in `scores` list can have different lengths.
        return np.array(scores, dtype=object)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(nseqs={self.nseqs}, ntokens={self.ntokens}, "
            f"has_trg={self.has_trg}, is_train={self.is_train})")
