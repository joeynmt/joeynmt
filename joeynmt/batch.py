# coding: utf-8
"""
Implementation of a mini-batch.
"""
from typing import List, Optional

import numpy as np
import torch
from torch import Tensor

from joeynmt.helpers import adjust_mask_size
from joeynmt.helpers_for_ddp import get_logger

logger = get_logger(__name__)


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
        src_prompt_mask: Optional[Tensor],
        trg: Optional[Tensor],
        trg_prompt_mask: Optional[Tensor],
        indices: Tensor,
        device: torch.device,
        pad_index: int,
        eos_index: int,
        is_train: bool = True,
    ):
        """
        Creates a new joey batch. This batch supports attributes with src and trg
        length, masks, number of non-padded tokens in trg. Furthermore, it can be
        sorted by src length.

        :param src: shape (batch_size, max_src_len)
        :param src_length: shape (batch_size,)
        :param src_prompt_mask: shape (batch_size, max_src_len)
        :param trg: shape (batch_size, max_trg_len)
        :param trg_prompt_mask: shape (batch_size, max_trg_len)
        :param device:
        :param pad_index: *must be the same for both src and trg
        :param eos_index:
        :param is_train: *can be used for online data augmentation, subsampling etc.
        """
        self.src: Tensor = src
        self.src_length: Tensor = src_length
        self.src_mask: Tensor = (self.src != pad_index).unsqueeze(1)
        self.src_prompt_mask: Optional[Tensor] = None  # equivalent to `token_type_ids`
        self.trg_input: Optional[Tensor] = None
        self.trg: Optional[Tensor] = None
        self.trg_mask: Optional[Tensor] = None
        self.trg_prompt_mask: Optional[Tensor] = None
        self.indices: Tensor = indices

        self.nseqs: int = src.size(0)
        self.ntokens: Optional[Tensor] = None
        self.has_trg: bool = trg is not None
        self.is_train: bool = is_train

        if src_prompt_mask is not None:
            self.src_prompt_mask = src_prompt_mask

        if self.has_trg:
            # trg_input is used for teacher forcing, last one (EOS) is cut off
            has_eos = torch.any(trg == eos_index).item()  # true in training
            trg_input = torch.where(trg == eos_index, pad_index, trg)
            self.trg_input: Tensor = trg_input[:, :-1] if has_eos else trg_input
            # trg is used for loss computation, shifted by one since BOS
            self.trg: Tensor = trg[:, 1:]  # trg: shape (batch_size, trg_len)
            # we exclude the padded areas (and blank areas) from the loss computation
            # `trg_mask` shape (batch_size, 1, trg_len); passed to attention layers
            self.trg_mask: Tensor = (self.trg != pad_index).unsqueeze(1)
            self.ntokens: int = self.trg_mask.sum().item()

            if trg_prompt_mask is not None:
                self.trg_prompt_mask = adjust_mask_size(
                    trg_prompt_mask, self.nseqs, self.trg_input.size(1)
                )

        if device.type == "cuda":
            self._make_cuda(device)

        # a batch has to contain more than one src sentence
        assert self.nseqs > 0, self.nseqs

    def _make_cuda(self, device: torch.device) -> None:
        """Move the batch to GPU"""
        self.src = self.src.to(device)
        self.src_length = self.src_length.to(device)
        self.src_mask = self.src_mask.to(device)
        self.indices = self.indices.to(device)

        if self.src_prompt_mask is not None:
            self.src_prompt_mask = self.src_prompt_mask.to(device)

        if self.has_trg:
            self.trg_input = self.trg_input.to(device)
            self.trg = self.trg.to(device)
            self.trg_mask = self.trg_mask.to(device)

            if self.trg_prompt_mask is not None:
                self.trg_prompt_mask = self.trg_prompt_mask.to(device)

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
        if tensor is None:
            return None
        assert torch.is_tensor(tensor), tensor

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

        self.src = self.src[perm_index]
        self.src_length = self.src_length[perm_index]
        self.src_mask = self.src_mask[perm_index]
        self.indices = self.indices[perm_index]

        if self.src_prompt_mask is not None:
            self.src_prompt_mask = self.src_prompt_mask[perm_index]

        if self.has_trg:
            self.trg_input = self.trg_input[perm_index]
            self.trg_mask = self.trg_mask[perm_index]
            self.trg = self.trg[perm_index]

            if self.trg_prompt_mask is not None:
                self.trg_prompt_mask = self.trg_prompt_mask[perm_index]

        assert max(rev_index) < len(rev_index), rev_index
        return rev_index

    @staticmethod
    def score(log_probs: Tensor, trg: Tensor, pad_index: int) -> np.ndarray:
        """Look up the score of the trg token (ground truth) in the batch"""
        assert log_probs.size(0) == trg.size(0)
        scores = []
        for i in range(log_probs.size(0)):
            scores.append(
                np.array([
                    log_probs[i, j, ind].item() for j, ind in enumerate(trg[i])
                    if ind != pad_index
                ])
            )
        # Note: each element in `scores` list can have different lengths.
        return np.array(scores, dtype=object)

    def __repr__(self) -> str:
        nseqs = self.nseqs.item() if torch.is_tensor(self.nseqs) else self.nseqs
        ntokens = self.ntokens.item() if torch.is_tensor(self.ntokens) else self.ntokens
        return (
            f"{self.__class__.__name__}(nseqs={nseqs}, ntokens={ntokens}, "
            f"has_trg={self.has_trg}, is_train={self.is_train})"
        )
