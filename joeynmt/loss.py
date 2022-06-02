# coding: utf-8
"""
Module to implement training loss
"""
import torch
from torch import Tensor, nn
from torch.autograd import Variable
from torch.nn.modules.loss import _Loss


class XentLoss(nn.Module):
    """
    Cross-Entropy Loss with optional label smoothing
    """

    def __init__(self, pad_index: int, smoothing: float = 0.0):
        super().__init__()
        self.smoothing = smoothing
        self.pad_index = pad_index
        self.criterion: _Loss  # (type annotation)
        if self.smoothing <= 0.0:
            # standard xent loss
            self.criterion = nn.NLLLoss(ignore_index=self.pad_index, reduction="sum")
        else:
            # custom label-smoothed loss, computed with KL divergence loss
            self.criterion = nn.KLDivLoss(reduction="sum")

    def _smooth_targets(self, targets: Tensor, vocab_size: int) -> Variable:
        """
        Smooth target distribution. All non-reference words get uniform
        probability mass according to "smoothing".

        :param targets: target indices, batch*seq_len
        :param vocab_size: size of the output vocabulary
        :return: smoothed target distributions, batch*seq_len x vocab_size
        """
        # batch*seq_len x vocab_size
        smooth_dist = targets.new_zeros((targets.size(0), vocab_size)).float()
        # fill distribution uniformly with smoothing
        smooth_dist.fill_(self.smoothing / (vocab_size - 2))
        # assign true label the probability of 1-smoothing ("confidence")
        smooth_dist.scatter_(1, targets.unsqueeze(1).data, 1.0 - self.smoothing)
        # give padding probability of 0 everywhere
        smooth_dist[:, self.pad_index] = 0
        # masking out padding area (sum of probabilities for padding area = 0)
        padding_positions = torch.nonzero(targets.data == self.pad_index,
                                          as_tuple=False)
        if len(padding_positions) > 0:
            smooth_dist.index_fill_(0, padding_positions.squeeze(), 0.0)
        return Variable(smooth_dist, requires_grad=False)

    def _reshape(self, log_probs: Tensor, targets: Tensor) -> Tensor:
        vocab_size = log_probs.size(-1)

        # reshape log_probs to (batch*seq_len x vocab_size)
        log_probs_flat = log_probs.contiguous().view(-1, vocab_size)

        if self.smoothing > 0:
            targets_flat = self._smooth_targets(targets=targets.contiguous().view(-1),
                                                vocab_size=vocab_size)
            # targets: distributions with batch*seq_len x vocab_size
            assert log_probs_flat.size() == targets_flat.size(), (
                log_probs.size(),
                targets_flat.size(),
            )
        else:
            # targets: indices with batch*seq_len
            targets_flat = targets.contiguous().view(-1)
            assert log_probs_flat.size(0) == targets_flat.size(0), (
                log_probs.size(0),
                targets_flat.size(0),
            )

        return log_probs_flat, targets_flat

    def forward(self, log_probs: Tensor, **kwargs) -> Tensor:
        """
        Compute the cross-entropy between logits and targets.

        If label smoothing is used, target distributions are not one-hot, but
        "1-smoothing" for the correct target token and the rest of the
        probability mass is uniformly spread across the other tokens.

        :param log_probs: log probabilities as predicted by model
        :return: logits
        """
        assert "trg" in kwargs
        log_probs, targets = self._reshape(log_probs, kwargs["trg"])

        # compute loss
        logits = self.criterion(log_probs, targets)
        return logits

    def __repr__(self):
        return (f"{self.__class__.__name__}(criterion={self.criterion}, "
                f"smoothing={self.smoothing})")
