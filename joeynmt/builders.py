# coding: utf-8
"""
Collection of builder functions
"""
import logging
from functools import partial
from typing import Callable, Generator, Optional

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    ExponentialLR,
    ReduceLROnPlateau,
    StepLR,
    _LRScheduler,
)

from joeynmt.helpers import ConfigurationError

logger = logging.getLogger(__name__)


def build_gradient_clipper(config: dict) -> Optional[Callable]:
    """
    Define the function for gradient clipping as specified in configuration.
    If not specified, returns None.

    Current options:
        - "clip_grad_val": clip the gradients if they exceed this value,
            see `torch.nn.utils.clip_grad_value_`
        - "clip_grad_norm": clip the gradients if their norm exceeds this value,
            see `torch.nn.utils.clip_grad_norm_`

    :param config: dictionary with training configurations
    :return: clipping function (in-place) or None if no gradient clipping
    """
    if "clip_grad_val" in config.keys() and "clip_grad_norm" in config.keys():
        raise ConfigurationError(
            "You can only specify either clip_grad_val or clip_grad_norm.")

    clip_grad_fun = None
    if "clip_grad_val" in config.keys():
        clip_grad_fun = partial(nn.utils.clip_grad_value_,
                                clip_value=config["clip_grad_val"])
    elif "clip_grad_norm" in config.keys():
        clip_grad_fun = partial(nn.utils.clip_grad_norm_,
                                max_norm=config["clip_grad_norm"])
    return clip_grad_fun


def build_optimizer(config: dict, parameters: Generator) -> Optimizer:
    """
    Create an optimizer for the given parameters as specified in config.

    Except for the weight decay and initial learning rate,
    default optimizer settings are used.

    Currently supported configuration settings for "optimizer":
        - "sgd" (default): see `torch.optim.SGD`
        - "adam": see `torch.optim.adam`
        - "adagrad": see `torch.optim.adagrad`
        - "adadelta": see `torch.optim.adadelta`
        - "rmsprop": see `torch.optim.RMSprop`

    The initial learning rate is set according to "learning_rate" in the config.
    The weight decay is set according to "weight_decay" in the config.
    If they are not specified, the initial learning rate is set to 3.0e-4, the
    weight decay to 0.

    Note that the scheduler state is saved in the checkpoint, so if you load
    a model for further training you have to use the same type of scheduler.

    :param config: configuration dictionary
    :param parameters:
    :return: optimizer
    """
    optimizer_name = config.get("optimizer", "sgd").lower()

    kwargs = {
        "lr": config.get("learning_rate", 3.0e-4),
        "weight_decay": config.get("weight_decay", 0),
    }

    if optimizer_name == "adam":
        kwargs["betas"] = config.get("adam_betas", (0.9, 0.999))
        optimizer = torch.optim.Adam(parameters, **kwargs)
    elif optimizer_name == "adagrad":
        optimizer = torch.optim.Adagrad(parameters, **kwargs)
    elif optimizer_name == "adadelta":
        optimizer = torch.optim.Adadelta(parameters, **kwargs)
    elif optimizer_name == "rmsprop":
        optimizer = torch.optim.RMSprop(parameters, **kwargs)
    elif optimizer_name == "sgd":
        # default
        kwargs["momentum"] = config.get("momentum", 0.0)
        optimizer = torch.optim.SGD(parameters, **kwargs)
    else:
        raise ConfigurationError("Invalid optimizer. Valid options: 'adam', "
                                 "'adagrad', 'adadelta', 'rmsprop', 'sgd'.")

    logger.info(
        "%s(%s)",
        optimizer.__class__.__name__,
        ", ".join([f"{k}={v}" for k, v in kwargs.items()]),
    )
    return optimizer


def build_scheduler(
    config: dict,
    optimizer: Optimizer,
    scheduler_mode: str,
    hidden_size: int = 0,
) -> (Optional[_LRScheduler], Optional[str]):
    """
    Create a learning rate scheduler if specified in config and determine when a
    scheduler step should be executed.

    Current options:
        - "plateau": see `torch.optim.lr_scheduler.ReduceLROnPlateau`
        - "decaying": see `torch.optim.lr_scheduler.StepLR`
        - "exponential": see `torch.optim.lr_scheduler.ExponentialLR`
        - "noam": see `joeynmt.builders.NoamScheduler`
        - "warmupexponentialdecay": see
          `joeynmt.builders.WarmupExponentialDecayScheduler`
        - "warmupinversesquareroot": see
          `joeynmt.builders.WarmupInverseSquareRootScheduler`

    If no scheduler is specified, returns (None, None) which will result in a constant
    learning rate.

    :param config: training configuration
    :param optimizer: optimizer for the scheduler, determines the set of parameters
        which the scheduler sets the learning rate for
    :param scheduler_mode: "min" or "max", depending on whether the validation score
        should be minimized or maximized. Only relevant for "plateau".
    :param hidden_size: encoder hidden size (required for NoamScheduler)
    :return:
        - scheduler: scheduler object,
        - scheduler_step_at: either "validation", "epoch", "step" or "none"
    """
    scheduler, scheduler_step_at = None, None
    if "scheduling" in config.keys() and config["scheduling"]:
        scheduler_name = config["scheduling"].lower()
        kwargs = {}
        if scheduler_name == "plateau":
            # learning rate scheduler
            kwargs = {
                "mode": scheduler_mode,
                "verbose": False,
                "threshold_mode": "abs",
                "eps": 0.0,
                "factor": config.get("decrease_factor", 0.1),
                "patience": config.get("patience", 10),
            }
            scheduler = ReduceLROnPlateau(optimizer=optimizer, **kwargs)
            # scheduler step is executed after every validation
            scheduler_step_at = "validation"
        elif scheduler_name == "decaying":
            kwargs = {"step_size": config.get("decaying_step_size", 1)}
            scheduler = StepLR(optimizer=optimizer, **kwargs)
            # scheduler step is executed after every epoch
            scheduler_step_at = "epoch"
        elif scheduler_name == "exponential":
            kwargs = {"gamma": config.get("decrease_factor", 0.99)}
            scheduler = ExponentialLR(optimizer=optimizer, **kwargs)
            # scheduler step is executed after every epoch
            scheduler_step_at = "epoch"
        elif scheduler_name == "noam":
            scheduler = NoamScheduler(
                optimizer=optimizer,
                hidden_size=hidden_size,
                factor=config.get("learning_rate_factor", 1),
                warmup=config.get("learning_rate_warmup", 4000),
            )
            scheduler_step_at = "step"
        elif scheduler_name == "warmupexponentialdecay":
            scheduler = WarmupExponentialDecayScheduler(
                min_rate=config.get("learning_rate_min", 1.0e-5),
                decay_rate=config.get("learning_rate_decay", 0.1),
                warmup=config.get("learning_rate_warmup", 4000),
                peak_rate=config.get("learning_rate_peak", 1.0e-3),
                decay_length=config.get("learning_rate_decay_length", 10000),
            )
            scheduler_step_at = "step"
        elif scheduler_name == "warmupinversesquareroot":
            lr = config.get("learning_rate", 1.0e-3)
            peak_rate = config.get("learning_rate_peak", lr)
            scheduler = WarmupInverseSquareRootScheduler(
                optimizer=optimizer,
                peak_rate=peak_rate,
                min_rate=config.get("learning_rate_min", 1.0e-5),
                warmup=config.get("learning_rate_warmup", 10000),
            )
            scheduler_step_at = "step"

    if scheduler is None:
        scheduler_step_at = "none"
    else:
        assert scheduler_step_at in {"validation", "epoch", "step", "none"}

    # print log
    if scheduler_name in [
            "noam",
            "warmupexponentialdecay",
            "warmupinversesquareroot",
    ]:
        logger.info(scheduler)
    else:
        logger.info(
            "%s(%s)",
            scheduler.__class__.__name__,
            ", ".join([f"{k}={v}" for k, v in kwargs.items()]),
        )
    return scheduler, scheduler_step_at


class BaseScheduler:
    """Base LR Scheduler
    decay at "step"
    """

    def __init__(self, optimizer: torch.optim.Optimizer):
        """
        :param optimizer:
        """
        self.optimizer = optimizer
        self._step = 0
        self._rate = 0
        self._state_dict = {"step": self._step, "rate": self._rate}

    def state_dict(self):
        """Returns dictionary of values necessary to reconstruct scheduler"""
        self._state_dict["step"] = self._step
        self._state_dict["rate"] = self._rate
        return self._state_dict

    def load_state_dict(self, state_dict):
        """Given a state_dict, this function loads scheduler's state"""
        self._step = state_dict["step"]
        self._rate = state_dict["rate"]

    def step(self, step):
        """Update parameters and rate"""
        self._step = step + 1  # sync with trainer.stats.steps
        rate = self._compute_rate()
        for p in self.optimizer.param_groups:
            p["lr"] = rate
        self._rate = rate

    def _compute_rate(self):
        raise NotImplementedError


class NoamScheduler(BaseScheduler):
    """
    The Noam learning rate scheduler used in "Attention is all you need"
    See Eq. 3 in https://arxiv.org/abs/1706.03762
    """

    def __init__(
        self,
        hidden_size: int,
        optimizer: torch.optim.Optimizer,
        factor: float = 1.0,
        warmup: int = 4000,
    ):
        """
        Warm-up, followed by learning rate decay.

        :param hidden_size:
        :param optimizer:
        :param factor: decay factor
        :param warmup: number of warmup steps
        """
        super().__init__(optimizer)
        self.warmup = warmup
        self.factor = factor
        self.hidden_size = hidden_size

    def _compute_rate(self):
        """Implement `lrate` above"""
        step = self._step
        upper_bound = min(step**(-0.5), step * self.warmup**(-1.5))
        return self.factor * (self.hidden_size**(-0.5) * upper_bound)

    def state_dict(self):
        """Returns dictionary of values necessary to reconstruct scheduler"""
        super().state_dict()
        self._state_dict["warmup"] = self.warmup
        self._state_dict["factor"] = self.factor
        self._state_dict["hidden_size"] = self.hidden_size
        return self._state_dict

    def load_state_dict(self, state_dict):
        """Given a state_dict, this function loads scheduler's state"""
        super().load_state_dict(state_dict)
        self.warmup = state_dict["warmup"]
        self.factor = state_dict["factor"]
        self.hidden_size = state_dict["hidden_size"]

    def __repr__(self):
        return (f"{self.__class__.__name__}(warmup={self.warmup}, "
                f"factor={self.factor}, hidden_size={self.hidden_size})")


class WarmupExponentialDecayScheduler(BaseScheduler):
    """
    A learning rate scheduler similar to Noam, but modified:
    Keep the warm up period but make it so that the decay rate can be tuneable.
    The decay is exponential up to a given minimum rate.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        peak_rate: float = 1.0e-3,
        decay_length: int = 10000,
        warmup: int = 4000,
        decay_rate: float = 0.5,
        min_rate: float = 1.0e-5,
    ):
        """
        Warm-up, followed by exponential learning rate decay.

        :param peak_rate: maximum learning rate at peak after warmup
        :param optimizer:
        :param decay_length: decay length after warmup
        :param decay_rate: decay rate after warmup
        :param warmup: number of warmup steps
        :param min_rate: minimum learning rate
        """
        super().__init__(optimizer)
        self.warmup = warmup
        self.decay_length = decay_length
        self.peak_rate = peak_rate
        self.decay_rate = decay_rate
        self.min_rate = min_rate

    def _compute_rate(self):
        """Implement `lrate` above"""
        step = self._step
        warmup = self.warmup

        if step < warmup:
            rate = step * self.peak_rate / warmup
        else:
            exponent = (step - warmup) / self.decay_length
            rate = self.peak_rate * (self.decay_rate**exponent)
        return max(rate, self.min_rate)

    def state_dict(self):
        """Returns dictionary of values necessary to reconstruct scheduler"""
        super().state_dict()
        self._state_dict["warmup"] = self.warmup
        self._state_dict["decay_length"] = self.decay_length
        self._state_dict["peak_rate"] = self.peak_rate
        self._state_dict["decay_rate"] = self.decay_rate
        self._state_dict["min_rate"] = self.min_rate
        return self._state_dict

    def load_state_dict(self, state_dict):
        """Given a state_dict, this function loads scheduler's state"""
        super().load_state_dict(state_dict)
        self.warmup = state_dict["warmup"]
        self.decay_length = state_dict["decay_length"]
        self.peak_rate = state_dict["peak_rate"]
        self.decay_rate = state_dict["decay_rate"]
        self.min_rate = state_dict["min_rate"]

    def __repr__(self):
        return (f"{self.__class__.__name__}(warmup={self.warmup}, "
                f"decay_length={self.decay_length}, "
                f"decay_rate={self.decay_rate}, "
                f"peak_rate={self.peak_rate}, "
                f"min_rate={self.min_rate})")


class WarmupInverseSquareRootScheduler(BaseScheduler):
    """
    Decay the LR based on the inverse square root of the update number.
    In the warmup phase, we linearly increase the learning rate.
    After warmup, we decrease the learning rate as follows:
    ```
    decay_factor = peak_rate * sqrt(warmup) # constant value
    lr = decay_factor / sqrt(step)
    ```
    cf.) https://github.com/pytorch/fairseq/blob/main/fairseq/optim/lr_scheduler/inverse_square_root_schedule.py
    """  # noqa

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        peak_rate: float = 1.0e-3,
        warmup: int = 10000,
        min_rate: float = 1.0e-5,
    ):
        """
        Warm-up, followed by inverse square root learning rate decay.
        :param optimizer:
        :param peak_rate: maximum learning rate at peak after warmup
        :param warmup: number of warmup steps
        :param min_rate: minimum learning rate
        """
        super().__init__(optimizer)
        self.warmup = warmup
        self.min_rate = min_rate
        self.peak_rate = peak_rate
        self.decay_rate = peak_rate * (warmup**0.5)  # constant value

    def _compute_rate(self):
        """Implement `lrate` above"""
        step = self._step
        warmup = self.warmup

        if step < warmup:
            # linear warmup
            rate = step * self.peak_rate / warmup
        else:
            # decay prop. to the inverse square root of the update number
            rate = self.decay_rate * (step**-0.5)
        return max(rate, self.min_rate)

    def state_dict(self):
        """Returns dictionary of values necessary to reconstruct scheduler"""
        super().state_dict()
        self._state_dict["warmup"] = self.warmup
        self._state_dict["peak_rate"] = self.peak_rate
        self._state_dict["decay_rate"] = self.decay_rate
        self._state_dict["min_rate"] = self.min_rate
        return self._state_dict

    def load_state_dict(self, state_dict):
        """Given a state_dict, this function loads scheduler's state"""
        super().load_state_dict(state_dict)
        self.warmup = state_dict["warmup"]
        self.decay_rate = state_dict["decay_rate"]
        self.peak_rate = state_dict["peak_rate"]
        self.min_rate = state_dict["min_rate"]

    def __repr__(self):
        return (f"{self.__class__.__name__}(warmup={self.warmup}, "
                f"decay_rate={self.decay_rate:.6f}, peak_rate={self.peak_rate}, "
                f"min_rate={self.min_rate})")
