# coding: utf-8
"""
Collection of builder functions
"""
from typing import Callable, Optional, Generator

import torch
from torch import nn
from torch.optim.lr_scheduler import _LRScheduler, ReduceLROnPlateau, \
    StepLR, ExponentialLR
from torch.optim import Optimizer

from joeynmt.helpers import ConfigurationError


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
    clip_grad_fun = None
    if "clip_grad_val" in config.keys():
        clip_value = config["clip_grad_val"]
        clip_grad_fun = lambda params: \
            nn.utils.clip_grad_value_(parameters=params,
                                      clip_value=clip_value)
    elif "clip_grad_norm" in config.keys():
        max_norm = config["clip_grad_norm"]
        clip_grad_fun = lambda params: \
            nn.utils.clip_grad_norm_(parameters=params, max_norm=max_norm)

    if "clip_grad_val" in config.keys() and "clip_grad_norm" in config.keys():
        raise ConfigurationError(
            "You can only specify either clip_grad_val or clip_grad_norm.")

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
    learning_rate = config.get("learning_rate", 3.0e-4)
    weight_decay = config.get("weight_decay", 0)

    if optimizer_name == "adam":
        adam_betas = config.get("adam_betas", (0.9, 0.999))
        optimizer = torch.optim.Adam(parameters, weight_decay=weight_decay,
                                     lr=learning_rate, betas=adam_betas)
    elif optimizer_name == "adagrad":
        optimizer = torch.optim.Adagrad(parameters, weight_decay=weight_decay,
                                        lr=learning_rate)
    elif optimizer_name == "adadelta":
        optimizer = torch.optim.Adadelta(parameters, weight_decay=weight_decay,
                                         lr=learning_rate)
    elif optimizer_name == "rmsprop":
        optimizer = torch.optim.RMSprop(parameters, weight_decay=weight_decay,
                                        lr=learning_rate)
    elif optimizer_name == "sgd":
        # default
        optimizer = torch.optim.SGD(parameters, weight_decay=weight_decay,
                                    lr=learning_rate)
    else:
        raise ConfigurationError("Invalid optimizer. Valid options: 'adam', "
                                 "'adagrad', 'adadelta', 'rmsprop', 'sgd'.")
    return optimizer


def build_scheduler(config: dict, optimizer: Optimizer, scheduler_mode: str,
                    hidden_size: int = 0) \
        -> (Optional[_LRScheduler], Optional[str]):
    """
    Create a learning rate scheduler if specified in config and
    determine when a scheduler step should be executed.

    Current options:
        - "plateau": see `torch.optim.lr_scheduler.ReduceLROnPlateau`
        - "decaying": see `torch.optim.lr_scheduler.StepLR`
        - "exponential": see `torch.optim.lr_scheduler.ExponentialLR`
        - "noam": see `joeynmt.builders.NoamScheduler`
        - "elan": see `joeynmt.builders.ElanScheduler`

    If no scheduler is specified, returns (None, None) which will result in
    a constant learning rate.

    :param config: training configuration
    :param optimizer: optimizer for the scheduler, determines the set of
        parameters which the scheduler sets the learning rate for
    :param scheduler_mode: "min" or "max", depending on whether the validation
        score should be minimized or maximized.
        Only relevant for "plateau".
    :param hidden_size: encoder hidden size (required for NoamScheduler)
    :return:
        - scheduler: scheduler object,
        - scheduler_step_at: either "validation" or "epoch"
    """
    scheduler, scheduler_step_at = None, None
    if "scheduling" in config.keys() and \
            config["scheduling"]:
        if config["scheduling"].lower() == "plateau":
            # learning rate scheduler
            scheduler = ReduceLROnPlateau(
                optimizer=optimizer,
                mode=scheduler_mode,
                verbose=False,
                threshold_mode='abs',
                factor=config.get("decrease_factor", 0.1),
                patience=config.get("patience", 10))
            # scheduler step is executed after every validation
            scheduler_step_at = "validation"
        elif config["scheduling"].lower() == "decaying":
            scheduler = StepLR(
                optimizer=optimizer,
                step_size=config.get("decaying_step_size", 1))
            # scheduler step is executed after every epoch
            scheduler_step_at = "epoch"
        elif config["scheduling"].lower() == "exponential":
            scheduler = ExponentialLR(
                optimizer=optimizer,
                gamma=config.get("decrease_factor", 0.99))
            # scheduler step is executed after every epoch
            scheduler_step_at = "epoch"
        elif config["scheduling"].lower() == "noam":
            factor = config.get("learning_rate_factor", 1)
            warmup = config.get("learning_rate_warmup", 4000)
            scheduler = NoamScheduler(hidden_size=hidden_size, factor=factor,
                                      warmup=warmup, optimizer=optimizer)

            scheduler_step_at = "step"
        elif config["scheduling"].lower() == "elan":
            min_rate = config.get("learning_rate_min", 1.0e-5)
            decay_rate = config.get("learning_rate_decay", 0.1)
            warmup = config.get("learning_rate_warmup", 4000)
            peak_rate = config.get("learning_rate_peak", 1.0e-3)
            decay_length = config.get("learning_rate_decay_length", 10000)
            scheduler = ElanScheduler(min_rate=min_rate, decay_rate=decay_rate,
                                      warmup=warmup, optimizer=optimizer,
                                      peak_rate=peak_rate,
                                      decay_length=decay_length)
            scheduler_step_at = "step"
    return scheduler, scheduler_step_at


class NoamScheduler:
    """
    The Noam learning rate scheduler used in "Attention is all you need"
    See Eq. 3 in https://arxiv.org/pdf/1706.03762.pdf
    """

    def __init__(self, hidden_size: int, optimizer: torch.optim.Optimizer,
                 factor: float = 1, warmup: int = 4000):
        """
        Warm-up, followed by learning rate decay.

        :param hidden_size:
        :param optimizer:
        :param factor: decay factor
        :param warmup: number of warmup steps
        """
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.hidden_size = hidden_size
        self._rate = 0

    def step(self):
        """Update parameters and rate"""
        self._step += 1
        rate = self._compute_rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate

    def _compute_rate(self):
        """Implement `lrate` above"""
        step = self._step
        return self.factor * \
            (self.hidden_size ** (-0.5) *
                min(step ** (-0.5), step * self.warmup ** (-1.5)))

    #pylint: disable=no-self-use
    def state_dict(self):
        return None


class ElanScheduler:
    """
    A learning rate scheduler similar to Noam, but modified as proposed by Elan:
    Keep the warm up period but make it so that the decay rate can be tuneable.
    The decay is exponential up to a given minimum rate.
    """

    def __init__(self, optimizer: torch.optim.Optimizer,
                 peak_rate: float = 1.0e-3,
                 decay_length: int = 10000, warmup: int = 4000,
                 decay_rate: float = 0.5, min_rate: float = 1.0e-5):
        """
        Warm-up, followed by exponential learning rate decay.

        :param peak_rate: maximum learning rate at peak after warmup
        :param optimizer:
        :param decay_length: decay length after warmup
        :param decay_rate: decay rate after warmup
        :param warmup: number of warmup steps
        :param min_rate: minimum learning rate
        """
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.decay_length = decay_length
        self.peak_rate = peak_rate
        self._rate = 0
        self.decay_rate = decay_rate
        self.min_rate = min_rate

    def step(self):
        """Update parameters and rate"""
        self._step += 1
        rate = self._compute_rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate

    def _compute_rate(self):
        """Implement `lrate` above"""
        step = self._step
        warmup = self.warmup

        if step < warmup:
            rate = step * self.peak_rate / warmup
        else:
            exponent = (step - warmup) / self.decay_length
            rate = self.peak_rate * (self.decay_rate ** exponent)
        return max(rate, self.min_rate)

    #pylint: disable=no-self-use
    def state_dict(self):
        return None
