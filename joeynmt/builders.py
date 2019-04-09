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
from joeynmt.transformer import NoamScheduler


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

    assert not ("clip_grad_val" in config.keys() and
                "clip_grad_norm" in config.keys()), \
        "you can only specify either clip_grad_val or clip_grad_norm"

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
        optimizer = torch.optim.Adam(parameters, weight_decay=weight_decay,
                                     lr=learning_rate)
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
        - "noam": see `joeynmt.transformer.NoamScheduler`

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
            scheduler = NoamScheduler(hidden_size, 1, 4000, optimizer)
            scheduler_step_at = "step"
    return scheduler, scheduler_step_at
