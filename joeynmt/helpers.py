# coding: utf-8
"""
Collection of helper functions
"""
import copy
import glob
import os
import os.path
from logging import Logger
from typing import Callable, Optional
import numpy as np
import yaml


import torch
from torch import nn

from torchtext.data import Dataset

from joeynmt.vocabulary import Vocabulary
from joeynmt.plotting import plot_heatmap


def log_cfg(cfg: dict, logger: Logger, prefix: str = "cfg"):
    """
    Write configuration to log.

    :param cfg: configuration to log
    :param logger: logger that defines where log is written to
    :param prefix: prefix for logging
    :return:
    """
    for k, v in cfg.items():
        if isinstance(v, dict):
            p = '.'.join([prefix, k])
            log_cfg(v, logger, prefix=p)
        else:
            p = '.'.join([prefix, k])
            logger.info("{:34s} : {}".format(p, v))


def clones(module: nn.Module, n: int):
    """
    Produce N identical layers. Transformer helper function.

    :param module: the module to clone
    :param n: clone this many times
    :return:
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


def subsequent_mask(size: int):
    """
    Mask out subsequent positions (to prevent attending to future positions)
    Transformer helper function.

    :param size:
    :return:
    """
    attn_shape = (1, size, size)
    mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(mask) == 0


def log_data_info(train_data: Dataset, valid_data: Dataset, test_data: Dataset,
                  src_vocab: Vocabulary, trg_vocab: Vocabulary,
                  logging_function: Callable[[str], None]):
    """
    Log statistics of data and vocabulary.

    :param train_data:
    :param valid_data:
    :param test_data:
    :param src_vocab:
    :param trg_vocab:
    :param logging_function:
    :return:
    """
    logging_function(
        "Data set sizes: \n\ttrain %d,\n\tvalid %d,\n\ttest %d",
            len(train_data), len(valid_data),
            len(test_data) if test_data is not None else 0)

    logging_function("First training example:\n\t[SRC] %s\n\t[TRG] %s",
        " ".join(vars(train_data[0])['src']),
        " ".join(vars(train_data[0])['trg']))

    logging_function("First 10 words (src): %s", " ".join(
        '(%d) %s' % (i, t) for i, t in enumerate(src_vocab.itos[:10])))
    logging_function("First 10 words (trg): %s", " ".join(
        '(%d) %s' % (i, t) for i, t in enumerate(trg_vocab.itos[:10])))

    logging_function("Number of Src words (types): %d", len(src_vocab))
    logging_function("Number of Trg words (types): %d", len(trg_vocab))


def load_config(path="configs/default.yaml"):
    """
    Loads and parses a YAML configuration file.

    :param path:
    :return:
    """
    with open(path, 'r') as ymlfile:
        cfg = yaml.load(ymlfile)
    return cfg


def bpe_postprocess(string):
    """
    Post-processor for BPE output. Recombines BPE-split tokens.

    :param string:
    :return:
    """
    return string.replace("@@ ", "")


def store_attention_plots(attentions, targets, sources, output_prefix,
                          idx):
    """
    Saves attention plots.

    :param attentions:
    :param targets:
    :param sources:
    :param output_prefix:
    :param idx:
    :return:
    """
    for i in idx:
        plot_file = "{}.{}.pdf".format(output_prefix, i)
        src = sources[i]
        trg = targets[i]
        attention_scores = attentions[i].T
        try:
            plot_heatmap(scores=attention_scores, column_labels=trg,
                         row_labels=src, output_path=plot_file)
        # pylint: disable=bare-except
        except:
            print("Couldn't plot example {}: src len {}, trg len {}, "
                  "attention scores shape {}".format(i, len(src), len(trg),
                                                     attention_scores.shape))
            continue


def get_latest_checkpoint(ckpt_dir: str) -> Optional[str]:
    """
    Returns the latest checkpoint (by time) from the given directory.
    If there is no checkpoint in this directory, returns None

    :param ckpt_dir:
    :return: latest checkpoint file
    """
    list_of_files = glob.glob("{}/*.ckpt".format(ckpt_dir))
    latest_checkpoint = None
    if list_of_files:
        latest_checkpoint = max(list_of_files, key=os.path.getctime)
    return latest_checkpoint


def load_model_from_checkpoint(path, use_cuda=True):
    """
    Load model from saved checkpoint.

    :param path:
    :param use_cuda:
    :return:
    """
    assert os.path.isfile(path), "Checkpoint %s not found" % path
    model_checkpoint = torch.load(path,
                                  map_location='cuda' if use_cuda else 'cpu')
    return model_checkpoint


# from onmt
def tile(x, count, dim=0):
    """
    Tiles x on dimension dim count times. From OpenNMT. Used for beam search.

    :param x:
    :param count:
    :param dim:
    :return:
    """
    if isinstance(x, tuple):
        h, c = x
        return tile(h, count, dim=dim), tile(c, count, dim=dim)

    perm = list(range(len(x.size())))
    if dim != 0:
        perm[0], perm[dim] = perm[dim], perm[0]
        x = x.permute(perm).contiguous()
    out_size = list(x.size())
    out_size[0] *= count
    batch = x.size(0)
    x = x.view(batch, -1) \
        .transpose(0, 1) \
        .repeat(count, 1) \
        .transpose(0, 1) \
        .contiguous() \
        .view(*out_size)
    if dim != 0:
        x = x.permute(perm).contiguous()
    return x


def freeze_params(module):
    """
    Freeze the parameters of this module,
    i.e. do not update them during training

    :param module:
    :return:
    """
    for _, p in module.named_parameters():
        p.requires_grad = False
