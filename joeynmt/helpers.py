# coding: utf-8
"""
Collection of helper functions
"""
from __future__ import annotations

import copy
import functools
import logging
import operator
import random
import re
import shutil
import sys
import unicodedata
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pkg_resources
import torch
import yaml
from torch import Tensor, nn
from torch.multiprocessing import cpu_count
from torch.utils.tensorboard import SummaryWriter

from joeynmt.plotting import plot_heatmap

if TYPE_CHECKING:
    from joeynmt.dataset import BaseDataset
    from joeynmt.vocabulary import Vocabulary  # to avoid circular import

np.set_printoptions(linewidth=sys.maxsize)  # format for printing numpy array


class ConfigurationError(Exception):
    """Custom exception for misspecifications of configuration"""


def make_model_dir(model_dir: Path, overwrite: bool = False) -> Path:
    """
    Create a new directory for the model.

    :param model_dir: path to model directory
    :param overwrite: whether to overwrite an existing directory
    :return: path to model directory
    """
    model_dir = model_dir.absolute()
    if model_dir.is_dir():
        if not overwrite:
            raise FileExistsError(f"Model directory {model_dir} exists "
                                  f"and overwriting is disabled.")
        # delete previous directory to start with empty dir again
        shutil.rmtree(model_dir)
    model_dir.mkdir()
    return model_dir


def make_logger(log_dir: Path = None, mode: str = "train") -> str:
    """
    Create a logger for logging the training/testing process.

    :param log_dir: path to file where log is stored as well
    :param mode: log file name. 'train', 'test' or 'translate'
    :return: joeynmt version number
    """
    logger = logging.getLogger("")  # root logger
    version = pkg_resources.require("joeynmt")[0].version

    # add handlers only once.
    if len(logger.handlers) == 0:
        logger.setLevel(level=logging.DEBUG)
        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(name)s - %(message)s")

        if log_dir is not None:
            if log_dir.is_dir():
                log_file = log_dir / f"{mode}.log"

                fh = logging.FileHandler(log_file.as_posix())
                fh.setLevel(level=logging.DEBUG)
                logger.addHandler(fh)
                fh.setFormatter(formatter)

        sh = logging.StreamHandler()
        sh.setLevel(logging.INFO)
        sh.setFormatter(formatter)

        logger.addHandler(sh)
        logger.info("Hello! This is Joey-NMT (version %s).", version)

    return version


def log_cfg(cfg: Dict, prefix: str = "cfg") -> None:
    """
    Write configuration to log.

    :param cfg: configuration to log
    :param prefix: prefix for logging
    """
    logger = logging.getLogger(__name__)
    for k, v in cfg.items():
        if isinstance(v, dict):
            p = ".".join([prefix, k])
            log_cfg(v, prefix=p)
        else:
            p = ".".join([prefix, k])
            logger.info("%34s : %s", p, v)


def clones(module: nn.Module, n: int) -> nn.ModuleList:
    """
    Produce N identical layers. Transformer helper function.

    :param module: the module to clone
    :param n: clone this many times
    :return: cloned modules
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


def subsequent_mask(size: int) -> Tensor:
    """
    Mask out subsequent positions (to prevent attending to future positions)
    Transformer helper function.

    :param size: size of mask (2nd and 3rd dim)
    :return: Tensor with 0s and 1s of shape (1, size, size)
    """
    ones = torch.ones(size, size, dtype=torch.bool)
    return torch.tril(ones, out=ones).unsqueeze(0)


def set_seed(seed: int) -> None:
    """
    Set the random seed for modules torch, numpy and random.

    :param seed: random seed
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed_all(seed)


def log_data_info(
    src_vocab: Vocabulary,
    trg_vocab: Vocabulary,
    train_data: Optional[BaseDataset],
    valid_data: Optional[BaseDataset],
    test_data: Optional[BaseDataset],
) -> None:
    """
    Log statistics of data and vocabulary.

    :param src_vocab:
    :param trg_vocab:
    :param train_data:
    :param valid_data:
    :param test_data:
    """
    logger = logging.getLogger(__name__)
    logger.info("Train dataset: %s", train_data)
    logger.info("Valid dataset: %s", valid_data)
    logger.info(" Test dataset: %s", test_data)

    if train_data:
        src = "\n\t[SRC] " + " ".join(
            train_data.get_item(idx=0, lang=train_data.src_lang, is_train=False))
        trg = "\n\t[TRG] " + " ".join(
            train_data.get_item(idx=0, lang=train_data.trg_lang, is_train=False))
        logger.info("First training example:%s%s", src, trg)

    logger.info("First 10 Src tokens: %s", src_vocab.log_vocab(10))
    logger.info("First 10 Trg tokens: %s", trg_vocab.log_vocab(10))

    logger.info("Number of unique Src tokens (vocab_size): %d", len(src_vocab))
    logger.info("Number of unique Trg tokens (vocab_size): %d", len(trg_vocab))


def load_config(path: Union[Path, str] = "configs/default.yaml") -> Dict:
    """
    Loads and parses a YAML configuration file.

    :param path: path to YAML configuration file
    :return: configuration dictionary
    """
    if isinstance(path, str):
        path = Path(path)
    with path.open("r", encoding="utf-8") as ymlfile:
        cfg = yaml.safe_load(ymlfile)
    return cfg


def write_list_to_file(output_path: Path, array: List[Any]) -> None:
    """
    Write list of str to file in `output_path`.

    :param output_path: output file path
    :param array: list of strings
    """
    with output_path.open("w", encoding="utf-8") as opened_file:
        for entry in array:
            opened_file.write(f"{entry}\n")


def read_list_from_file(input_path: Path) -> List[str]:
    """
    Read list of str from file in `input_path`.

    :param input_path: input file path
    :return: list of strings
    """
    if input_path is None:
        return []
    return [
        line.rstrip("\n")
        for line in input_path.read_text(encoding="utf-8").splitlines()
    ]


def parse_train_args(cfg: Dict, mode: str = "training") -> Tuple:
    """Parse and validate train args specified in config file"""
    logger = logging.getLogger(__name__)

    model_dir: Path = Path(cfg["model_dir"])
    assert model_dir.is_dir(), f"{model_dir} not found."

    use_cuda: bool = cfg["use_cuda"] and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    n_gpu: int = torch.cuda.device_count() if use_cuda else 0
    num_workers: int = cfg.get("num_workers", 0)
    if num_workers > 0:
        num_workers = min(cpu_count(), num_workers)

    # normalization
    normalization: str = cfg.get("normalization", "batch")
    if normalization not in ["batch", "tokens", "none"]:
        raise ConfigurationError(
            "Invalid `normalization` option. Valid options: {`batch`, `token`, `none`}."
        )

    # model initialization
    def _load_path(path):
        load_path = cfg.get(path, None)
        if load_path is not None:
            load_path = Path(load_path)
            assert load_path.is_file()
        return load_path

    load_model: Optional[Path] = _load_path("load_model")

    if mode == "prediction":
        return model_dir, load_model, device, n_gpu, num_workers, normalization

    # layer initialization
    load_encoder: Optional[Path] = _load_path("load_encoder")
    load_decoder: Optional[Path] = _load_path("load_decoder")

    # objective
    loss_type: str = cfg.get("loss", "crossentropy")
    label_smoothing: float = cfg.get("label_smoothing", 0.0)
    if loss_type not in ["crossentropy"]:
        raise ConfigurationError("Invalid `loss` type. Valid option: {`crossentropy`}.")

    # minimum learning rate for early stopping
    learning_rate_min: float = cfg.get("learning_rate_min", 1.0e-8)

    # save/delete checkpoints
    keep_best_ckpts: int = int(cfg.get("keep_best_ckpts", 5))
    _keep_last_ckpts: Optional[int] = cfg.get("keep_last_ckpts", None)
    if _keep_last_ckpts is not None:  # backward compatibility
        keep_best_ckpts = _keep_last_ckpts
        logger.warning("`keep_last_ckpts` option is outdated. "
                       "Please use `keep_best_ckpts`, instead.")

    # logging, validation
    logging_freq: int = cfg.get("logging_freq", 100)
    validation_freq: int = cfg.get("validation_freq", 1000)
    log_valid_sents: List[int] = cfg.get("print_valid_sents", [0, 1, 2])

    # early stopping
    early_stopping_metric: str = cfg.get("early_stopping_metric", "ppl").lower()
    if early_stopping_metric not in ["acc", "loss", "ppl", "bleu", "chrf"]:
        raise ConfigurationError(
            "Invalid setting for `early_stopping_metric`. "
            "Valid options: {`acc`, `loss`, `ppl`, `bleu`, `chrf`}.")

    # data & batch handling
    seed: int = cfg.get("random_seed", 42)
    shuffle: bool = cfg.get("shuffle", True)
    epochs: int = cfg["epochs"]
    max_updates: float = cfg.get("updates", np.inf)
    batch_size: int = cfg["batch_size"]
    batch_type: str = cfg.get("batch_type", "sentence")
    if batch_type not in ["sentence", "token"]:
        raise ConfigurationError(
            "Invalid `batch_type` option. Valid options: {`sentence`, `token`}.")
    batch_multiplier: int = cfg.get("batch_multiplier", 1)

    # fp16
    fp16: bool = cfg.get("fp16", False)

    # resume training process
    reset_best_ckpt = cfg.get("reset_best_ckpt", False)
    reset_scheduler = cfg.get("reset_scheduler", False)
    reset_optimizer = cfg.get("reset_optimizer", False)
    reset_iter_state = cfg.get("reset_iter_state", False)

    return (
        model_dir,
        load_model,
        load_encoder,
        load_decoder,
        loss_type,
        label_smoothing,
        normalization,
        learning_rate_min,
        keep_best_ckpts,
        logging_freq,
        validation_freq,
        log_valid_sents,
        early_stopping_metric,
        seed,
        shuffle,
        epochs,
        max_updates,
        batch_size,
        batch_type,
        batch_multiplier,
        device,
        n_gpu,
        num_workers,
        fp16,
        reset_best_ckpt,
        reset_scheduler,
        reset_optimizer,
        reset_iter_state,
    )


def parse_test_args(cfg: Dict) -> Tuple:
    """Parse test args"""
    logger = logging.getLogger(__name__)

    # batch options
    batch_size: int = cfg.get("batch_size", 64)
    batch_type: str = cfg.get("batch_type", "sentences")
    if batch_type not in ["sentence", "token"]:
        raise ConfigurationError(
            "Invalid `batch_type` option. Valid options: {`sentence`, `token`}.")
    if batch_size > 1000 and batch_type == "sentence":
        logger.warning(
            "WARNING: Are you sure you meant to work on huge batches like this? "
            "`batch_size` is > 1000 for sentence-batching. Consider decreasing it "
            "or switching to `batch_type: 'token'`.")

    # limit on generation length
    max_output_length: int = cfg.get("max_output_length", -1)
    min_output_length: int = cfg.get("min_output_length", 1)

    # eval metrics
    if "eval_metrics" in cfg:
        eval_metrics = [s.strip().lower() for s in cfg["eval_metrics"]]
    elif "eval_metric" in cfg:
        eval_metrics = [cfg["eval_metric"].strip().lower()]
        logger.warning(
            "`eval_metric` option is obsolete. Please use `eval_metrics`, instead.")
    else:
        eval_metrics = []
    for eval_metric in eval_metrics:
        if eval_metric not in ["bleu", "chrf", "token_accuracy", "sequence_accuracy"]:
            raise ConfigurationError(
                "Invalid setting for `eval_metrics`. "
                "Valid options: 'bleu', 'chrf', 'token_accuracy', 'sequence_accuracy'.")

    # sacrebleu cfg
    sacrebleu_cfg: Dict = cfg.get("sacrebleu_cfg", {})
    if "sacrebleu" in cfg:
        sacrebleu_cfg: Dict = cfg["sacrebleu"]
        logger.warning(
            "`sacrebleu` option is obsolete. Please use `sacrebleu_cfg`, instead.")

    # beam search options
    n_best: int = cfg.get("n_best", 1)
    beam_size: int = cfg.get("beam_size", 1)
    beam_alpha: float = cfg.get("beam_alpha", -1)
    if "alpha" in cfg:
        beam_alpha = cfg["alpha"]
        logger.warning("`alpha` option is obsolete. Please use `beam_alpha`, instead.")
    assert beam_size > 0, "Beam size must be > 0."
    assert n_best > 0, "N-best size must be > 0."
    assert n_best <= beam_size, "`n_best` must be smaller than or equal to `beam_size`."

    # control options
    return_attention: bool = cfg.get("return_attention", False)
    return_prob: str = cfg.get("return_prob", "none")
    if return_prob not in ["hyp", "ref", "none"]:
        raise ConfigurationError(
            "Invalid `return_prob` option. Valid options: {`hyp`, `ref`, `none`}.")
    generate_unk: bool = cfg.get("generate_unk", True)
    repetition_penalty: float = cfg.get("repetition_penalty", -1)
    if 0 < repetition_penalty < 1:
        raise ConfigurationError(
            "Repetition penalty must be > 1. (-1 indicates no repetition penalty.)")
    no_repeat_ngram_size: int = cfg.get("no_repeat_ngram_size", -1)

    return (
        batch_size,
        batch_type,
        max_output_length,
        min_output_length,
        eval_metrics,
        sacrebleu_cfg,
        beam_size,
        beam_alpha,
        n_best,
        return_attention,
        return_prob,
        generate_unk,
        repetition_penalty,
        no_repeat_ngram_size,
    )


def store_attention_plots(
    attentions: np.ndarray,
    targets: List[List[str]],
    sources: List[List[str]],
    output_prefix: str,
    indices: List[int],
    tb_writer: Optional[SummaryWriter] = None,
    steps: int = 0,
) -> None:
    """
    Saves attention plots.

    :param attentions: attention scores
    :param targets: list of tokenized targets
    :param sources: list of tokenized sources
    :param output_prefix: prefix for attention plots
    :param indices: indices selected for plotting
    :param tb_writer: Tensorboard summary writer (optional)
    :param steps: current training steps, needed for tb_writer
    :param dpi: resolution for images
    """
    for i in indices:
        if i >= len(sources):
            continue
        plot_file = f"{output_prefix}.{i}.pdf"
        src = sources[i]
        trg = targets[i]
        attention_scores = attentions[i].T
        try:
            fig = plot_heatmap(
                scores=attention_scores,
                column_labels=trg,
                row_labels=src,
                output_path=plot_file,
                dpi=100,
            )
            if tb_writer is not None:
                # lower resolution for tensorboard
                fig = plot_heatmap(
                    scores=attention_scores,
                    column_labels=trg,
                    row_labels=src,
                    output_path=None,
                    dpi=50,
                )
                tb_writer.add_figure(f"attention/{i}.", fig, global_step=steps)
        except Exception:  # pylint: disable=broad-except
            print(f"Couldn't plot example {i}: "
                  f"src len {len(src)}, trg len {len(trg)}, "
                  f"attention scores shape {attention_scores.shape}")
            continue


def get_latest_checkpoint(ckpt_dir: Path) -> Optional[Path]:
    """
    Returns the latest checkpoint (by creation time, not the steps number!)
    from the given directory.
    If there is no checkpoint in this directory, returns None

    :param ckpt_dir:
    :return: latest checkpoint file
    """
    list_of_files = ckpt_dir.glob("*.ckpt")
    latest_checkpoint = None
    if list_of_files:
        latest_checkpoint = max(list_of_files, key=lambda f: f.stat().st_ctime)

    # check existence
    if latest_checkpoint is None:
        raise FileNotFoundError(f"No checkpoint found in directory {ckpt_dir}.")
    return latest_checkpoint


def load_checkpoint(path: Path, device: torch.device) -> Dict:
    """
    Load model from saved checkpoint.

    :param path: path to checkpoint
    :param device: cuda device name or cpu
    :return: checkpoint (dict)
    """
    logger = logging.getLogger(__name__)
    assert path.is_file(), f"Checkpoint {path} not found."
    checkpoint = torch.load(path.as_posix(), map_location=device)
    logger.info("Load model from %s.", path.resolve())
    return checkpoint


def resolve_ckpt_path(ckpt: str, load_model: str, model_dir: Path) -> Path:
    """
    Resolve checkpoint path

    :param ckpt: str passed from stdin args (--ckpt)
    :param load_model: config entry (cfg['training']['load_model'])
    :param model_dir: Path(cfg['training']['model_dir'])
    :return: resolved checkpoint path
    """
    if ckpt is None:
        if load_model is None:
            if (model_dir / "best.ckpt").is_file():
                ckpt = model_dir / "best.ckpt"
            else:
                ckpt = get_latest_checkpoint(model_dir)
        else:
            ckpt = Path(load_model)
    return Path(ckpt)


# from onmt
def tile(x: Tensor, count: int, dim=0) -> Tensor:
    """
    Tiles x on dimension dim count times. From OpenNMT. Used for beam search.

    :param x: tensor to tile
    :param count: number of tiles
    :param dim: dimension along which the tensor is tiled
    :return: tiled tensor
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
    x = (x.view(batch,
                -1).transpose(0, 1).repeat(count,
                                           1).transpose(0,
                                                        1).contiguous().view(*out_size))
    if dim != 0:
        x = x.permute(perm).contiguous()
    return x


def freeze_params(module: nn.Module) -> None:
    """
    Freeze the parameters of this module,
    i.e. do not update them during training

    :param module: freeze parameters of this module
    """
    for _, p in module.named_parameters():
        p.requires_grad = False


def delete_ckpt(to_delete: Path) -> None:
    """
    Delete checkpoint

    :param to_delete: checkpoint file to be deleted
    """
    logger = logging.getLogger(__name__)
    try:
        logger.info("delete %s", to_delete.as_posix())
        to_delete.unlink()

    except FileNotFoundError as e:
        logger.warning(
            "Wanted to delete old checkpoint %s but "
            "file does not exist. (%s)",
            to_delete,
            e,
        )


def symlink_update(target: Path, link_name: Path) -> Optional[Path]:
    """
    This function finds the file that the symlink currently points to, sets it
    to the new target, and returns the previous target if it exists.

    :param target: A path to a file that we want the symlink to point to.
                    no parent dir, filename only, i.e. "10000.ckpt"
    :param link_name: This is the name of the symlink that we want to update.
                    link name with parent dir, i.e. "models/my_model/best.ckpt"

    :return:
        - current_last: This is the previous target of the symlink, before it is
            updated in this function. If the symlink did not exist before or did
            not have a target, None is returned instead.
    """
    if link_name.is_symlink():
        current_last = link_name.resolve()
        link_name.unlink()
        link_name.symlink_to(target)
        return current_last
    link_name.symlink_to(target)
    return None


def flatten(array: List[List[Any]]) -> List[Any]:
    """
    Flatten a nested 2D list. faster even with a very long array than
    [item for subarray in array for item in subarray] or newarray.extend().

    :param array: a nested list
    :return: flattened list
    """
    return functools.reduce(operator.iconcat, array, [])


def expand_reverse_index(reverse_index: List[int], n_best: int = 1) -> List[int]:
    """
    Expand resort_reverse_index for n_best prediction

    ex. 1) reverse_index = [1, 0, 2] and n_best = 2, then this will return
    [2, 3, 0, 1, 4, 5].

    ex. 2) reverse_index = [1, 0, 2] and n_best = 3, then this will return
    [3, 4, 5, 0, 1, 2, 6, 7, 8]

    :param reverse_index: reverse_index returned from batch.sort_by_src_length()
    :param n_best:
    :return: expanded sort_reverse_index
    """
    if n_best == 1:
        return reverse_index

    resort_reverse_index = []
    for ix in reverse_index:
        for n in range(0, n_best):
            resort_reverse_index.append(ix * n_best + n)
    assert len(resort_reverse_index) == len(reverse_index) * n_best
    return resort_reverse_index


def remove_extra_spaces(s: str) -> str:
    """
    Remove extra spaces
    - used in pre_process() / post_process() in tokenizer.py

    :param s: input string
    :return: string w/o extra white spaces
    """
    s = re.sub("\u200b", "", s)
    s = re.sub("[ 　]+", " ", s)

    s = s.replace(" ?", "?")
    s = s.replace(" !", "!")
    s = s.replace(" ,", ",")
    s = s.replace(" .", ".")
    s = s.replace(" :", ":")
    return s.strip()


def unicode_normalize(s: str) -> str:
    """
    apply unicodedata NFKC normalization
    - used in pre_process() in tokenizer.py

    :param s: input string
    :return: normalized string
    """
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("’", "'")
    s = s.replace("“", '"')
    s = s.replace("”", '"')
    return s
