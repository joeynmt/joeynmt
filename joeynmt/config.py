# coding: utf-8
"""
Module for configuration

This can only be a temporary solution.
TODO: Consider better configuration and validation
cf. https://github.com/joeynmt/joeynmt/issues/196
"""
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, NamedTuple, Optional

import torch
import yaml
from torch.multiprocessing import cpu_count

from joeynmt.helpers_for_ddp import get_logger, use_ddp

logger = get_logger(__name__)


class ConfigurationError(Exception):
    """Custom exception for misspecifications of configuration"""


TrainConfig = NamedTuple(
    "TrainConfig",
    [
        ("load_model", Optional[Path]),
        ("load_encoder", Optional[Path]),
        ("load_decoder", Optional[Path]),
        ("loss", str),
        ("normalization", str),
        ("label_smoothing", float),
        ("optimizer", str),
        ("adam_betas", List[float]),
        ("learning_rate", float),
        ("learning_rate_min", float),
        ("learning_rate_factor", int),
        ("learning_rate_warmup", int),
        ("scheduling", Optional[str]),
        ("patience", int),
        ("decrease_factor", float),
        ("weight_decay", float),
        ("clip_grad_norm", Optional[float]),
        ("clip_grad_val", Optional[float]),
        ("keep_best_ckpts", int),
        ("logging_freq", int),
        ("validation_freq", int),
        ("print_valid_sents", List[int]),
        ("early_stopping_metric", str),
        ("minimize_metric", bool),
        ("shuffle", bool),
        ("epochs", int),
        ("max_updates", int),
        ("batch_size", int),
        ("batch_type", str),
        ("batch_multiplier", int),
        ("reset_best_ckpt", bool),
        ("reset_scheduler", bool),
        ("reset_optimizer", bool),
        ("reset_iter_state", bool),
    ],
)

TestConfig = NamedTuple(
    "TestConfig",
    [
        ("load_model", Optional[Path]),
        ("batch_size", int),
        ("batch_type", str),
        ("max_output_length", int),
        ("min_output_length", int),
        ("eval_metrics", List[str]),
        ("sacrebleu_cfg", Optional[Dict]),
        ("beam_size", int),
        ("beam_alpha", int),
        ("n_best", int),
        ("return_attention", bool),
        ("return_prob", str),
        ("generate_unk", bool),
        ("repetition_penalty", float),
        ("no_repeat_ngram_size", int),
    ],
)

BaseConfig = NamedTuple(
    "BaseConfig",
    [
        ("name", str),
        ("joeynmt_version", Optional[str]),
        ("model_dir", Path),
        ("device", torch.device),
        ("n_gpu", int),
        ("num_workers", int),
        ("autocast", Dict),
        ("seed", int),
        ("train", TrainConfig),
        ("test", TestConfig),
        ("data", Dict),  # TODO: validate
        ("model", Dict),  # TODO: validate
    ],
)


def _check_path(path: str, allow_empty: bool = True) -> Path:
    """check if given path exists"""
    if path is not None:
        path = Path(path).absolute()
        if not allow_empty:
            assert path.exists(), f"{path} not found."
    return path


def _check_options(name: str, choice: Any, valid_options: List[Any]) -> None:
    """check if given choice is valid"""
    if choice not in valid_options:
        valids = "{" + ", ".join([f"`{option}`" for option in valid_options]) + "}"
        raise ConfigurationError(
            f"Invalid setting for `{name}`. "
            f"Valid choices: {valids}."
        )


def _check_special_symbols(special_symbols: Dict) -> Dict:
    special_symbols["unk_id"] = special_symbols.get("unk_id", 0)
    special_symbols["unk_token"] = special_symbols.get("unk_token", "<unk>")
    special_symbols["pad_id"] = special_symbols.get("pad_id", 1)
    special_symbols["pad_token"] = special_symbols.get("pad_token", "<pad>")
    special_symbols["bos_id"] = special_symbols.get("bos_id", 2)
    special_symbols["bos_token"] = special_symbols.get("bos_token", "<s>")
    special_symbols["eos_id"] = special_symbols.get("eos_id", 3)
    special_symbols["eos_token"] = special_symbols.get("eos_token", "</s>")
    special_symbols["sep_id"] = special_symbols.get("sep_id", None)
    special_symbols["sep_token"] = special_symbols.get("sep_token", None)
    special_symbols["lang_tags"] = special_symbols.get("lang_tags", [])
    return special_symbols


def log_config(cfg: Dict, prefix: str = "cfg") -> None:
    """
    Print configuration to console log.

    :param cfg: configuration to log
    :param prefix: prefix for logging
    """
    for k, v in cfg.items():
        if isinstance(v, dict):
            p = ".".join([prefix, k])
            log_config(v, prefix=p)
        else:
            p = ".".join([prefix, k])
            logger.info("%34s : %s", p, v)


def load_config(cfg_file: str = "configs/default.yaml") -> Dict:
    """
    Loads and parses a YAML configuration file.

    :param cfg_file: path to YAML configuration file
    :return: configuration dictionary
    """
    cfg_file = _check_path(cfg_file, allow_empty=False)
    with cfg_file.open("r", encoding="utf-8") as ymlfile:
        cfg = yaml.safe_load(ymlfile)

    # for backwards compatibility
    if "model_dir" not in cfg:
        cfg["model_dir"] = cfg["training"]["model_dir"]
    return cfg


def parse_global_args(
    cfg: Dict = None, rank: int = 0, mode: str = "train"
) -> BaseConfig:
    """
    Parse and validate global args

    :param cfg: config specified in yaml file
    :param rank:
    :param mode:
    """

    # gpu / cpu
    use_cuda = cfg.get("use_cuda", cfg["training"].get("use_cuda", True))
    if use_cuda and (not torch.cuda.is_available()):
        logger.warning("CUDA is not available. Use cpu device.")
        use_cuda = False
    if use_cuda:
        device = torch.device("cuda", rank) if use_ddp() else torch.device("cuda")
    else:
        device = torch.device("cpu")
    n_gpu = torch.cuda.device_count() if use_cuda else 0

    num_workers = cfg.get("num_workers", cfg["training"].get("num_workers", 0))
    if num_workers > 0:
        num_workers = min(cpu_count(), num_workers)

    if mode == "translate" and n_gpu > 1:
        raise RuntimeError(
            "Currently, translate mode is only available on CPU or single GPU."
        )

    # normalization
    normalization = cfg.get("normalization", "batch").lower()
    _check_options("normalization", normalization, ["batch", "tokens", "none"])

    # fp16
    fp16 = cfg.get("fp16", False)
    if device.type == "cpu" and fp16:
        logger.warning(
            "On cpu, half-precision training may raise an error. Disable fp16."
        )
        fp16 = False
    autocast = {"device_type": device.type, "enabled": fp16}
    if fp16:
        autocast["dtype"] = torch.float16  # TODO: torch.bfloat16 for cpu?

    # special symbols
    _special_symbols = cfg["data"].get("special_symbols", {})
    if isinstance(_special_symbols, dict):
        _special_symbols = _check_special_symbols(_special_symbols)
        cfg["data"]["special_symbols"] = SimpleNamespace(**_special_symbols)
    assert isinstance(cfg["data"]["special_symbols"], SimpleNamespace)

    return BaseConfig(
        name=cfg["name"],
        joeynmt_version=cfg.get("joeynmt_version", "2.3.0"),
        model_dir=_check_path(cfg["model_dir"]),
        device=device,
        n_gpu=n_gpu,
        num_workers=num_workers,
        autocast=autocast,
        seed=cfg.get("random_seed", 42),
        train=parse_train_args(cfg["training"], mode),
        test=parse_test_args(cfg["testing"], mode),
        data=cfg["data"],  # TODO: parse and validate DataConfig
        model=cfg["model"],  # TODO: parse and validate ModelConfig
    )


def parse_train_args(cfg: Dict = None, mode: str = "train") -> TrainConfig:
    """
    Parse and validate train args

    :param cfg: `training` section in config yaml
    :param mode:
    """

    # normalization
    normalization = cfg.get("normalization", "batch").lower()
    _check_options("normalization", normalization, ["batch", "tokens", "none"])

    # objective
    loss_type = cfg.get("loss", "crossentropy")
    _check_options("loss", loss_type, ["crossentropy"])

    # save/delete checkpoints
    keep_best_ckpts = int(cfg.get("keep_best_ckpts", 5))
    _keep_last_ckpts = cfg.get("keep_last_ckpts", None)
    if _keep_last_ckpts is not None:  # backward compatibility
        keep_best_ckpts = _keep_last_ckpts
        logger.warning(
            "`keep_last_ckpts` option is outdated. "
            "Please use `keep_best_ckpts`, instead."
        )

    # early stopping
    early_stopping_metric = cfg.get("early_stopping_metric", "ppl").lower()
    _check_options(
        "early_stopping_metric", early_stopping_metric,
        ["acc", "loss", "ppl", "bleu", "chrf"]
    )

    # early_stopping_metric decides on how to find the early stopping point: ckpts
    # are written when there's a new high/low score for this metric. If we schedule
    # after loss/ppl, we want to minimize the score, else we want to maximize it.
    if early_stopping_metric in ["ppl", "loss"]:  # lower is better
        minimize_metric = True
    elif early_stopping_metric in ["acc", "bleu", "chrf"]:  # higher is better
        minimize_metric = False

    # batch handling
    batch_type = cfg.get("batch_type", "sentence").lower()
    _check_options("batch_type", batch_type, ["sentence", "token"])
    if use_ddp():
        assert batch_type == "sentence", (
            "Token-based batch sampling is not supported in distributed learning. "
            "Please specify batch size based on the num. of sentences."
        )

    # logging
    logging_freq = cfg.get("logging_freq", 100)
    validation_freq = cfg.get("validation_freq", 1000)
    if logging_freq > validation_freq:
        raise ConfigurationError(
            "`logging_freq` must be smaller than `validation_freq`."
        )
    if validation_freq % logging_freq != 0:
        raise ConfigurationError(
            "`validation_freq` must be divisible by `logging_freq`."
        )

    is_test = mode != "train"

    return TrainConfig(
        load_model=_check_path(cfg.get("load_model", None), allow_empty=is_test),
        load_encoder=_check_path(cfg.get("load_encoder", None), allow_empty=is_test),
        load_decoder=_check_path(cfg.get("load_decoder", None), allow_empty=is_test),
        normalization=normalization,
        loss=loss_type,
        label_smoothing=cfg.get("label_smoothing", 0.0),
        optimizer=cfg.get("optimizer", "adam").lower(),
        adam_betas=cfg.get("adam_betas", [0.9, 0.999]),
        learning_rate=cfg.get("learning_rate", 0.005),
        learning_rate_min=cfg.get("learning_rate_min", 0.0001),
        learning_rate_factor=cfg.get("learning_rate_factor", 1),
        learning_rate_warmup=cfg.get("learning_rate_warmup", 4000),
        scheduling=cfg.get("scheduling", None),  # constant
        patience=cfg.get("patience", 5),
        decrease_factor=cfg.get("decrease_factor", 0.5),
        weight_decay=cfg.get("weight_decay", 0.0),
        clip_grad_norm=cfg.get("clip_grad_norm", None),
        clip_grad_val=cfg.get("clip_grad_val", None),
        keep_best_ckpts=keep_best_ckpts,
        logging_freq=logging_freq,
        validation_freq=validation_freq,
        print_valid_sents=cfg.get("print_valid_sents", [0, 1, 2]),
        early_stopping_metric=early_stopping_metric,
        minimize_metric=minimize_metric,
        shuffle=cfg.get("shuffle", True),
        epochs=cfg.get("epochs", 3),
        max_updates=cfg.get("updates", float('inf')),
        batch_size=cfg["batch_size"],
        batch_type=batch_type,
        batch_multiplier=cfg.get("batch_multiplier", 1),
        reset_best_ckpt=cfg.get("reset_best_ckpt", False),
        reset_scheduler=cfg.get("reset_scheduler", False),
        reset_optimizer=cfg.get("reset_optimizer", False),
        reset_iter_state=cfg.get("reset_iter_state", False),
    )


def parse_test_args(cfg: Dict = None, mode: str = "test") -> TestConfig:
    """
    Parse and validate test args

    :param cfg: `testing` section in config yaml
    :param mode:
    """

    # batch options
    batch_size = cfg.get("batch_size", 64)
    batch_type = cfg.get("batch_type", "sentence").lower()
    _check_options("batch_type", batch_type, ["sentence", "token"])
    if batch_size > 1000 and batch_type == "sentence":
        logger.warning(
            "WARNING: Are you sure you meant to work on huge batches like this? "
            "`batch_size` is > 1000 for sentence-batching. Consider decreasing it "
            "or switching to `batch_type: 'token'`."
        )

    # eval metrics
    if "eval_metrics" in cfg:
        eval_metrics = [s.strip().lower() for s in cfg["eval_metrics"]]
    elif "eval_metric" in cfg:
        eval_metrics = [cfg["eval_metric"].strip().lower()]
        logger.warning(
            "`eval_metric` option is obsolete. Please use `eval_metrics`, instead."
        )
    else:
        eval_metrics = []
    for eval_metric in eval_metrics:
        _check_options(
            "eval_metric", eval_metric,
            ["bleu", "chrf", "token_accuracy", "sequence_accuracy"]
        )

    # sacrebleu cfg
    sacrebleu_cfg: Dict = cfg.get("sacrebleu_cfg", {})
    if "sacrebleu" in cfg:
        sacrebleu_cfg: Dict = cfg["sacrebleu"]
        logger.warning(
            "`sacrebleu` option is obsolete. Please use `sacrebleu_cfg`, instead."
        )

    # beam search options
    n_best = cfg.get("n_best", 1)
    if n_best < 1:
        raise ConfigurationError("N-best size must be > 0.")

    beam_size = cfg.get("beam_size", 1)
    if beam_size < 1:
        raise ConfigurationError("Beam size must be > 0.")

    if n_best > beam_size:
        raise ConfigurationError(
            "`n_best` must be smaller than or equal to `beam_size`."
        )

    beam_alpha = cfg.get("beam_alpha", -1)
    if "alpha" in cfg:
        beam_alpha = cfg["alpha"]
        logger.warning("`alpha` option is obsolete. Please use `beam_alpha`, instead.")

    # generation control
    return_prob = cfg.get("return_prob", "none")
    _check_options("return_prob", return_prob, ["hyp", "ref", "none"])

    repetition_penalty: float = cfg.get("repetition_penalty", -1)
    if 0 < repetition_penalty < 1:
        raise ConfigurationError(
            "Repetition penalty must be > 1. (-1 indicates no repetition penalty.)"
        )

    return TestConfig(
        load_model=_check_path(
            cfg.get("load_model", None), allow_empty=mode == "train"
        ),
        batch_size=batch_size,
        batch_type=batch_type,
        max_output_length=cfg.get("max_output_length", -1),
        min_output_length=cfg.get("min_output_length", 1),
        eval_metrics=eval_metrics,
        sacrebleu_cfg=sacrebleu_cfg,
        beam_size=beam_size,
        beam_alpha=beam_alpha,
        n_best=n_best,
        return_attention=cfg.get("return_attention", False),
        return_prob=return_prob,
        generate_unk=cfg.get("generate_unk", True),
        repetition_penalty=repetition_penalty,
        no_repeat_ngram_size=cfg.get("no_repeat_ngram_size", -1),
    )


def set_validation_args(args: TestConfig) -> TestConfig:
    """
    Config for validation

    :param args: `testing` section in config yaml
    """
    if use_ddp():
        assert args.batch_type == "sentence", (
            "Token-based batch sampling is not supported in distributed learning. "
            "Please specify batch size based on the num. of sentences."
        )
    args = args._replace(
        beam_size=1,  # greedy decoding during train loop
        n_best=1,  # no further exploration during training
        return_attention=False,
        return_prob="none",
        generate_unk=True,
        repetition_penalty=-1,  # turn off
        no_repeat_ngram_size=-1,  # turn off
    )
    return args
