import logging
from functools import partial
from pathlib import Path
from typing import Dict, List, Union

from torch import nn

from joeynmt.datasets import build_dataset, BaseDataset, StreamDataset
from joeynmt.helpers import (
    load_checkpoint,
    load_config,
    parse_train_args,
    resolve_ckpt_path,
)
from joeynmt.model import build_model, Model
from joeynmt.prediction import predict
from joeynmt.tokenizers import build_tokenizer
from joeynmt.vocabulary import build_vocab

logger = logging.getLogger(__name__)


def _check_file_path(path: Union[str, Path], model_dir: Path) -> Path:
    if path is None:
        return None
    p = Path(path) if isinstance(path, str) else path
    if not p.is_file():
        p = model_dir / p.name
    assert p.is_file(), p
    return p


def _from_pretrained(
    model_name_or_path: Union[str, Path],
    ckpt_file: Union[str, Path] = None,
    cfg_file: Union[str, Path] = "config.yaml",
    **kwargs,
):
    # model dir
    model_dir = Path(model_name_or_path) if isinstance(model_name_or_path, str) else model_name_or_path
    assert model_dir.is_dir(), model_dir

    # cfg file
    cfg_file = _check_file_path(cfg_file, model_dir)
    assert cfg_file.is_file(), cfg_file
    cfg = load_config(cfg_file)
    cfg.update(kwargs)

    # rewrite paths in cfg
    for side in ["src", "trg"]:
        cfg["data"][side]["voc_file"] = _check_file_path(cfg["data"][side]["voc_file"],
                                                         model_dir).as_posix()
        if ("tokenizer_cfg" in cfg["data"][side]
            and "codes" in cfg["data"][side]["tokenizer_cfg"]):
            cfg["data"][side]["tokenizer_cfg"]["codes"] = _check_file_path(
                cfg["data"][side]["tokenizer_cfg"]["codes"], model_dir).as_posix()
    if "load_model" in cfg["training"]:
        cfg["training"]["load_model"] = _check_file_path(cfg["training"]["load_model"],
                                                         model_dir).as_posix()
    if not Path(cfg["training"]["model_dir"]).is_dir():
        cfg["training"]["model_dir"] = model_dir.as_posix()

    # parse and validate cfg
    _, load_model_path, device, n_gpu, _, _, fp16 = parse_train_args(cfg["training"],
                                                                     mode="prediction")

    # read vocabs
    src_vocab, trg_vocab = build_vocab(cfg["data"], model_dir=model_dir)

    # build model
    model = build_model(cfg["model"], src_vocab=src_vocab, trg_vocab=trg_vocab)

    # load model state from disk
    ckpt_file = _check_path(ckpt_file, model_dir)
    load_model_path = load_model_path if ckpt_file is None else ckpt_file
    ckpt = resolve_ckpt_path(load_model_path, model_dir)
    model_checkpoint = load_checkpoint(ckpt, device=device)
    model.load_state_dict(model_checkpoint["model_state"])

    if device.type == "cuda":
        model.to(device)

    # create stream dataset
    src_lang = cfg["data"]["src"]["lang"]
    trg_lang = cfg["data"]["trg"]["lang"]
    tokenizer = build_tokenizer(cfg["data"])
    sequence_encoder = {
        src_lang: partial(src_vocab.sentences_to_ids, bos=False, eos=True),
        trg_lang: None,
    }
    test_data = build_dataset(
        dataset_type="stream",
        path=None,
        src_lang=src_lang,
        trg_lang=trg_lang,
        split="test",
        tokenizer=tokenizer,
        sequence_encoder=sequence_encoder,
    )

    config = {
        "device": device,
        "n_gpu": n_gpu,
        "fp16": fp16,
        "test_cfg": cfg["testing"],
    }
    return config, test_data, model


class TranslatorHubInterface(nn.Module):
    """
    PyTorch Hub interface for generating sequences from a pre-trained
    encoder-decoder model.
    """

    def __init__(self, config: Dict, dataset: BaseDataset, model: Model):
        super().__init__()
        self.test_cfg = config["test_cfg"]
        self.device = config["device"]
        self.n_gpu = config["n_gpu"]
        self.fp16 = config["fp16"]
        self.dataset = dataset
        self.model = model

    def translate(self, sentences: List[str],
                  **kwargs) -> Union[List[str], List[List[str]]]:
        # overwrite config
        self.test_cfg.update(kwargs)

        if isinstance(self.dataset, StreamDataset):
            self.test_cfg["batch_type"] = "sentence"
            self.test_cfg["batch_size"] = len(sentences)
            self.dataset.cache = {}  # reset cache
            for sentence in sentences:
                self.dataset.set_item(sentence)

        assert len(self.dataset) > 0

        _, _, translations, _, _, _ = predict(
            model=self.model,
            data=self.dataset,
            compute_loss=False,
            device=self.device,
            n_gpu=self.n_gpu,
            normalization="none",
            num_workers=0,
            cfg=self.test_cfg,
            fp16=self.fp16,
        )
        assert len(sentences) * self.test_cfg.get("n_best", 1) == len(translations)

        if isinstance(self.dataset, StreamDataset):
            self.dataset.cache = {}  # reset cache

        return translations
