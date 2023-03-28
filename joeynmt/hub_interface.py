import logging
from functools import partial
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Union

import numpy as np
import plotly.express as px
from torch import nn

from joeynmt.constants import EOS_TOKEN
from joeynmt.datasets import BaseDataset, build_dataset
from joeynmt.helpers import (
    load_checkpoint,
    load_config,
    parse_train_args,
    resolve_ckpt_path,
)
from joeynmt.model import Model, build_model
from joeynmt.prediction import predict
from joeynmt.tokenizers import build_tokenizer
from joeynmt.vocabulary import build_vocab

logger = logging.getLogger(__name__)

PredictionOutput = NamedTuple(
    "PredictionOutput",
    [
        ("translation", List[str]),
        ("tokens", Optional[List[List[str]]]),
        ("token_probs", Optional[List[List[float]]]),
        ("sequence_probs", Optional[List[float]]),
        ("attention_probs", Optional[List[List[float]]]),
    ],
)


def _check_file_path(path: Union[str, Path], model_dir: Path) -> Path:
    """Check torch hub cache path"""
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
    """Prepare model and data placeholder"""
    # model dir
    model_dir = Path(model_name_or_path) if isinstance(model_name_or_path,
                                                       str) else model_name_or_path
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
        if "tokenizer_cfg" in cfg["data"][side]:
            for tok_model in ["codes", "model_file"]:
                if tok_model in cfg["data"][side]["tokenizer_cfg"]:
                    cfg["data"][side]["tokenizer_cfg"][tok_model] = _check_file_path(
                        cfg["data"][side]["tokenizer_cfg"][tok_model],
                        model_dir).as_posix()

    if "load_model" in cfg["training"]:
        cfg["training"]["load_model"] = _check_file_path(cfg["training"]["load_model"],
                                                         model_dir).as_posix()
    if not Path(cfg["training"]["model_dir"]).is_dir():
        cfg["training"]["model_dir"] = model_dir.as_posix()

    # parse and validate cfg
    (_, load_model_path, device, n_gpu, num_workers, normalization,
     fp16) = parse_train_args(cfg["training"], mode="prediction")

    # read vocabs
    src_vocab, trg_vocab = build_vocab(cfg["data"], model_dir=model_dir)

    # build model
    model = build_model(cfg["model"], src_vocab=src_vocab, trg_vocab=trg_vocab)

    # load model state from disk
    logger.info("Preparing a joeynmt model...")
    ckpt_file = _check_file_path(ckpt_file, model_dir)
    load_model_path = load_model_path if ckpt_file is None else ckpt_file
    ckpt = resolve_ckpt_path(load_model_path, model_dir)
    model_checkpoint = load_checkpoint(ckpt, device=device)
    model.load_state_dict(model_checkpoint["model_state"])

    # create stream dataset
    src_lang = cfg["data"]["src"]["lang"]
    trg_lang = cfg["data"]["trg"]["lang"]
    tokenizer = build_tokenizer(cfg["data"])
    sequence_encoder = {
        src_lang: partial(src_vocab.sentences_to_ids, bos=False, eos=True),
        trg_lang: partial(trg_vocab.sentences_to_ids, bos=True, eos=True),
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
        "cfg": cfg,
        "num_workers": num_workers,
        "normalization": normalization,
    }
    return config, test_data, model


class TranslatorHubInterface(nn.Module):
    """
    PyTorch Hub interface for generating sequences from a pre-trained
    encoder-decoder model.
    """

    def __init__(self, config: Dict, dataset: BaseDataset, model: Model):
        super().__init__()
        self.cfg = config["cfg"]
        self.device = config["device"]
        self.n_gpu = config["n_gpu"]
        self.fp16 = config["fp16"]
        self.num_workers = config["num_workers"]
        self.normalization = config["normalization"]
        self.dataset = dataset
        self.model = model
        if self.device.type == "cuda":
            self.model.to(self.device)
        self.model.eval()

    def score(
        self,
        src: List[str],
        trg: Optional[List[str]] = None,
        **kwargs,
    ) -> List[PredictionOutput]:
        assert isinstance(src, list), "Please provide a list of sentences!"
        kwargs["return_prob"] = "hyp" if trg is None else "ref"
        kwargs["return_attention"] = True

        if trg is not None and self.model.loss_function is None:
            self.model.loss_function = (
                # need to instantiate loss func to compute ref scores
                self.cfg["training"].get("loss", "crossentropy"),
                self.cfg["training"].get("label_smoothing", 0.1),
            )

        translations, tokens, probs, attn, test_cfg = self._generate(src, trg, **kwargs)

        beam_size = test_cfg.get("beam_size", 1)
        n_best = test_cfg.get("n_best", 1)

        out = []
        for i in range(len(src)):
            offset = i * n_best
            p_range = probs[offset:offset + n_best]
            pred = PredictionOutput(
                translation=trg[i] if trg else translations[offset:offset + n_best],
                tokens=tokens[offset:offset + n_best],
                token_probs=[p.tolist() for p in p_range] if beam_size == 1 else None,
                sequence_probs=[p[0] for p in p_range] if beam_size > 1 else None,
                attention_probs=attn[offset:offset + n_best] if attn else None,
            )
            out.append(pred)
        return out

    def translate(self, src: List[str], **kwargs) -> List[str]:
        assert isinstance(src, list), "Please provide a list of sentences!"
        kwargs["return_prob"] = "none"

        translations, _, _, _, _ = self._generate(src, **kwargs)

        return translations

    def _generate(
        self,
        src: List[str],
        trg: Optional[List[str]] = None,
        **kwargs,
    ) -> List[str]:

        # overwrite config
        test_cfg = self.cfg['testing'].copy()
        test_cfg.update(kwargs)

        assert self.dataset.__class__.__name__ == "StreamDataset", self.dataset
        test_cfg["batch_type"] = "sentence"
        test_cfg["batch_size"] = len(src)

        self.dataset.cache = {}  # reset cache
        if trg is not None:
            assert len(src) == len(trg), "src and trg must have the same length!"
            self.dataset.has_trg = True
            test_cfg["n_best"] = 1
            test_cfg["beam_size"] = 1
            test_cfg["return_prob"] = "ref"
            for src_sent, trg_sent in zip(src, trg):
                self.dataset.set_item(src_sent, trg_sent)
        else:
            self.dataset.has_trg = False
            for sentence in src:
                self.dataset.set_item(sentence)

        assert len(self.dataset) > 0

        _, _, translations, tokens, probs, attention_probs = predict(
            model=self.model,
            data=self.dataset,
            compute_loss=trg is not None,
            device=self.device,
            n_gpu=self.n_gpu,
            normalization=self.normalization,
            num_workers=self.num_workers,
            cfg=test_cfg,
            fp16=self.fp16,
        )
        if translations:
            assert len(src) * test_cfg.get("n_best", 1) == len(translations)

        self.dataset.cache = {}  # reset cache

        return translations, tokens, probs, attention_probs, test_cfg

    def plot_attention(self, src: str, trg: str, attention_scores: np.ndarray) -> None:
        # preprocess and tokenize sentences
        self.dataset.cache = {}  # reset cache
        self.dataset.has_trg = True
        self.dataset.set_item(src, trg)
        src_tokens = self.dataset.get_item(idx=0,
                                           lang=self.dataset.src_lang,
                                           is_train=False)
        trg_tokens = self.dataset.get_item(idx=0,
                                           lang=self.dataset.trg_lang,
                                           is_train=False)
        self.dataset.cache = {}  # reset cache

        assert len(src_tokens) + 1 == attention_scores.shape[1]
        assert len(trg_tokens) + 1 == attention_scores.shape[0]

        # plot attention scores
        fig = px.imshow(
            attention_scores,
            labels={
                "x": "Src",
                "y": "Trg",
            },
            x=src_tokens + [EOS_TOKEN],
            y=trg_tokens + [EOS_TOKEN],
        )
        fig.update_xaxes(side="top", tickangle=270)
        fig.show()
