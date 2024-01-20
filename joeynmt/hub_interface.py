# coding: utf-8
"""
Torch Hub Interface
"""
from pathlib import Path
from typing import List, NamedTuple, Optional, Union

import numpy as np
import plotly.express as px
from torch import nn

from joeynmt.config import BaseConfig, TestConfig, load_config, parse_global_args
from joeynmt.datasets import BaseDataset, StreamDataset
from joeynmt.helpers_for_ddp import get_logger
from joeynmt.model import Model
from joeynmt.prediction import predict, prepare

logger = get_logger(__name__)

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
    cfg_file: Union[str, Path] = "config.yaml",
    **kwargs,
):
    """Prepare model and data placeholder"""
    # model dir
    model_dir = Path(model_name_or_path
                     ) if isinstance(model_name_or_path, str) else model_name_or_path
    assert model_dir.is_dir(), model_dir

    # cfg file
    cfg_file = _check_file_path(cfg_file, model_dir)
    assert cfg_file.is_file(), cfg_file
    cfg = load_config(cfg_file)
    cfg.update(kwargs)
    cfg["model_dir"] = model_dir.as_posix()  # override model_dir

    # rewrite paths in cfg
    for side in ["src", "trg"]:
        data_side = cfg["data"][side]
        data_side["voc_file"] = _check_file_path(data_side["voc_file"],
                                                 model_dir).as_posix()
        if "tokenizer_cfg" in data_side:
            for tok_model in ["codes", "model_file"]:
                if tok_model in data_side["tokenizer_cfg"]:
                    data_side["tokenizer_cfg"][tok_model] = _check_file_path(
                        data_side["tokenizer_cfg"][tok_model], model_dir
                    ).as_posix()

    if "load_model" in cfg["testing"]:
        cfg["testing"]["load_model"] = _check_file_path(
            cfg["testing"]["load_model"], model_dir
        ).as_posix()

    # parse args
    args = parse_global_args(cfg, rank=0, mode="translate")

    # load the data
    model, _, _, test_data = prepare(args, rank=0, mode="translate")

    return model, test_data, args


class TranslatorHubInterface(nn.Module):
    """
    PyTorch Hub interface for generating sequences from a pre-trained
    encoder-decoder model.
    """

    def __init__(self, model: Model, dataset: BaseDataset, args: BaseConfig):
        super().__init__()
        self.args = args
        self.dataset = dataset
        self.model = model
        if self.args.device.type == "cuda":
            self.model.to(self.args.device)
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

        translations, tokens, probs, attn, test_cfg = self._generate(src, trg, **kwargs)

        beam_size = test_cfg.get("beam_size", 1)
        n_best = test_cfg.get("n_best", 1)

        out = []
        for i in range(len(src)):
            offset = i * n_best
            pred = PredictionOutput(
                translation=trg[i] if trg else translations[offset:offset + n_best],
                tokens=tokens[offset:offset + n_best],
                token_probs=probs[offset:offset + n_best] if beam_size == 1 else None,
                sequence_probs=[p[0] for p in probs[offset:offset + n_best]] \
                    if beam_size > 1 else None,  # noqa: E131
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
        src_prompt: Optional[List[str]] = None,
        trg_prompt: Optional[List[str]] = None,
        **kwargs,
    ) -> List[str]:

        # overwrite config
        test_cfg = self.args.test._asdict()
        test_cfg.update(kwargs)

        assert isinstance(self.dataset, StreamDataset), self.dataset
        test_cfg["batch_type"] = "sentence"
        test_cfg["batch_size"] = len(src)

        if src_prompt:
            assert len(src) == len(
                src_prompt
            ), "src and src_prompt must have the same length!"
        else:
            src_prompt = [None] * len(src)

        if trg_prompt:
            assert len(src) == len(
                trg_prompt
            ), "trg and trg_prompt must have the same length!"
        else:
            trg_prompt = [None] * len(src)

        self.dataset.reset_cache()  # reset cache
        if trg is not None:
            assert len(src) == len(trg), "src and trg must have the same length!"
            self.dataset.has_trg = True
            test_cfg["n_best"] = 1
            test_cfg["beam_size"] = 1
            test_cfg["return_prob"] = "ref"
            for src_sent, trg_sent, src_p, trg_p in zip(
                src, trg, src_prompt, trg_prompt
            ):
                self.dataset.set_item(src_sent, trg_sent, src_p, trg_p)
        else:
            self.dataset.has_trg = False
            for src_sent, src_p, trg_p in zip(src, src_prompt, trg_prompt):
                self.dataset.set_item(src_sent, None, src_p, trg_p)

        assert len(self.dataset) == len(src), (len(self.dataset), self.dataset.cache)

        _, _, translations, tokens, probs, attention_probs = predict(
            model=self.model,
            data=self.dataset,
            compute_loss=trg is not None,
            device=self.args.device,
            n_gpu=self.args.n_gpu,
            normalization=self.args.train.normalization,
            num_workers=self.args.num_workers,
            args=TestConfig(**test_cfg),
            autocast=self.args.autocast,
        )
        if translations:
            assert len(src) * test_cfg.get("n_best", 1) == len(translations)

        self.dataset.reset_cache()  # reset cache

        return translations, tokens, probs, attention_probs, test_cfg

    def plot_attention(self, src: str, trg: str, attention_scores: np.ndarray) -> None:
        # preprocess and tokenize sentences
        self.dataset.reset_cache()  # reset cache
        self.dataset.has_trg = True
        self.dataset.set_item(src, trg)
        src_tokens = self.dataset.get_item(
            idx=0, lang=self.dataset.src_lang, is_train=False
        )
        trg_tokens = self.dataset.get_item(
            idx=0, lang=self.dataset.trg_lang, is_train=False
        )

        self.dataset.reset_cache()  # reset cache

        assert len(src_tokens) + 1 == attention_scores.shape[1]
        assert len(trg_tokens) + 1 == attention_scores.shape[0]

        # plot attention scores
        fig = px.imshow(
            attention_scores,
            labels={
                "x": "Src",
                "y": "Trg",
            },
            x=src_tokens + [self.dataset.tokenizer[self.dataset.src_lang].eos_token],
            y=trg_tokens + [self.dataset.tokenizer[self.dataset.trg_lang].eos_token],
        )
        fig.update_xaxes(side="top", tickangle=270)
        fig.show()
