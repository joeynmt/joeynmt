"""PyTorch Hub models

Usage:
    import torch
    model = torch.hub.load('repo', 'model')
"""

from pathlib import Path
import importlib
import tarfile

import torch.hub

dependencies = ['torch', 'yaml', 'numpy', 'sentencepiece', 'subword_nmt', 'sacremoses']

# Check for required dependencies and raise a RuntimeError if any are missing.
missing_deps = []
for dep in dependencies:
    try:
        importlib.import_module(dep)
    except ImportError:
        missing_deps.append(dep)
if len(missing_deps) > 0:
    raise RuntimeError("Missing dependencies: {}".format(", ".join(missing_deps)))


# only do joeynmt imports after checking for dependencies
from joeynmt.hub_interface import _from_pretrained, TranslatorHubInterface

ROOT_URL = "https://cl.uni-heidelberg.de/statnlpgroup/joeynmt2"


def _download_and_extract(model_name_or_path: str, ext: str = ".tar.gz") -> Path:
    hub_dir = Path(torch.hub.get_dir())
    download_dir = hub_dir / model_name_or_path
    torch.hub.download_url_to_file(f"{ROOT_URL}/{model_name_or_path}{ext}",
                                   download_dir.with_suffix(ext))
    # extract
    if ext.startswith(".tar"):
        with tarfile.open(download_dir.with_suffix(ext)) as f:
            f.extractall(hub_dir)
    # delete .tar.gz
    download_dir.with_suffix(ext).unlink()
    assert download_dir.is_dir(), download_dir
    return download_dir


def _load_from_remote(
    model_name_or_path: str,
    ckpt_file: str = "best.ckpt",
    cfg_file: str = "config.yaml",
    **kwargs
) -> TranslatorHubInterface:
    download_dir = _download_and_extract(model_name_or_path)
    config, test_data, model = _from_pretrained(
        model_name_or_path=download_dir,
        ckpt_file=ckpt_file,
        cfg_file=cfg_file,
        **kwargs,
    )
    return TranslatorHubInterface(config, test_data, model)


def transformer_iwslt14_deen_bpe(*args, **kwargs) -> TranslatorHubInterface:
    """
    IWSLT14 deen transformer
    See
    """
    return _load_from_remote(
        model_name_or_path="transformer_iwslt14_deen_bpe",
        ckpt_file="best.ckpt",
        cfg_file="config_v2.yaml",
        **kwargs
    )


def rnn_iwslt14_deen_bpe(*args, **kwargs) -> TranslatorHubInterface:
    """
    IWSLT14 deen RNN
    See
    """
    return _load_from_remote(
        model_name_or_path="rnn_iwslt14_deen_bpe",
        ckpt_file="best.ckpt",
        cfg_file="config_v2.yaml",
        **kwargs
    )


def wmt14_deen(*args, **kwargs) -> TranslatorHubInterface:
    """
    WMT14 deen
    See
    """
    return _load_from_remote(
        model_name_or_path="wmt14_deen",
        ckpt_file="avg5.ckpt",
        cfg_file="config.yaml",
        **kwargs
    )


def wmt14_ende(*args, **kwargs) -> TranslatorHubInterface:
    """
    WMT14 ende
    See
    """
    return _load_from_remote(
        model_name_or_path="wmt14_ende",
        ckpt_file="avg5.ckpt",
        cfg_file="config.yaml",
        **kwargs
    )


def local(model_name_or_path, ckpt_file, cfg_file, **kwargs) -> TranslatorHubInterface:
    """
    joeynmt model saved in local
    """
    config, test_data, model = _from_pretrained(
        model_name_or_path=model_name_or_path,
        ckpt_file=ckpt_file,
        cfg_file=cfg_file,
        **kwargs,
    )
    return TranslatorHubInterface(config, test_data, model)


if __name__ == '__main__':
    translator = transformer_iwslt14_deen_bpe()
    print(translator.translate(['Hello World!']))
