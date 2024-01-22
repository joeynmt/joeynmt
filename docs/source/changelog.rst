.. _changelog:

==========
Change log
==========

v2.3 \- Jan 25, 2024
--------------------

- introduced `DistributedDataParallel <https://pytorch.org/tutorials/beginner/dist_overview.html>`_
- implemented language tags, see :notebooks:`torchhub.ipynb`
- released a `iwslt14 de-en-fr multilingual model <https://huggingface.co/may-ohta/iwslt14_prompt>`_ (trained using DDP)
- special symbols definition refactoring
- configuration refactoring
- autocast refactoring
- enabled activation function selection
- bugfixes
- upgrade to python 3.11, torch 2.1.2
- documentation refactoring


v2.2 \- Jan 15, 2023
--------------------

- compatibility with torch 2.0 tested
- torchhub introduced
- bugfixes, minor refactoring


v2.1 \- Sep 18, 2022
--------------------

- upgrade to python 3.10, torch 1.12
- replace Automated Mixed Precision from NVIDA's amp to Pytorch's amp package
- replace `discord.py <https://github.com/Rapptz/discord.py>`_ with `pycord <https://github.com/Pycord-Development/pycord>`_ in the Discord Bot demo
- data iterator refactoring
- add wmt14 ende / deen benchmark trained on v2 from scratch
- add tokenizer tutorial
- minor bugfixes


v2.0 \- Jun 2, 2022
-------------------

*Breaking change!*

- upgrade to python 3.9, torch 1.11
- ``torchtext.legacy`` dependencies are completely replaced by ``torch.utils.data``
- :joeynmt:`tokenizers.py`: handles tokenization internally (also supports bpe-dropout!)
- :joeynmt:`datasets.py`: loads data from plaintext, tsv, and huggingface's `datasets <https://github.com/huggingface/datasets>`_
- :joeynmt:`build_vocab.py`: trains subwords, creates joint vocab
- enhancement in decoding
  - scoring with hypotheses or references
  - repetition penalty, ngram blocker
  - attention plots for transformers
- yapf, isort, flake8 introduced
- bugfixes, minor refactoring

.. warning::

    The models trained with Joey NMT v1.x can be decoded with Joey NMT v2.0.
    But there is no guarantee that you can reproduce the same score as before.


v1.5 \- Jan 18, 2022
--------------------

- requirements update (Six >= 1.12)


v1.4 \- Jan 18, 2022
--------------------

- upgrade to sacrebleu 2.0, python 3.7, torch 1.8
- bugfixes


v1.3 \- Apr 14, 2021
--------------------

- upgrade to torchtext 0.9 (torchtext -> torchtext.legacy)
- n-best decoding
- demo colab notebook


v1.0 \- Oct 31, 2020
--------------------

- Multi-GPU support
- fp16 (half precision) support


v0.9  \- Jul 28, 2019
---------------------

- pre-release
