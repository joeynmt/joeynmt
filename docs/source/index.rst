.. _index:

====================================
Welcome to Joey NMT's documentation!
====================================

Goals & Purposes
----------------

Joey NMT framework is developed for educational purposes.
It aims to be a **clean** and **minimalistic** code base to help novices find fast answers to the following questions.

- How to implement classic NMT architectures (RNN and Transformer) in PyTorch?
- What are the building blocks of these architectures and how do they interact?
- How to modify these blocks (e.g. deeper, wider, ...)?
- How to modify the training procedure (e.g. add a regularizer)?

In contrast to other NMT frameworks, we will **not** aim for the most recent features or speed through engineering or training tricks since this often goes in hand with an increase in code complexity and a decrease in readability.

However, Joey NMT re-implements baselines from major publications.

Features
--------

Joey NMT implements the following features (aka the minimalist toolkit of NMT):

- Recurrent Encoder-Decoder with GRUs or LSTMs
- Transformer Encoder-Decoder
- Attention Types: MLP, Dot, Multi-Head, Bilinear
- Word-, BPE- and character-based tokenization
- BLEU, ChrF evaluation
- Beam search with length penalty and greedy decoding
- Customizable initialization
- Attention visualization
- Learning curve plotting
- Scoring hypotheses and references
- Multilingual translation with language tags


.. toctree::
    :hidden:
    :caption: Getting Started
    :maxdepth: 3

    install
    cli
    tutorial
    benchmarks


.. toctree::
    :hidden:
    :caption: Development
    :maxdepth: 3

    overview
    api
    faq
    resources
    changelog
