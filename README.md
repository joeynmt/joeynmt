# &nbsp; ![Joey-NMT](joey-small.png) Joey NMT
[![Build Status](https://travis-ci.com/joeynmt/joeynmt.svg?branch=master)](https://travis-ci.org/joeynmt/joeynmt)
[![Gitter](https://badges.gitter.im/joeynmt/community.svg)](https://gitter.im/joeynmt/community?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge)

## Goal and Purpose
Joey NMT framework is developed for educational purposes. 
It aims to be a **clean** and **minimalistic** code base to help novices 
find fast answers to the following questions.
- How to implement classic NMT architectures (RNN and Transformer) in PyTorch?
- What are the building blocks of these architectures and how do they interact?
- How to modify these blocks (e.g. deeper, wider, ...)?
- How to modify the training procedure (e.g. add a regularizer)?

In contrast to other NMT frameworks, we will **not** aim for the most recent 
features or speed through engineering or training tricks
since this often goes in hand with an increase in code complexity 
and a decrease in readability.

However, Joey NMT re-implements baselines from major publications.

Check out the detailed [documentation](https://joeynmt.readthedocs.io) and our [paper](https://arxiv.org/abs/1907.12484).

## Contributors
Joey NMT is developed by [Joost Bastings](https://bastings.github.io) (University of Amsterdam) and [Julia Kreutzer](http://www.cl.uni-heidelberg.de/~kreutzer/) (Heidelberg University).

## Features
Joey NMT implements the following features (aka the minimalist toolkit of NMT):
- Recurrent Encoder-Decoder with GRUs or LSTMs
- Transformer Encoder-Decoder
- Attention Types: MLP, Dot, Multi-Head, Bilinear
- Word-, BPE- and character-based input handling
- BLEU, ChrF evaluation
- Beam search with length penalty and greedy decoding
- Customizable initialization
- Attention visualization
- Learning curve plotting

## Coding
In order to keep the code clean and readable, we make use of:
- Style checks: pylint with (mostly) PEP8 conventions, see `.pylintrc`.
- Typing: Every function has documented input types.
- Docstrings: Every function, class and module has docstrings describing their purpose and usage.
- Unittests: Every module has unit tests, defined in `test/unit/`.
Travis CI runs the tests and pylint on every push to ensure the repository stays clean.

## Installation
Joey NMT is built on [PyTorch](https://pytorch.org/) and [torchtext](https://github.com/pytorch/text) for Python >= 3.5.

1. Clone this repository:
`git clone https://github.com/joeynmt/joeynmt.git`
2. Install joeynmt and it's requirements:
`cd joeynmt`
`pip3 install .` (you might want to add `--user` for a local installation).
3. Run the unit tests:
`python3 -m unittest`

**Warning!** When running on *GPU* you need to manually install the suitable PyTorch version for your [CUDA](https://developer.nvidia.com/cuda-zone) version. This is described in the [PyTorch installation instructions](https://pytorch.org/get-started/locally/).


## Usage

For details, follow the tutorial in [the docs](https://joeynmt.readthedocs.io).

### Data Preparation

#### Parallel Data
For training a translation model, you need parallel data, i.e. a collection of source sentences and reference translations that are aligned sentence-by-sentence and stored in two files, 
such that each line in the reference file is the translation of the same line in the source file.

#### Pre-processing
Before training a model on it, parallel data is most commonly filtered by length ratio, tokenized and true- or lowercased.

The Moses toolkit provides a set of useful [scripts](https://github.com/moses-smt/mosesdecoder/tree/master/scripts) for this purpose.

In addition, you might want to build the NMT model not on the basis of words, but rather sub-words or characters (the `level` in JoeyNMT configurations).
Currently, JoeyNMT supports the byte-pair-encodings (BPE) format by [subword-nmt](https://github.com/rsennrich/subword-nmt).

### Configuration
Experiments are specified in configuration files, in simple [YAML](http://yaml.org/) format. You can find examples in the `configs` directory.
`small.yaml` contains a detailed explanation of configuration options.

Most importantly, the configuration contains the description of the model architecture (e.g. number of hidden units in the encoder RNN), 
paths to the training, development and test data, and the training hyperparameters (learning rate, validation frequency etc.).

### Training

#### Start
For training, run 

`python3 -m joeynmt train configs/small.yaml`. 

This will train a model on the training data specified in the config (here: `small.yaml`), 
validate on validation data, 
and store model parameters, vocabularies, validation outputs and a small number of attention plots in the `model_dir` (also specified in config).

Note that pre-processing like tokenization or BPE-ing is not included in training, but has to be done manually before.

Tip: Be careful not to overwrite models, set `overwrite: False` in the model configuration.

#### Validations
The `validations.txt` file in the model directory reports the validation results at every validation point. 
Models are saved whenever a new best validation score is reached, in `batch_no.ckpt`, where `batch_no` is the number of batches the model has been trained on so far.
`best.ckpt` links to the checkpoint that has so far achieved the best validation score.


#### Visualization
JoeyNMT uses Tensorboard to visualize training and validation curves and attention matrices during training.
Launch [Tensorboard](https://github.com/tensorflow/tensorboard) with `tensorboard --logdir model_dir/tensorboard` (or `python -m tensorboard.main ...`) and then open the url (default: `localhost:6006`) with a browser. 

For a stand-alone plot, run `python3 scripts/plot_validation.py model_dir --plot_values bleu PPL --output_path my_plot.pdf` to plot curves of validation BLEU and PPL.

#### CPU vs. GPU
For training on a GPU, set `use_cuda` in the config file to `True`. This requires the installation of required CUDA libraries.


### Translating

There are three options for testing what the model has learned.

Whatever data you feed the model for translating, make sure it is properly pre-processed, just as you pre-processed the training data, e.g. tokenized and split into subwords (if working with BPEs).

#### 1. Test Set Evaluation 
For testing and evaluating on your parallel test/dev set, run 

`python3 -m joeynmt test configs/small.yaml --output_path out`.

This will generate translations for validation and test set (as specified in the configuration) in `out.[dev|test]`
with the latest/best model in the `model_dir` (or a specific checkpoint set with `load_model`).
It will also evaluate the outputs with `eval_metric`.
If `--output_path` is not specified, it will not store the translation, and only do the evaluation and print the results.

#### 2. File Translation
In order to translate the contents of a file not contained in the configuration (here `my_input.txt`), simply run

`python3 -m joeynmt translate configs/small.yaml < my_input.txt > out`.

The translations will be written to stdout or alternatively`--output_path` if specified.

#### 3. Interactive
If you just want try a few examples, run

`python3 -m joeynmt translate configs/small.yaml`

and you'll be prompted to type input sentences that JoeyNMT will then translate with the model specified in the configuration.


## Documentation and Tutorial
[The docs](https://joeynmt.readthedocs.io) include an overview of the NMT implementation, a walk-through tutorial for building, training, tuning, testing and inspecting an NMT system, the [API documentation]() and [FAQs]().

A screencast of the tutorial is available on [YouTube](https://www.youtube.com/watch?v=PzWRWSIwSYc).

## Benchmarks
Benchmark results on WMT and IWSLT datasets are reported [here](benchmarks.md).

## Pre-trained Models
Pre-trained models from reported benchmarks for download (contains config, vocabularies, best checkpoint and dev/test hypotheses):

### WMT17
Following the pre-processing of the [Sockeye paper](https://arxiv.org/abs/1712.05690).

- [WMT17 en-de "best" RNN](https://www.cl.uni-heidelberg.de/statnlpgroup/joeynmt/wmt_ende_best.tar.gz) (2G)
- [WMT17 lv-en "best" RNN](https://www.cl.uni-heidelberg.de/statnlpgroup/joeynmt/wmt_lven_best.tar.gz) (1.9G)
- [WMT17 en-de Transformer](https://www.cl.uni-heidelberg.de/statnlpgroup/joeynmt/wmt_ende_transformer.tar.gz) (664M)
- [WMT17 lv-en Transformer](https://www.cl.uni-heidelberg.de/statnlpgroup/joeynmt/wmt_lven_transformer.tar.gz) (650M)

### Autshumato
Traing with data provided in the [Ukuxhumana project](https://github.com/LauraMartinus/ukuxhumana), with additional tokenization of the training data with the [Moses tokenizer](https://github.com/moses-smt/mosesdecoder/blob/master/scripts/tokenizer/tokenizer.perl).

- [Autshumato en-af small Transformer](https://www.cl.uni-heidelberg.de/statnlpgroup/joeynmt/autshumato_enaf_transformer.tar.gz) (147M)
- [Autshumato af-en small Transformer](https://www.cl.uni-heidelberg.de/statnlpgroup/joeynmt/autshumato_afen_transformer.tar.gz) (147M)
- [Autshumato en-nso small Transformer](https://www.cl.uni-heidelberg.de/statnlpgroup/joeynmt/autshumato_ennso_transformer.tar.gz) (147M)
- [Autshumato nso-en small Transformer](https://www.cl.uni-heidelberg.de/statnlpgroup/joeynmt/autshumato_nsoen_transformer.tar.gz) (147M)
- [Autshumato en-tn small Transformer](https://www.cl.uni-heidelberg.de/statnlpgroup/joeynmt/autshumato_entn_transformer.tar.gz) (319M)
- [Autshumato tn-en small Transformer](https://www.cl.uni-heidelberg.de/statnlpgroup/joeynmt/autshumato_tnen_transformer.tar.gz) (321M)
- [Autshumato en-ts small Transformer](https://www.cl.uni-heidelberg.de/statnlpgroup/joeynmt/autshumato_ents_transformer.tar.gz) (229M)
- [Autshumato ts-en small Transformer](https://www.cl.uni-heidelberg.de/statnlpgroup/joeynmt/autshumato_tsen_transformer.tar.gz) (229M)
- [Autshumato en-zu small Transformer](https://www.cl.uni-heidelberg.de/statnlpgroup/joeynmt/autshumato_enzu_transformer.tar.gz) (147M)
- [Autshumato zu-en small Transformer](https://www.cl.uni-heidelberg.de/statnlpgroup/joeynmt/autshumato_zuen_transformer.tar.gz) (147M)

If you trained JoeyNMT on your own data and would like to share it, please email us so we can add it to the collection of pre-trained models.

## Contributing
Since this codebase is supposed to stay clean and minimalistic, contributions addressing the following are welcome:
- code correctness
- code cleanliness
- documentation quality
- speed or memory improvements
- resolving issues
- providing pre-trained models

Code extending the functionalities beyond the basics will most likely not end up in the master branch, but we're curions to learn what you used Joey for.

## Projects and Extensions
Here we'll collect projects and repositories that are based on Joey, so you can find inspiration and examples on how to modify and extend the code.

- **African NMT**. [@jaderabbit](https://github.com/jaderabbit) started an initiative at the Indaba Deep Learning School 2019 to ["put African NMT on the map"](https://twitter.com/alienelf/status/1168159616167010305). The goal is to build and collect NMT models for low-resource African languages. The [Masakhane repository](https://github.com/jaderabbit/masakhane) contains and explains all the code you need to train JoeyNMT and points to data sources. It also contains benchmark models and configurations that members of Masakhane have built for various African languages.
- **Slack Joey**. [Code](https://github.com/juliakreutzer/slack-joey) to locally deploy a Joey NMT model as chat bot in a Slack workspace. It's a convenient way to probe your model without having to implement an API. And bad translations for chat messages can be very entertaining, too ;)
- **Flask Joey**. [@kevindegila](https://github.com/kevindegila) built a [flask interface to Joey](https://github.com/kevindegila/flask-joey), so you can deploy your trained model in a web app and query it in the browser. 
- **User Study**. We evaluated the code quality of this repository by testing the understanding of novices through quiz questions. Find the details in Section 3 of the [Joey NMT paper](https://arxiv.org/abs/1907.12484).
- **Self-Regulated Interactive Seq2Seq Learning**. Julia Kreutzer and Stefan Riezler. Published at ACL 2019. [Paper](https://arxiv.org/abs/1907.05190) and [Code](https://github.com/juliakreutzer/joeynmt/tree/acl19). This project augments the standard fully-supervised learning regime by weak and self-supervision for a better trade-off of quality and supervision costs in interactive NMT.
- **Speech Joey**. [@Sariyusha](https://github.com/Sariyusha) is giving Joey ears for speech translation. [Code](https://github.com/Sariyusha/speech_joey).
- **Hieroglyph Translation**. Joey NMT was used to translate hieroglyphs in [this IWSLT 2019 paper](https://www.cl.uni-heidelberg.de/statnlpgroup/publications/IWSLT2019.pdf) by Philipp Wiesenbach and Stefan Riezler. They gave Joey NMT multi-tasking abilities. 

If you used Joey NMT for a project, publication or built some code on top of it, let us know and we'll link it here.


## Contact
Please leave an issue if you have questions or issues with the code.

For general questions, email us at `joeynmt <at> gmail.com`.

## Reference
If you use Joey NMT in a publication or thesis, please cite the following [paper](https://arxiv.org/abs/1907.12484):

```
@ARTICLE{JoeyNMT,
author = {{Kreutzer}, Julia and {Bastings}, Joost and {Riezler}, Stefan},
title = {Joey {NMT}: A Minimalist {NMT} Toolkit for Novices},
journal = {To Appear in EMNLP-IJCNLP 2019: System Demonstrations},
year = {2019},
month = {Nov},
address = {Hong Kong}
url = {https://arxiv.org/abs/1907.12484}
}
```

## Naming
Joeys are [infant marsupials](https://en.wikipedia.org/wiki/Marsupial#Early_development). 

