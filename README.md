# &nbsp; ![Joey-NMT](joey-small.png) JoeyNMT 2.0
![joeynmt](https://github.com/may-/main/actions/workflows/main.yml/badge.svg)

## Goal and Purpose
:koala: JoeyNMT framework is developed for educational purposes.
It aims to be a **clean** and **minimalistic** code base to help novices 
find fast answers to the following questions.
- :grey_question: How to implement classic NMT architectures (RNN and Transformer) in PyTorch?
- :grey_question: What are the building blocks of these architectures and how do they interact?
- :grey_question: How to modify these blocks (e.g. deeper, wider, ...)?
- :grey_question: How to modify the training procedure (e.g. add a regularizer)?

In contrast to other NMT frameworks, we will **not** aim for the most recent features
or speed through engineering or training tricks since this often goes in hand with an
increase in code complexity and a decrease in readability. :eyes:

However, JoeyNMT re-implements baselines from major publications.

Check out the detailed [documentation](https://joeynmt.readthedocs.io) and our
[paper](https://arxiv.org/abs/1907.12484).

## Contributors
JoeyNMT was initially developed and is maintained by [Jasmijn Bastings](https://github.com/bastings) (University of Amsterdam) and [Julia Kreutzer](https://juliakreutzer.github.io/) (Heidelberg University), now both at Google Research. [Mayumi Ohta](https://www.cl.uni-heidelberg.de/statnlpgroup/members/ohta/) at Heidelberg University is continuing the legacy.

Welcome to our new contributors :hearts:, please don't hesitate to open a PR or an issue
if there's something that needs improvement!

## Features
JoeyNMT implements the following features (aka the minimalist toolkit of NMT :wrench:):
- Recurrent Encoder-Decoder with GRUs or LSTMs
- Transformer Encoder-Decoder
- Attention Types: MLP, Dot, Multi-Head, Bilinear
- Word-, BPE- and character-based tokenization
- BLEU, ChrF evaluation
- Beam search with length penalty and greedy decoding
- Customizable initialization
- Attention visualization
- Learning curve plotting



## Installation
JoeyNMT is built on [PyTorch](https://pytorch.org/) and [torchtext](https://github.com/pytorch/text) for Python >= 3.9.

A. [*Now also directly with pip!*](https://pypi.org/project/joeynmt/)
  ```
  $ pip install joeynmt
  ```
  
  If you want to use GPUs add: `pip install torch --extra-index-url https://download.pytorch.org/whl/cu113`, for CUDA v11.3.
  You'll need this in particular when working on Google Colab (if it doesn't already have the right Torch version installed).
  
B. From source
  1. Clone this repository:
  `git clone https://github.com/joeynmt/joeynmt.git`
  2. Install joeynmt and it's requirements:
  `cd joeynmt`
  `pip3 install .` (you might want to add `--user` for a local installation).
  3. Run the unit tests:
  `python3 -m unittest`

> :warning: **Warning!** When running on *GPU* you need to manually install the suitable
> PyTorch version (1.11.0) for your [CUDA](https://developer.nvidia.com/cuda-zone)
> version. This is described in the [PyTorch installation instructions](https://pytorch.org/get-started/locally/).

## Change logs
### v2.0.0 *Breaking change!*
- upgrade to python 3.9, torch 1.11
- `torchtext.legacy` dependencies are completely replaced by `torch.utils.data`
- `joeynmt/tokenizers.py`: handles tokenization internally
- `joeynmt/datasets.py`: loads data from plaintext, tsv, huggingface's [datasets]()
- `scripts/build_vocab.py`: trains subwords, creates joint vocab
- black, isort, flake8
- bugfixes, minor refactoring

> :warning: 

### v1.4
- upgrade to sacrebleu 2.0, python 3.7, torch 1.8
- bugfixes

### v1.3
- upgrade to torchtext 0.9 (torchtext -> torchtext.legacy)
- n-best decoding
- demo colab notebook

### v1.0
- Multi-GPU support

## Usage

> :warning: We made a breaking change in v2.0. For v1.x, please refer the
> [archive](docs/JoeyNMT_v1.md).

For details, follow the tutorials in [`notebooks`](notebooks) dir.
- [quick-start-with-joeynmt2.0](notebooks/joeynmt_v2.ipynb)
- [pretrained embeddings](notebooks/pretrained embeddings.ipynb)
- [speech-translation](notebooks/speech_joeynmt.ipynb)


JoeyNMT has 3 modes: `train`, `test`, and `translate`, and all of them takes a
YAML-style config file as argument. You can find examples in the `configs` directory.
`small.yaml` contains a detailed explanation of configuration options.

Most importantly, the configuration contains the description of the model architecture
(e.g. number of hidden units in the encoder RNN), paths to the training, development and
test data, and the training hyperparameters (learning rate, validation frequency etc.).

> :bulb: Note that subword model training and joint vocabulary creation is not included
> in the 3 modes above, has to be done before. We provide a script that takes care of it: 
> `scritps/build_vocab.py`. See the [quick-start tutorial](notebooks/joeynmt_v2.0.ipynb)
> for details.
> ```
> $ python scripts/build_vocab.py configs/small.yaml --joint
> ```

### `train` mode
For training, run 
```bash
$ python -m joeynmt train configs/small.yaml
```
This will train a model on the training data, validate on validation data, and store
model parameters, vocabularies, validation outputs. All needed information should be
specified in the `data`, `training` and `model` section in the config file (here
`configs/small.yaml`).

```
model_dir/

```



> :bulb: **Tip:** Be careful not to overwrite `model_dir`, set `overwrite: False` in the
> config file.

### `test` mode
```
$ python -m joeynmt test configs/small.yaml --ckpt model_dir/avg.ckpt
```
This will generate translations for validation and test set (as specified in the
configuration) in `out.[dev|test]` with the specified checkpoint. If `--ckpt` is not
specified, the path in `load_model` in config file or the best model in the `model_dir`
will be loaded.

### `translate` mode
This mode accepts inputs from stdin and generate translations.

- File translation
  ```
  $ python -m joeynmt translate configs/small.yaml < my_input.txt > output.txt
  ```

- Interactive translation
  ```
  $ python -m joeynmt translate configs/small.yaml
  ```
  You'll be prompted to type an input sentence. JoeyNMT will then translate with the 
  model specified in `--ckpt` or the config file.

  > :bulb: **Tip:** interactive `translate` mode doesn't work with Multi-GPU.
  > Please run it on single GPU or CPU.



## Benchmarks & pretrained models

> :warning: *Warning!* These models are trained with JoeynNMT v2.0, not compatible with JoeyNMT v1.x.

### iwslt14 deen



### wmt17 ende



### JParaCrawl


## Coding
In order to keep the code clean and readable, we make use of:
- Style checks:
  - [pylint]() with (mostly) PEP8 conventions, see `.pylintrc`.
  - [black](), [isort](), and [flake8](); see `setup.cfg` and `Makefile`.
- Typing: Every function has documented input types.
- Docstrings: Every function, class and module has docstrings describing their purpose and usage.
- Unittests: Every module has unit tests, defined in `test/unit/`.

On every push, unittests and linters are triggered by github's workflow to ensure the
repository stays clean.

## Contributing
Since this codebase is supposed to stay clean and minimalistic, contributions addressing
the following are welcome:
- code correctness
- code cleanliness
- documentation quality
- speed or memory improvements
- resolving issues
- providing pre-trained models

Code extending the functionalities beyond the basics will most likely not end up in the
main branch, but we're curious to learn what you used JoeyNMT for.

## Projects and Extensions
Here we'll collect projects and repositories that are based on JoeyNMT, so you can find
inspiration and examples on how to modify and extend the code.
- :spider_web: **Masakhane Web**. [@CateGitau](https://github.com/categitau), [@Kabongosalomon](https://github.com/Kabongosalomon), [@vukosim](https://github.com/vukosim) and team built a whole web translation platform for the African NMT models that Masakhane built with JoeyNMT. The best is: it's completely open-source, so anyone can contribute new models or features. Try it out [here](http://translate.masakhane.io/), and check out the [code](https://github.com/dsfsi/masakhane-web).
- :gear: **MutNMT**. [@sjarmero](https://github.com/sjarmero) created a web application to train NMT: it lets the user train, inspect, evaluate and translate with JoeyNMT --- perfect for NMT newbies! Code [here](https://github.com/Prompsit/mutnmt). The tool was developed by [Prompsit](https://www.prompsit.com/) in the framework of the European project [MultiTraiNMT](http://www.multitrainmt.eu/).
- :star2: **Cantonese-Mandarin Translator**. [@evelynkyl](https://github.com/evelynkyl/) trained different NMT models for translating between the low-resourced Cantonese and Mandarin,  with the help of some cool parallel sentence mining tricks! Check out her work [here](https://github.com/evelynkyl/yue_nmt).
- :book: **Russian-Belarusian Translator**. [@tsimafeip](https://github.com/tsimafeip) built a translator from Russian to Belarusian and adapted it to legal and medical domains. The code can be found [here](https://github.com/tsimafeip/Translator/).
- :muscle: **Reinforcement Learning**. [@samuki](https://github.com/samuki/) implemented various policy gradient variants in JoeyNMT: here's the [code](https://github.com/samuki/reinforce-joey), could the logo be any more perfect? :muscle: :koala:
- :hand: **Sign Language Translation**. [@neccam](https://github.com/neccam/) built a sign language translator that continuosly recognizes sign language and translates it. Check out the [code](https://github.com/neccam/slt) and the [CVPR 2020 paper](https://openaccess.thecvf.com/content_CVPR_2020/html/Camgoz_Sign_Language_Transformers_Joint_End-to-End_Sign_Language_Recognition_and_Translation_CVPR_2020_paper.html)!
- :abc: [@bpopeters](https://github.com/bpopeters/) built [Possum-NMT](https://github.com/deep-spin/sigmorphon-seq2seq) for multilingual grapheme-to-phoneme transduction and morphologic inflection. Read their [paper](https://www.aclweb.org/anthology/2020.sigmorphon-1.4.pdf) for SIGMORPHON 2020!
- :camera: **Image Captioning**. [@pperle](https://github.com/pperle) and [@stdhd](https://github.com/stdhd) built an imagine captioning tool on top of JoeyNMT, check out the [code](https://github.com/stdhd/image_captioning) and the [demo](https://image2caption.pascalperle.de/)!
- :bulb: **Joey Toy Models**. [@bricksdont](https://github.com/bricksdont) built a [collection of scripts](https://github.com/bricksdont/joeynmt-toy-models) showing how to install JoeyNMT, preprocess data, train and evaluate models. This is a great starting point for anyone who wants to run systematic experiments, tends to forget python calls, or doesn't like to run notebook cells! 
- :earth_africa: **African NMT**. [@jaderabbit](https://github.com/jaderabbit) started an initiative at the Indaba Deep Learning School 2019 to ["put African NMT on the map"](https://twitter.com/alienelf/status/1168159616167010305). The goal is to build and collect NMT models for low-resource African languages. The [Masakhane repository](https://github.com/masakhane-io/masakhane-mt) contains and explains all the code you need to train JoeyNMT and points to data sources. It also contains benchmark models and configurations that members of Masakhane have built for various African languages. Furthermore, you might be interested in joining the [Masakhane community](https://github.com/masakhane-io/masakhane-community) if you're generally interested in low-resource NLP/NMT. Also see the [EMNLP Findings paper](https://arxiv.org/abs/2010.02353).
- :speech_balloon: **Slack Joey**. [Code](https://github.com/juliakreutzer/slack-joey) to locally deploy a JoeyNMT model as chat bot in a Slack workspace. It's a convenient way to probe your model without having to implement an API. And bad translations for chat messages can be very entertaining, too ;)
- :globe_with_meridians: **Flask Joey**. [@kevindegila](https://github.com/kevindegila) built a [flask interface to Joey](https://github.com/kevindegila/flask-joey), so you can deploy your trained model in a web app and query it in the browser. 
- :busts_in_silhouette: **User Study**. We evaluated the code quality of this repository by testing the understanding of novices through quiz questions. Find the details in Section 3 of the [JoeyNMT paper](https://arxiv.org/abs/1907.12484).
- :pencil: **Self-Regulated Interactive Seq2Seq Learning**. Julia Kreutzer and Stefan Riezler. Published at ACL 2019. [Paper](https://arxiv.org/abs/1907.05190) and [Code](https://github.com/juliakreutzer/joeynmt/tree/acl19). This project augments the standard fully-supervised learning regime by weak and self-supervision for a better trade-off of quality and supervision costs in interactive NMT.
- :camel: **Hieroglyph Translation**. JoeyNMT was used to translate hieroglyphs in [this IWSLT 2019 paper](https://www.cl.uni-heidelberg.de/statnlpgroup/publications/IWSLT2019.pdf) by Philipp Wiesenbach and Stefan Riezler. They gave JoeyNMT multi-tasking abilities. 

If you used JoeyNMT for a project, publication or built some code on top of it, let us know and we'll link it here.


## Contact
Please leave an issue if you have questions or issues with the code.

For general questions, email us at `joeynmt <at> gmail.com`. :love_letter:

## Reference
If you use JoeyNMT in a publication or thesis, please cite the following [paper](https://arxiv.org/abs/1907.12484):

```
@inproceedings{kreutzer-etal-2019-joey,
    title = "Joey {NMT}: A Minimalist {NMT} Toolkit for Novices",
    author = "Kreutzer, Julia  and
      Bastings, Jasmijn  and
      Riezler, Stefan",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP): System Demonstrations",
    month = nov,
    year = "2019",
    address = "Hong Kong, China",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/D19-3019",
    doi = "10.18653/v1/D19-3019",
    pages = "109--114",
}
```

## Naming
Joeys are [infant marsupials](https://en.wikipedia.org/wiki/Marsupial#Early_development). :koala:

