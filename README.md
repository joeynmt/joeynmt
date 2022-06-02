# &nbsp; ![Joey-NMT](joey2-small.png) Joey NMT 2.0
[![build](https://github.com/may-/joeynmt/actions/workflows/main.yml/badge.svg)](https://github.com/may-/joeynmt/actions/workflows/main.yml)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

## Goal and Purpose
:koala: Joey NMT framework is developed for educational purposes.
It aims to be a **clean** and **minimalistic** code base to help novices 
find fast answers to the following questions.
- :grey_question: How to implement classic NMT architectures (RNN and Transformer) in PyTorch?
- :grey_question: What are the building blocks of these architectures and how do they interact?
- :grey_question: How to modify these blocks (e.g. deeper, wider, ...)?
- :grey_question: How to modify the training procedure (e.g. add a regularizer)?

In contrast to other NMT frameworks, we will **not** aim for the most recent features
or speed through engineering or training tricks since this often goes in hand with an
increase in code complexity and a decrease in readability. :eyes:

However, Joey NMT re-implements baselines from major publications.

Check out the detailed [documentation](https://joeynmt.readthedocs.io) and our
[paper](https://arxiv.org/abs/1907.12484).

## Contributors
Joey NMT was initially developed and is maintained by [Jasmijn Bastings](https://github.com/bastings) (University of Amsterdam) and [Julia Kreutzer](https://juliakreutzer.github.io/) (Heidelberg University), now both at Google Research. [Mayumi Ohta](https://www.cl.uni-heidelberg.de/statnlpgroup/members/ohta/) at Heidelberg University is continuing the legacy.

Welcome to our new contributors :hearts:, please don't hesitate to open a PR or an issue
if there's something that needs improvement!

## Features
Joey NMT implements the following features (aka the minimalist toolkit of NMT :wrench:):
- Recurrent Encoder-Decoder with GRUs or LSTMs
- Transformer Encoder-Decoder
- Attention Types: MLP, Dot, Multi-Head, Bilinear
- Word-, BPE- and character-based tokenization
- BLEU, ChrF evaluation
- Beam search with length penalty and greedy decoding
- Customizable initialization
- Attention visualization (for RNN)
- Learning curve plotting
- Scoring hypotheses and references



## Installation
Joey NMT is built on [PyTorch](https://pytorch.org/). Please make sure you have a compatible environment.
We tested Joey NMT 2.0 with
- python 3.9
- torch 1.11.0
- cuda 11.5

You can install Joey NMT either A. via [pip](https://pypi.org/project/joeynmt/) or B. from source.

### A. Via pip
for latest stable version:
```bash
$ pip install joeynmt
```
  
### B. From source
1. Clone this repository:
  ```bash
  $ git clone https://github.com/may-/joeynmt.git
  $ cd joeynmt
  ```
2. Install Joey NMT and it's requirements:
  ```bash
  $ pip install . -e
  ```
3. Run the unit tests:
  ```bash
  $ python -m unittest
  ```

> :warning: **Warning**
> When running on **GPU** you need to manually install the suitable PyTorch version 
> for your [CUDA](https://developer.nvidia.com/cuda-zone) version.
> For example, you can install PyTorch 1.11.0 with CUDA v11.3 as follows:
> ```
> $ pip install --upgrade torch==1.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
> ```
> This is described in the [PyTorch installation instructions](https://pytorch.org/get-started/locally/).
> You'll need this in particular when working on Google Colab (if it doesn't already
> have the right PyTorch version installed).

**[Optional]** For fp16 training, install NVIDIA's [apex](https://github.com/NVIDIA/apex) library:
```bash
$ git clone https://github.com/NVIDIA/apex
$ cd apex
$ pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

## Change logs
### v2.0 *Breaking change!*
- upgrade to python 3.9, torch 1.11
- `torchtext.legacy` dependencies are completely replaced by `torch.utils.data`
- `joeynmt/tokenizers.py`: handles tokenization internally (also supports bpe-dropout!)
- `joeynmt/datasets.py`: loads data from plaintext, tsv, and huggingface's [datasets](https://github.com/huggingface/datasets)
- `scripts/build_vocab.py`: trains subwords, creates joint vocab
- enhancement in decoding
  - scoring with hypotheses or references
  - repetition penalty, ngram blocker
  - attention plots for transformers
- yapf, isort, flake8 introduced
- bugfixes, minor refactoring

> :warning: **Warning**
> The models trained with Joey NMT v1.x can be decoded with Joey NMT v2.0.
> But there is no guarantee that you can reproduce the same score as before.

<details><summary>previous releases</summary>

### v1.4
- upgrade to sacrebleu 2.0, python 3.7, torch 1.8
- bugfixes

### v1.3
- upgrade to torchtext 0.9 (torchtext -> torchtext.legacy)
- n-best decoding
- demo colab notebook

### v1.0
- Multi-GPU support
- fp16 (half precision) support

</details>

## Documentation & Tutorials

We also updated the [documentation](https://joeynmt.readthedocs.io) thoroughly for Joey NMT 2.0!

For details, follow the tutorials in [notebooks](notebooks) dir.
- [quick-start-with-joeynmt2.0]() (will be released in June 2022)
- [how-to-extend-joeynmt]() (will be released in July 2022)
- [speech-translation]() (will be released in August 2022)


## Usage
> :warning: **Warning**
> For Joey NMT v1.x, please refer the archive [here](docs/JoeyNMT_v1.md).

Joey NMT has 3 modes: `train`, `test`, and `translate`, and all of them takes a
[YAML](https://yaml.org/)-style config file as argument.
You can find examples in the `configs` directory.
`small.yaml` contains a detailed explanation of configuration options.

Most importantly, the configuration contains the description of the model architecture
(e.g. number of hidden units in the encoder RNN), paths to the training, development and
test data, and the training hyperparameters (learning rate, validation frequency etc.).

> :memo: **Info**
> Note that subword model training and joint vocabulary creation is not included
> in the 3 modes above, has to be done separately.
> We provide a script that takes care of it: `scritps/build_vocab.py`.
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
specified in the `data`, `training` and `model` section of the config file (here
`configs/small.yaml`).

```
model_dir/
├── *.ckpt          # checkpoints
├── *.hyps          # translated texts at validation
├── config.yaml     # config file
├── spm.model       # sentencepiece model / subword-nmt codes file
├── src_vocab.txt   # src vocab
├── trg_vocab.txt   # trg vocab
├── train.log       # train log
└── validation.txt  # validation scores
```

> :bulb: **Tip**
> Be careful not to overwrite `model_dir`, set `overwrite: False` in the config file.



### `test` mode
This mode will generate translations for validation and test set (as specified in the
configuration) in `model_dir/out.[dev|test]`.
```
$ python -m joeynmt test configs/small.yaml --ckpt model_dir/avg.ckpt
```
If `--ckpt` is not specified above, the checkpoint path in `load_model` of the config
file or the best model in `model_dir` will be used to generate translations.

You can specify i.e. [sacrebleu](https://github.com/mjpost/sacrebleu) options in the
`test` section of the config file.

> :bulb: **Tip**
> `scripts/average_checkpoints.py` will generate averaged checkpoints for you.
> ```
> $ python scripts/average_checkpoints.py configs/small.yaml --joint
> ```

If you want to output the log-probabilities of the hypotheses or references, you can
specify `return_score: 'hyp'` or `return_score: 'ref'` in the testing section of the
config. And run `test` with `--output_path` and `--save_scores` options.
```
$ python -m joeynmt test configs/small.yaml --ckpt model_dir/avg.ckpt --output_path model_dir/pred --save_scores
```
This will generate `model_dir/pred.{dev|test}.{scores|tokens}` which contains scores and corresponding tokens.

> :memo: **Info**
> - If you set `return_score: 'hyp'` with greedy decoding, then token-wise scores will be returned. The beam search will return sequence-level scores, because the scores are summed up per sequence during beam exploration.
> - If you set `return_score: 'ref'`, the model looks up the probabilities of the given ground truth tokens, and both decoding and evaluation will be skipped.
> - If you specify `n_best` >1 in config, the first translation in the nbest list will be used in the evaluation.



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
  You'll be prompted to type an input sentence. Joey NMT will then translate with the 
  model specified in `--ckpt` or the config file.

  > :bulb: **Tip**
  > Interactive `translate` mode doesn't work with Multi-GPU.
  > Please run it on single GPU or CPU.



## Benchmarks & pretrained models

> :warning: **Warning**
> These models are trained with JoeynNMT v1.x, and decoded with Joey NMT v2.0. 
> See `config_v1.yaml` and `config_v2.yaml` in the linked zip, respectively.
> Joey NMT v1.x benchmarks are archived [here](docs/benchmarks_v1.md).

### iwslt14 deen

Pre-processing with Moses decoder tools as in [this script](scripts/get_iwslt14_bpe.sh).

Direction | Architecture | tok | dev | test | #params | download
--------- | :----------: | :-- | --: | ---: | ------: | :-------
de->en | RNN | subword-nmt | 31.77 | 30.74 | 61M | [rnn_iwslt14_deen_bpe.tar.gz](https://cl.uni-heidelberg.de/statnlpgroup/joeynmt2/rnn_iwslt14_deen_bpe.tar.gz) (672MB)
de->en | Transformer | subword-nmt | 34.53 | 33.73 | 19M | [transformer_iwslt14_deen_bpe.tar.gz](https://cl.uni-heidelberg.de/statnlpgroup/joeynmt2/transformer_iwslt14_deen_bpe.tar.gz) (221MB)

sacrebleu signature: `nrefs:1|case:lc|eff:no|tok:13a|smooth:exp|version:2.0.0`

> :memo: **Info**
> For interactive translate mode, you should specify `pretokenizer: "moses"` in the both src's and trg's `tokenizer_cfg`,
> so that you can input raw sentence. Then `MosesTokenizer` and `MosesDetokenizer` will be applied internally.
> For test mode, we used the preprocessed texts as input and set `pretokenizer: "none"` in the config.


### Masakhane JW300 afen / enaf

We picked the pretrained models and configs (bpe codes file etc.) from [masakhane.io](https://github.com/masakhane-io/masakhane-mt).

Direction | Architecture | tok | dev | test | #params | download
--------- | :----------: | :-- | --: | ---: | ------: | :-------
af->en | Transformer | subword-nmt | - | 57.70 | 46M | [transformer_jw300_afen.tar.gz](https://cl.uni-heidelberg.de/statnlpgroup/joeynmt2/transformer_jw300_afen.tar.gz) (525MB)
en->af | Transformer | subword-nmt | 47.24 | 47.31 | 24M | [transformer_jw300_enaf.tar.gz](https://cl.uni-heidelberg.de/statnlpgroup/joeynmt2/transformer_jw300_enaf.tar.gz) (285MB)

sacrebleu signature: `nrefs:1|case:mixed|eff:no|tok:intl|smooth:exp|version:2.0.0`


### JParaCrawl enja

For training, we split JparaCrawl v2 into train and dev set and trained a model on them.
Please check the preprocessing script [here](scripts/get_jparacrawl.sh).
We tested then on [kftt](http://www.phontron.com/kftt/) test set and [wmt20]() test set, respectively. 

Direction | Architecture | tok | wmt20 | kftt | #params | download
--------- | ------------ | :-- | ---: | ------: | ------: | :-------
en->ja | Transformer | sentencepiece | 17.66 | 14.31 | 225M | [jparacrawl_enja.tar.gz](https://cl.uni-heidelberg.de/statnlpgroup/joeynmt2/jparacrawl_enja.tar.gz) (2.3GB)
ja->en | Transformer | sentencepiece | 14.97 | 11.49 | 221M | [jparacrawl_jaen.tar.gz](https://cl.uni-heidelberg.de/statnlpgroup/joeynmt2/jparacrawl_jaen.tar.gz) (2.2GB)

sacrebleu signature: 
- en->ja `nrefs:1|case:mixed|eff:no|tok:ja-mecab-0.996-IPA|smooth:exp|version:2.0.0`
- ja->en `nrefs:1|case:mixed|eff:no|tok:intl|smooth:exp|version:2.0.0`

* In wmt20 test set, `newstest2020-enja` has 1000 examples, `newstest2020-jaen` has 993 examples.

## Coding
In order to keep the code clean and readable, we make use of:
- Style checks:
  - [pylint](https://pylint.pycqa.org/) with (mostly) PEP8 conventions, see `.pylintrc`.
  - [yapf](https://github.com/google/yapf), [isort](https://github.com/PyCQA/isort),
    and [flake8](https://flake8.pycqa.org/); see `.style.yapf`, `setup.cfg` and `Makefile`.
- Typing: Every function has documented input types.
- Docstrings: Every function, class and module has docstrings describing their purpose and usage.
- Unittests: Every module has unit tests, defined in `test/unit/`.

To ensure the repository stays clean, unittests and linters are triggered by github's
workflow on every push or pull request to `main` branch. Before you create a pull request,
you can check the validity of your modifications with the following commands:
```
$ make check
$ make test
```

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
main branch, but we're curious to learn what you used Joey NMT for.


## Projects and Extensions
Here we'll collect projects and repositories that are based on Joey NMT, so you can find
inspiration and examples on how to modify and extend the code.

### Joey NMT v1.x
- :spider_web: **Masakhane Web**. [@CateGitau](https://github.com/categitau), [@Kabongosalomon](https://github.com/Kabongosalomon), [@vukosim](https://github.com/vukosim) and team built a whole web translation platform for the African NMT models that Masakhane built with Joey NMT. The best is: it's completely open-source, so anyone can contribute new models or features. Try it out [here](http://translate.masakhane.io/), and check out the [code](https://github.com/dsfsi/masakhane-web).
- :gear: **MutNMT**. [@sjarmero](https://github.com/sjarmero) created a web application to train NMT: it lets the user train, inspect, evaluate and translate with Joey NMT --- perfect for NMT newbies! Code [here](https://github.com/Prompsit/mutnmt). The tool was developed by [Prompsit](https://www.prompsit.com/) in the framework of the European project [MultiTraiNMT](http://www.multitrainmt.eu/).
- :star2: **Cantonese-Mandarin Translator**. [@evelynkyl](https://github.com/evelynkyl/) trained different NMT models for translating between the low-resourced Cantonese and Mandarin,  with the help of some cool parallel sentence mining tricks! Check out her work [here](https://github.com/evelynkyl/yue_nmt).
- :book: **Russian-Belarusian Translator**. [@tsimafeip](https://github.com/tsimafeip) built a translator from Russian to Belarusian and adapted it to legal and medical domains. The code can be found [here](https://github.com/tsimafeip/Translator/).
- :muscle: **Reinforcement Learning**. [@samuki](https://github.com/samuki/) implemented various policy gradient variants in Joey NMT: here's the [code](https://github.com/samuki/reinforce-joey), could the logo be any more perfect? :muscle: :koala:
- :hand: **Sign Language Translation**. [@neccam](https://github.com/neccam/) built a sign language translator that continuosly recognizes sign language and translates it. Check out the [code](https://github.com/neccam/slt) and the [CVPR 2020 paper](https://openaccess.thecvf.com/content_CVPR_2020/html/Camgoz_Sign_Language_Transformers_Joint_End-to-End_Sign_Language_Recognition_and_Translation_CVPR_2020_paper.html)!
- :abc: [@bpopeters](https://github.com/bpopeters/) built [Possum-NMT](https://github.com/deep-spin/sigmorphon-seq2seq) for multilingual grapheme-to-phoneme transduction and morphologic inflection. Read their [paper](https://www.aclweb.org/anthology/2020.sigmorphon-1.4.pdf) for SIGMORPHON 2020!
- :camera: **Image Captioning**. [@pperle](https://github.com/pperle) and [@stdhd](https://github.com/stdhd) built an image captioning tool on top of Joey NMT, check out the [code](https://github.com/stdhd/image_captioning) and the [demo](https://image2caption.pascalperle.de/)!
- :bulb: **Joey Toy Models**. [@bricksdont](https://github.com/bricksdont) built a [collection of scripts](https://github.com/bricksdont/joeynmt-toy-models) showing how to install Joey NMT, preprocess data, train and evaluate models. This is a great starting point for anyone who wants to run systematic experiments, tends to forget python calls, or doesn't like to run notebook cells! 
- :earth_africa: **African NMT**. [@jaderabbit](https://github.com/jaderabbit) started an initiative at the Indaba Deep Learning School 2019 to ["put African NMT on the map"](https://twitter.com/alienelf/status/1168159616167010305). The goal is to build and collect NMT models for low-resource African languages. The [Masakhane repository](https://github.com/masakhane-io/masakhane-mt) contains and explains all the code you need to train Joey NMT and points to data sources. It also contains benchmark models and configurations that members of Masakhane have built for various African languages. Furthermore, you might be interested in joining the [Masakhane community](https://github.com/masakhane-io/masakhane-community) if you're generally interested in low-resource NLP/NMT. Also see the [EMNLP Findings paper](https://arxiv.org/abs/2010.02353).
- :speech_balloon: **Slack Joey**. [Code](https://github.com/juliakreutzer/slack-joey) to locally deploy a Joey NMT model as chat bot in a Slack workspace. It's a convenient way to probe your model without having to implement an API. And bad translations for chat messages can be very entertaining, too ;)
- :globe_with_meridians: **Flask Joey**. [@kevindegila](https://github.com/kevindegila) built a [flask interface to Joey](https://github.com/kevindegila/flask-joey), so you can deploy your trained model in a web app and query it in the browser. 
- :busts_in_silhouette: **User Study**. We evaluated the code quality of this repository by testing the understanding of novices through quiz questions. Find the details in Section 3 of the [Joey NMT paper](https://arxiv.org/abs/1907.12484).
- :pencil: **Self-Regulated Interactive Seq2Seq Learning**. Julia Kreutzer and Stefan Riezler. Published at ACL 2019. [Paper](https://arxiv.org/abs/1907.05190) and [Code](https://github.com/juliakreutzer/joeynmt/tree/acl19). This project augments the standard fully-supervised learning regime by weak and self-supervision for a better trade-off of quality and supervision costs in interactive NMT.
- :camel: **Hieroglyph Translation**. Joey NMT was used to translate hieroglyphs in [this IWSLT 2019 paper](https://www.cl.uni-heidelberg.de/statnlpgroup/publications/IWSLT2019.pdf) by Philipp Wiesenbach and Stefan Riezler. They gave Joey NMT multi-tasking abilities. 

If you used Joey NMT for a project, publication or built some code on top of it, let us know and we'll link it here.


## Contact
Please leave an issue if you have questions or issues with the code.

For general questions, email us at `joeynmt <at> gmail.com`. :love_letter:


## Reference
If you use Joey NMT in a publication or thesis, please cite the following [paper](https://arxiv.org/abs/1907.12484):

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

