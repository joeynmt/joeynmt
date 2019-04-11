# &nbsp; ![Joey-NMT](joey-small.png) Joey NMT
[![Build Status](https://travis-ci.com/joeynmt/joeynmt.svg?branch=master)](https://travis-ci.org/joeynmt/joeynmt)


## Goal and Purpose
Joey NMT framework is developed for educational purposes. 
It aims to be a **clean** and **minimalistic** code base to help novices 
pursuing the understanding of the following questions.
- How to implement classic NMT architectures (RNN and Transformer) in PyTorch?
- What are the building blocks of these architectures and how do they interact?
- How to modify these blocks (e.g. deeper, wider, ...)?
- How to modify the training procedure (e.g. add a regularizer)?

In contrast to other NMT frameworks, we will **not** aim for 
state-of-the-art results or speed through engineering or training tricks
since this often goes in hand with an increase in code complexity 
and a decrease in readability.

However, Joey NMT re-implements baselines from major publications.

## Contributors

Joey NMT is developed by [Joost Bastings](https://bastings.github.io) (University of Amsterdam) and [Julia Kreutzer](http://www.cl.uni-heidelberg.de/~kreutzer/) (Heidelberg University).

## Features
We aim to implement the following features (aka the minimalist toolkit of NMT):
- Recurrent Encoder-Decoder with GRUs or LSTMs
- Transformer Encoder-Decoder
- Attention Types: MLP, Dot, Multi-Head, Bilinear
- Word-, BPE- and character-based input handling
- BLEU, ChrF evaluation
- Beam search with length penalty and greedy decoding
- Customizable initialization
- Attention visualization
- Learning curve plotting

[Work in progress: Transformer, Multi-Head and Dot still missing.]

## Coding
In order to keep the code clean and readable, we make use of:
- Style checks: pylint with (mostly) PEP8 conventions, see `.pylintrc`.
- Typing: Every function has documented input types.
- Docstrings: Every function, class and module has docstrings describing their purpose and usage.
- Unittests: Every module has unit tests, defined in `test/unit/`.
Travis CI runs the tests and pylint on every push to ensure the repository stays clean.


## Teaching
We will create dedicated material for teaching with Joey NMT. This will include:
- An overview and explanation of the code architecture.
- A tutorial how to train and test a baseline model.
- A walk-through example of how to implement a modification of a baseline model.

[Work in progress!]

## Installation
Joey NMT is built on [PyTorch](https://pytorch.org/) v.0.4.1 and [torchtext](https://github.com/pytorch/text) for Python >= 3.6.

1. Clone this repository:
`git clone https://github.com/joeynmt/joeynmt.git`
2. Install the requirements:
`cd joeynmt`
`pip3 install -r requirements.txt` (you might want to add `--user` for a local installation).
3. Install joeynmt:
`python3 setup.py install`
4. Run the unit tests:
`python3 -m unittest`


## Usage

### Data Preparation

#### Parallel Data
For training a translation model, you need parallel data, i.e. a collection of source sentences and reference translations that are aligned sentence-by-sentence and stored in two files, 
such that each line in the reference file is the translation of the same line in the source file.

The shared tasks of the yearly [Conference on Machine Translation (WMT)](http://www.statmt.org/wmt19/) provide lots of parallel data.

#### Pre-processing
Before training a model on it, parallel data is most commonly filtered by length ratio, tokenized and true- or lowercased.

The Moses toolkit provides a set of useful [scripts](https://github.com/moses-smt/mosesdecoder/tree/master/scripts) for this purpose.

In addition, you might want to build the NMT model not on the basis of words, but rather sub-words or characters (the `level` in JoeyNMT configurations).
Currently, JoeyNMT supports the byte-pair-encodings (BPE) format by [subword-nmt](https://github.com/rsennrich/subword-nmt).

### Configuration
Experiments are specified in configuration files, in simple [YAML](http://yaml.org/) format. You can find examples in the `configs` directory.
`default.yaml` contains a detailed explanation of configuration options.

Most importantly, the configuration contains the description of the model architecture (e.g. number of hidden units in the encoder RNN), 
paths to the training, development and test data, and the training hyperparameters (learning rate, validation frequency etc.).

### Training

#### Start
For training, run 

`python3 -m joeynmt train configs/default.yaml`. 

This will train a model on the training data specified in the config (here: `default.yaml`), 
validate on validation data, 
and store model parameters, vocabularies, validation outputs and a small number of attention plots in the `model_dir` (also specified in config).

Note that pre-processing like tokenization or BPE-ing is not included in training, but has to be done manually before.

Tip: Be careful not to overwrite models, set `overwrite: False` in the model configuration.

#### Validations
The `validations.txt` file in the model directory reports the validation results at every validation point. 
Models are saved whenever a new best validation score is reached, in `batch_no.ckpt`, where `batch_no` is the number of batches the model has been trained on so far.
`best.ckpt` links to the checkpoint that has so far achieved the best validation score.


#### Visualization
JoeyNMT uses [TensorboardX](https://github.com/lanpa/tensorboardX) to visualize training and validation curves and attention matrices during training.
Launch [Tensorboard](https://github.com/tensorflow/tensorboard) with `tensorboard --logdir model_dir/tensorboard` (or `python -m tensorboard.main ...`) and then open the url (default: `localhost:6006`) with a browser. 

For a stand-alone plot, run `python3 scripts/plot_validation.py model_dir --plot_values bleu PPL --output_path my_plot.pdf` to plot curves of validation BLEU and PPL.

#### CPU vs. GPU
For training on a GPU, set `use_cuda` in the config file to `True`. This requires the installation of required CUDA libraries.


### Translating

There's 3 options for testing what the model has learned.

Whatever data you feed the model for translating, make sure it is properly pre-processed, just as you pre-processed the training data, e.g. tokenized and split into subwords (if working with BPEs).

#### 1. Test Set Evaluation 
For testing and evaluating on your parallel test/dev set, run 

`python3 -m joeynmt test configs/default.yaml --output_path out`.

This will generate translations for validation and test set (as specified in the configuration) in `out.[dev|test]`
with the latest/best model in the `model_dir` (or a specific checkpoint set with `load_model`).
It will also evaluate the outputs with `eval_metric`.
If `--output_path` is not specified, it will not store the translation, and only do the evaluation and print the results.

#### 2. File Translation
In order to translate the contents of a file not contained in the configuration (here `my_input.txt`), simply run

`python3 -m joeynmt translate configs/default.yaml < my_input.txt > out`.

The translations will be written to stdout or alternatively`--output_path` if specified.

#### 3. Interactive
If you just want try a few examples, run

`python3 -m joeynmt translate configs/default.yaml`

and you'll be prompted to type input sentences that JoeyNMT will then translate with the model specified in the configuration.



## API Documentation
Read [the docs](https://joeynmt.readthedocs.io).

## Benchmarks
Benchmarks on small models trained on GPU/CPU on standard data sets will be 
posted here.

- IWSLT15 En-Vi, word-based
- IWSLT14 De-En, 32000 joint BPE, word-based
- WMT17 En-De and Lv-En, 32000 joint BPE

### IWSLT English-Vietnamese

We compare against [Tensorflow NMT](https://github.com/tensorflow/nmt) on the IWSLT15 En-Vi data set as preprocessed by Stanford.
You can download the data with `scripts/get_iwslt15_envi.sh`, and then use `configs/iwslt_envi_luong.yaml` to replicate the experiment.

Systems | tst2012 (dev) | test2013 (test)
--- | :---: | :---:
TF NMT (greedy)    | 23.2 | 25.5
TF NMT (beam=10)   | 23.8 | 26.1
Joey NMT (greedy)  | 23.2 | 25.8 
Joey NMT (beam=10, alpha=1.0) | 23.8 | 26.5
[(Luong & Manning, 2015)](https://nlp.stanford.edu/pubs/luong-manning-iwslt15.pdf) | - | 23.3

We also compare against [xnmt](https://github.com/neulab/xnmt/tree/master/recipes/stanford-iwslt) which uses different hyperparameters, so we use a different configuration for Joey NMT too: `configs/iwslt_envi_xnmt.yaml`.

Systems | tst2012 (dev) | test2013 (test)
--- | :---: | :---:
xnmt (beam=5)                | 25.0 | 27.3
Joey NMT (greedy)            | 24.6 | 27.4
Joey NMT (beam=5, alpha=1.0) | 24.9 | 27.7


### IWSLT  German-English
We compare against the baseline scores reported in [(Wiseman & Rush, 2016)](https://arxiv.org/pdf/1606.02960.pdf) (W&R), 
[(Bahdanau et al., 2017)](https://arxiv.org/pdf/1607.07086.pdf) (B17) with tokenized, lowercased BLEU (using `sacrebleu`).
Ẁe compare a word-based model of the same size and vocabulary as in W&R and B17.
The [script](https://github.com/harvardnlp/BSO/blob/master/data_prep/MT/prepareData.sh) to obtain and pre-process the data is the one published with W&R.
Use `configs/iwslt_deen_bahdanau.yaml` for training the model.
On a K40-GPU word-level training took <1h, beam search decoding for both dev and test <2min.

Systems | level | dev | test | #params 
--- | :---: | :---: | :---: | :---: 
W&R (greedy)   | word | - | 22.53  |  
W&R (beam=10)  | word | - | 23.87  |
B17 (greedy)   | word | -| 25.82  |
B17 (beam=10)  | word | -| 27.56  | 
Joey NMT (greedy) | word | 28.41 | 26.68 | 22.05M 
Joey NMT (beam=10, alpha=1.0) | word | 28.96 | 27.03| 22.05M 

On CPU (`use_cuda: False`): 
(approx 8-10x slower: 8h for training, beam search decoding for both dev and test 19min, greedy decoding 5min)

Systems | level | dev | test | #params 
--- | :---: | :---: | :---: | :---: 
Joey NMT (greedy) | word | 28.35 | 26.46 | 22.05M 
Joey NMT (beam=10, alpha=1.0) | word | 28.85 | 27.06 | 22.05M  

In addition, we compare to a BPE-based GRU model with 32k (Groundhog style). 
Use `scripts/get_iwslt14_bpe.sh` to pre-process the data and `configs/iwslt14_deen_bpe.yaml` to train the model.
This model is available for download [here](https://www.cl.uni-heidelberg.de/~kreutzer/joeynmt/models/iwslt14-deen-bpe/).

Systems | level | dev | test | #params 
--- | :---: | :---: | :---: | :---: 
Joey NMT (greedy) | bpe | 27.57 | | 60.69M 
Joey NMT (beam=5, alpha=1.0) | bpe | 28.55 | 27.34 | 60.69M 

## WMT 17 English-German and Latvian-English
We compare against the results for recurrent BPE-based models that were reported in the [Sockeye paper](https://arxiv.org/pdf/1712.05690.pdf). 
We only consider the ``Groundhog`` setting here, where toolkits are used out-of-the-box for creating a Groundhog-like model (1 layer, LSTMs, MLP attention).
The data is pre-processed as described in the paper ([code](https://github.com/awslabs/sockeye/tree/arxiv_1217/arxiv/code)).
Postprocessing is done with [Moses' detokenizer](https://github.com/moses-smt/mosesdecoder/blob/master/scripts/tokenizer/detokenizer.perl), evaluation with `sacrebleu`.

Note that the scores reported for other models might not reflect the current state of the code, but the state at the time of the Sockeye evaluation.
Please also consider the difference in number of parameters despite "the same" setup: our models are the smallest in numbers of parameters.

### English-German
Groundhog setting: `configs/wmt_ende_default.yaml`  with `encoder rnn=500`, `lr=0.0003`, `init_hidden="bridge"`.

Systems | level | dev | test | #params 
--- | :---: | :---: | :---: | :---: | 
Sockeye (beam=5) | bpe | - | 23.18 | 87.83M 
OpenNMT-Py (beam=5) | bpe | - | 18.66 | 87.62M 
Joey NMT (beam=5) | bpe | 24.33 | 23.45  | 86.37M  

The Joey NMT model was trained for 4 days (14 epochs).

### Latvian-English
Groundhog setting: `configs/wmt_lven_default.yaml` with `encoder rnn=500`, `lr=0.0003`, `init_hidden="bridge"`.

Systems | level | dev | test | #params 
--- | :---: | :---: | :---: | :---: 
Sockeye (beam=5) | bpe | - | 14.40 | ? 
OpenNMT-Py (beam=5) | bpe | - | 9.98 | ? 
Joey NMT (beam=5) | bpe | 12.09 | 8.75 | 64.52M 


## Contributing
Since this codebase is supposed to stay clean and minimalistic, contributions addressing the following are welcome:
- Code correctness
- Code cleanliness
- Documentation quality
- Speed or memory improvements
- Code addressing issues

Code extending the functionalities beyond the basics will most likely not end up in the master branch, but we're curions to learn what you used Joey for.

## Use-cases and Projects
Here we'll collect projects and repositories that are based on Joey. If you used Joey for a project, publication or built some code on top of it, let us know and we'll link it here.

Projects:
- TBD

## Contact
Please leave an issue if you have questions or issues with the code.

For general questions, email us at `joeynmt <at> gmail.com`.


## Naming
Joeys are [infant marsupials](https://en.wikipedia.org/wiki/Marsupial#Early_development). 

