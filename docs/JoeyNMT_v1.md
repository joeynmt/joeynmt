## Usage

For details, follow the tutorial in [the docs](https://joeynmt.readthedocs.io). :book:

### Data Preparation

#### Parallel Data
For training a translation model, you need parallel data, i.e. a collection of source sentences and reference translations that are aligned sentence-by-sentence and stored in two files, 
such that each line in the reference file is the translation of the same line in the source file.

#### Pre-processing
Before training a model on it, parallel data is most commonly filtered by length ratio, tokenized and true- or lowercased.

The Moses toolkit provides a set of useful [scripts](https://github.com/moses-smt/mosesdecoder/tree/master/scripts) for this purpose.

In addition, you might want to build the NMT model not on the basis of words, but rather sub-words or characters (the `level` in JoeyNMT configurations).
Currently, JoeyNMT supports the byte-pair-encodings (BPE) format by [subword-nmt](https://github.com/rsennrich/subword-nmt) and [sentencepiece](https://github.com/google/sentencepiece).

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
- [The docs](https://joeynmt.readthedocs.io) include an overview of the NMT implementation, a walk-through tutorial for building, training, tuning, testing and inspecting an NMT system, the [API documentation](https://joeynmt.readthedocs.io/en/latest/api.html) and [FAQs](https://joeynmt.readthedocs.io/en/latest/faq.html).
- A screencast of the tutorial is available on [YouTube](https://www.youtube.com/watch?v=PzWRWSIwSYc). :movie_camera:
- Jade Abbott wrote a [notebook](https://github.com/masakhane-io/masakhane-mt/blob/master/starter_notebook-custom-data.ipynb) that runs on Colab that shows how to prepare data, train and evaluate a model, at the example of low-resource African languages.
- Matthias Müller wrote a [collection of scripts](https://github.com/bricksdont/joeynmt-toy-models) for installation, data download and preparation, model training and evaluation.

## Benchmarks
Benchmark results on WMT and IWSLT datasets are reported [here](benchmarks.md). Please also check the [Masakhane MT repository](https://github.com/masakhane-io/masakhane-mt) for benchmarks and available models for African languages.

## Pre-trained Models
Pre-trained models from reported benchmarks for download (contains config, vocabularies, best checkpoint and dev/test hypotheses):

### IWSLT14 de-en
Pre-processing with Moses decoder tools as in [this script](https://github.com/joeynmt/joeynmt/blob/main/scripts/get_iwslt14_bpe.sh).

- [IWSLT14 de-en BPE RNN](https://www.cl.uni-heidelberg.de/statnlpgroup/joeynmt/iwslt14-deen-bpe.tar.gz) (641M)
- [IWSLT14 de-en Transformer](https://www.cl.uni-heidelberg.de/statnlpgroup/joeynmt/transformer_iwslt14_deen_bpe.tar.gz) (210M)

### IWSLT15 en-vi
The data came preprocessed from Stanford NLP, see [this script](https://github.com/joeynmt/joeynmt/blob/main/scripts/get_iwslt15_envi.sh).

- [IWSLT15 en-vi Transformer](https://www.cl.uni-heidelberg.de/statnlpgroup/joeynmt/transformer_iwslt15_envi.tar.gz) (186M)

### WMT17
Following the pre-processing of the [Sockeye paper](https://arxiv.org/abs/1712.05690).

- [WMT17 en-de "best" RNN](https://www.cl.uni-heidelberg.de/statnlpgroup/joeynmt/wmt_ende_best.tar.gz) (2G)
- [WMT17 lv-en "best" RNN](https://www.cl.uni-heidelberg.de/statnlpgroup/joeynmt/wmt_lven_best.tar.gz) (1.9G)
- [WMT17 en-de Transformer](https://www.cl.uni-heidelberg.de/statnlpgroup/joeynmt/wmt_ende_transformer.tar.gz) (664M)
- [WMT17 lv-en Transformer](https://www.cl.uni-heidelberg.de/statnlpgroup/joeynmt/wmt_lven_transformer.tar.gz) (650M)

### Autshumato
Training with data provided in the [Ukuxhumana project](https://github.com/LauraMartinus/ukuxhumana), with additional tokenization of the training data with the [Moses tokenizer](https://github.com/moses-smt/mosesdecoder/blob/master/scripts/tokenizer/tokenizer.perl).

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
