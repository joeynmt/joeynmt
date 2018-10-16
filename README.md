# &nbsp; ![Joey-NMT](joey-small.png) joey-nmt

Joey NMT


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

The goal is to match the basic Groundhog baseline on WMT17 En-De 
that is reported in the [Sockeye](https://arxiv.org/abs/1712.05690) paper.
Data download and pre-processing scripts are [available](https://github.com/awslabs/sockeye/tree/arxiv_1217/arxiv/code)


## Features
We aim to implement the following features (aka the minimalist toolkit of NMT):
- Recurrent Encoder-Decoder with GRUs or LSTMs
- Transformer Encoder-Decoder
- Attention Types: MLP, Dot, Multi-Head, Bilinear
- Word-, BPE- and character-based input handling
- BLEU, ChrF evaluation
- Beam search with length penalty and greedy decoding
- Customizable initialization

[Work in progress: Transformer, Multi-Head and Dot still missing.]

## Coding
In order to keep the code clean and readable, we make use of:
- Style checks: Pylint, PEP8
- Typing
- Docstrings

[Work in progress!]


## Teaching
We will create dedicated material for teaching with Joey NMT. This will include:
- An overview and explanation of the code architecture.
- A tutorial how to train and test a baseline model.
- A walk-through example of how to implement a modification of a baseline model.

[Work in progress!]

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
On a XX GPU word-level training took <1h, beam search decoding for both dev and test <2min.

Systems | level | dev | test | #params | Joey NMT config
--- | :---: | :---: | :---: | :---: | :---:
W&R (greedy)   | word | | 22.53  |  
W&R (beam=10)  | word | | 23.87  |
B17 (greedy)   | word | | 25.82  |
B17 (beam=10)  | word | | 27.56  | 
Joey NMT (greedy) | word | 28.41 | 26.68 | 22049792 |
Joey NMT (beam=10, alpha=1.0) | word | 28.96 | 27.03| 22049792 | 

On CPU (`use_cuda: False`): 
(approx 8-10x slower: 8h for training, beam search decoding for both dev and test 19min, greedy decoding 5min)

Systems | level | dev | test | #params | Joey NMT config
--- | :---: | :---: | :---: | :---: | :---:
Joey NMT (greedy) | word | 28.35 | 26.46 | 22049792 |
Joey NMT (beam=10, alpha=1.0) | word | 28.85 | 27.06 | 22049792 | 

In addition, we compare to a BPE-based GRU model TODO describe data & training: bpe32k, GRU, size, bridge. Sockeye has additional output bias

Systems | level | dev | test | #params | Joey NMT config
--- | :---: | :---: | :---: | :---: | :---:  
Sockeye (greedy) | bpe | TODO test | | 60880750 |
Sockeye (beam=5) | bpe | TODO test| | 60880750 |
Joey NMT (greedy) | bpe | 27.8 | | 60677100 | 
Joey NMT (beam=5, alpha=1.0) | bpe | 28.74 | 27.63 | 60677100 |

## WMT 17 English-German and Latvian-English
We compare against the results for recurrent BPE-based models that were reported in the [Sockeye paper](https://arxiv.org/pdf/1712.05690.pdf). 
We only consider the ``Groundhog`` setting here, where toolkits are used out-of-the-box for creating a Groundhog-like model (1 layer, LSTMs, MLP attention).
The data is pre-processed as described in the paper ([code](https://github.com/awslabs/sockeye/tree/arxiv_1217/arxiv/code)).
Postprocessing is done with [Moses' detokenizer](https://github.com/moses-smt/mosesdecoder/blob/master/scripts/tokenizer/detokenizer.perl), evaluation with `sacrebleu`.

Note that the scores reported for other models might not reflect the current state of the code, but the state at the time of the Sockeye evaluation.
Our models are the smallest in numbers of parameters. 

### English-German
Groundhog setting: `encoder rnn=500`, `lr=0.0003`, `bridge=True`

Systems | level | dev | test | #params | Joey NMT config
--- | :---: | :---: | :---: | :---: | :---:  
Sockeye (beam=5) | bpe | - | 23.18 | 87.83M | 
OpenNMT-Py (beam=5) | bpe | - | 18.66 | 87.62M |
Joey NMT (beam=5) | bpe | 24.33 | 23.45  | 86.37M | `wmt_ende_default.yaml`  

The Joey NMT model was trained for 4 days (14 epochs).

### Latvian-English
Groundhog setting: `encoder rnn=500`, `lr=0.0003`, `bridge=True`

Systems | level | dev | test | #params | Joey NMT config
--- | :---: | :---: | :---: | :---: | :---:  
Sockeye (beam=5) | bpe | - | 14.40 | ? | 
OpenNMT-Py (beam=5) | bpe | - | 9.98 | ? | 
Joey NMT (beam=5) | bpe | 12.09 | 8.75 | 64.52M | `wmt_lven_default.yaml`  


## Contributing
Since this codebase is supposed to stay clean and minimalistic, contributions addressing the following are welcome:
- Code correctness
- Code cleanliness
- Documentation quality

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

