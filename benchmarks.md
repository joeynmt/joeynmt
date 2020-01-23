# Benchmarks

Here we provide benchmarks for reference. We provide all the scripts to replicate these results.
Note that some benchmarks were also reported in our paper: https://arxiv.org/abs/1907.12484

## WMT 17 English-German and Latvian-English
We compare against the results for recurrent BPE-based models that were reported in the [Sockeye paper](https://arxiv.org/pdf/1712.05690.pdf). 

We first consider the ``Groundhog`` setting, where toolkits are used out-of-the-box for creating a Groundhog-like model (1 layer, LSTMs, MLP attention).
The data is pre-processed as described in the Sockeye paper ([code](https://github.com/awslabs/sockeye/tree/arxiv_1217/arxiv/code)).

A shared vocabulary can be built with the [build_vocab.py script](scripts/build_vocab.py).
Postprocessing is done with [Moses' detokenizer](https://github.com/moses-smt/mosesdecoder/blob/master/scripts/tokenizer/detokenizer.perl), evaluation with `sacrebleu`.

Note that the scores reported for other models might not reflect the current state of the code, but the state at the time of the Sockeye evaluation.
Also note that our models are the smallest in numbers of parameters.

### English-German

#### Groundhog setting
Groundhog setting: `configs/wmt_ende_default.yaml`  with `encoder rnn=500`, `lr=0.0003`, `init_hidden="bridge"`.

Systems | level | dev | test | #params 
--- | :---: | :---: | :---: | :---: | 
Sockeye (beam=5) | bpe | - | 23.18 | 87.83M 
OpenNMT-Py (beam=5) | bpe | - | 18.66 | 87.62M 
Joey NMT (beam=5) | bpe | 24.33 | 23.45  | 86.37M  

The Joey NMT model was trained for 4 days (14 epochs).

#### "Best found" setting
"Best found" setting with a shared vocabulary: `configs/wmt_ende_best.yaml`  with `tied_embeddings=True`, `hidden_size=1024`, `lr=0.0002`, `init_hidden="bridge"` and `num_layers=4`.

Systems | level | dev | test | #params 
--- | :---: | :---: | :---: | :---: | 
Sockeye (beam=5) | bpe | - | 25.55 | ?
OpenNMT-Py (beam=5) | bpe | - | 21.95 | ? 
Joey NMT (beam=5) | bpe |  | 26.0  | 187M 

#### Transformer
Transformer setting: `configs/transformer_wmt17_ende.yaml` with 6 transformer layers of 512 units, 2048 hidden size, and 8 attention heads in en- and decoder.

Systems | level |  test | #params
--- | :---: | :---: | :---: 
Sockeye  | bpe | 27.5 | ?
Marian  | bpe | 27.4 | ?
Tensor2Tensor  | bpe | 26.3 | ?
Joey NMT (beam=5) | bpe | 27.4 | ?

### Latvian-English

#### Groundhog setting
Groundhog setting: `configs/wmt_lven_default.yaml` with `encoder rnn=500`, `lr=0.0003`, `init_hidden="bridge"`.

Systems | level | dev | test | #params 
--- | :---: | :---: | :---: | :---: 
Sockeye (beam=5) | bpe | - | 14.40 | ? 
OpenNMT-Py (beam=5) | bpe | - | 9.98 | ? 
Joey NMT (beam=5) | bpe | 12.09 | 8.75 | 64.52M 


#### "Best found" setting
"Best found" setting with a shared vocabulary: `configs/wmt_lven_best.yaml` with `tied_embeddings=True`, `hidden_size=1024`, `lr=0.0002`, `init_hidden="bridge"` and `num_layers=4`.

Systems | level | dev | test | #params 
--- | :---: | :---: | :---: | :---: 
Sockeye (beam=5) | bpe | - | 15.9 | ? 
OpenNMT-Py (beam=5) | bpe | - | 13.6 | ? 
Joey NMT (beam=5) | bpe | 20.12 | 15.8 | 182M

### Transformer
`configs/transformer_wmt17_lven.yaml` with 6 transformer layers of 512 units, 2048 hidden size, and 8 attention heads in en- and decoder.

Systems | level |  test 
--- | :---: | :---: 
Sockeye  | bpe | 18.1
Marian  | bpe | 217.6 
Tensor2Tensor  | bpe | 17.7
Joey NMT (beam=5) | bpe | 18.0

## IWSLT  German-English
We compare the RNN against the baseline scores reported in [(Wiseman & Rush, 2016)](https://arxiv.org/pdf/1606.02960.pdf) (W&R), 
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

We also evaluate using the Transformer. We use 256 hidden units, 4 attention heads, a feed-forward layer size of 1024, and dropout value of 0.3. You can find the settings in `configs/transformer_iwslt14_deen_bpe.yaml`.

Systems | level | dev | test | #params 
--- | :---: | :---: | :---: | :---: 
Joey NMT (greedy)                        | bpe | 27.57 |       | 60.69M 
Joey NMT (beam=5, alpha=1.0)             | bpe | 28.55 | 27.34 | 60.69M 
Joey NMT Transformer (greedy)            | bpe | 31.21 |       | 19.18M
Joey NMT Transformer (beam=5, alpha=1.0) | bpe | 32.16 | 31.00 | 19.18M



## IWSLT English-Vietnamese

We compare the RNN model against [Tensorflow NMT](https://github.com/tensorflow/nmt) on the IWSLT15 En-Vi data set as preprocessed by Stanford.
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

## Autshumato 

We built small Transformer models for the South-African languages (af: Afrikaans, nso: Norther Sotho, ts: Xitsonga, tn: Setswana, zu: isiZulu) of the [Autshumato benchmark](http://autshumato.sourceforge.net/) with the data as prepared in the [Uxhumana project](https://github.com/LauraMartinus/ukuxhumana). The training data from the "clean" subdirectory of the Uxhumana repository is additionally tokenized with Moses' tokenizer. 
The models are evaluated with sacrebleu on the tokenized test sets (beam=5, alpha=1.0, sacrebleu with `--tokenize none`). For comparison to the Uxhumana baselines ([Martinus & Abbott, 2019](https://arxiv.org/pdf/1906.10511.pdf)) built with the Tensor2Tensor base Transformer, we also compare to the results obtained with the international tokenizer.

System | Source language | Target language | **test (`--tokenize none`)** | test (`--tokenize intl`)
--- | :---: | :---: | :---: | :---:
Joey NMT | en | af | 23.5 | 24.48
Uxhumana Transformer | en | af | -- | 20.60
Joey NMT | af | en | 27.7 | 27.9
Joey NMT | en | nso | 14.0 | 16.97
Uxhumana Transformer | en | nso | -- | 10.94
Joey NMT | nso | en | 9.9 | 12.96
Joey NMT | en | tn | 15.1 | 19.51
Uxhumana Transformer | en | tn  | -- | 20.60
Joey NMT | tn | en | 12.8 | 17.41
Joey NMT | en | ts | 18.6 | 18.45
Uxhumana Transformer | en | ts  | -- | 17.98
Joey NMT | ts | en | 18.7 | 18.7
Joey NMT | en | zu | 1.4 | 3.64
Uxhumana Transformer | en | zu  | -- | 1.34
Joey NMT | zu | en | 6.9 | 8.74

