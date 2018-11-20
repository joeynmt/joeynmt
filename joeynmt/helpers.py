# coding: utf-8

import copy
import torch
from torch import nn
import numpy as np
import yaml
import glob
import os
from collections import Counter
import os.path

from torchtext.datasets import TranslationDataset
from torchtext import data

from joeynmt.constants import UNK_TOKEN, DEFAULT_UNK_ID, \
    EOS_TOKEN, BOS_TOKEN, PAD_TOKEN
from joeynmt.vocabulary import Vocabulary
from joeynmt.plotting import plot_heatmap


def log_cfg(cfg, logger, prefix="cfg"):
    """
    Write configuration to log.
    :param cfg:
    :param logger:
    :param prefix:
    :return:
    """
    for k, v in cfg.items():
        if isinstance(v, dict):
            p = '.'.join([prefix, k])
            log_cfg(v, logger, prefix=p)
        else:
            p = '.'.join([prefix, k])
            logger.info("{:34s} : {}".format(p, v))


def build_vocab(field, max_size, min_freq, data, vocab_file=None):
    """
    Builds vocabulary for a torchtext `field`

    :param field:
    :param max_size:
    :param min_freq:
    :param data:
    :param vocab_file:
    :return:
    """

    # special symbols
    specials = [UNK_TOKEN, PAD_TOKEN, BOS_TOKEN, EOS_TOKEN]

    if vocab_file is not None:
        # load it from file
        vocab = Vocabulary(file=vocab_file)
        vocab.add_tokens(specials)
    else:
        # create newly
        def filter_min(counter, min_freq):
            """ Filter counter by min frequency """
            filtered_counter = Counter({t: c for t, c in counter.items()
                                   if c >= min_freq})
            return filtered_counter

        def sort_and_cut(counter, limit):
            """ Cut counter to most frequent,
            sorted numerically and alphabetically"""
            # sort by frequency, then alphabetically
            tokens_and_frequencies = sorted(counter.items(),
                                            key=lambda tup: tup[0])
            tokens_and_frequencies.sort(key=lambda tup: tup[1], reverse=True)
            vocab_tokens = [i[0] for i in tokens_and_frequencies[:limit]]
            return vocab_tokens

        tokens = []
        for i in data.examples:
            if field == "src":
                tokens.extend(i.src)
            elif field == "trg":
                tokens.extend(i.trg)

        counter = Counter(tokens)
        if min_freq > -1:
            counter = filter_min(counter, min_freq)
        vocab_tokens = specials + sort_and_cut(counter, max_size)
        assert vocab_tokens[DEFAULT_UNK_ID()] == UNK_TOKEN
        assert len(vocab_tokens) <= max_size + len(specials)
        vocab = Vocabulary(tokens=vocab_tokens)

    # check for all except for UNK token whether they are OOVs
    for s in specials[1:]:
        assert not vocab.is_unk(s)

    return vocab


def array_to_sentence(array, vocabulary, cut_at_eos=True):
    """
    Converts an array of IDs to a sentence, optionally cutting the result
    off at the end-of-sequence token.

    :param array:
    :param vocabulary:
    :param cut_at_eos:
    :return:
    """
    sentence = []
    for i in array:
        s = vocabulary.itos[i]
        if cut_at_eos and s == EOS_TOKEN:
            break
        sentence.append(s)
    return sentence


def arrays_to_sentences(arrays, vocabulary, cut_at_eos=True):
    """
    Convert multiple arrays containing sequences of token IDs to their
    sentences, optionally cutting them off at the end-of-sequence token.

    :param arrays:
    :param vocabulary:
    :param cut_at_eos:
    :return:
    """
    sentences = []
    for array in arrays:
        sentences.append(
            array_to_sentence(array=array, vocabulary=vocabulary,
                              cut_at_eos=cut_at_eos))
    return sentences


def clones(module, N):
    """
    Produce N identical layers. Transformer helper function.

    :param module: the module to clone
    :param N: clone this many times
    :return:
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def subsequent_mask(size):
    """
    Mask out subsequent positions (to prevent attending to future positions)
    Transformer helper function.
    :param size:
    :return:
    """
    attn_shape = (1, size, size)
    mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(mask) == 0


def log_data_info(train_data, valid_data, test_data, src_vocab, trg_vocab,
                  logging_function):
    """
    Log statistics of data and vocabulary.

    :param train_data:
    :param valid_data:
    :param test_data:
    :param src_vocab:
    :param trg_vocab:
    :param logging_function:
    :return:
    """
    logging_function("Data set sizes: \n\ttrain {},\n\tvalid {},\n\ttest {}".format(
        len(train_data), len(valid_data), len(test_data) if test_data is not None else "N/A"))

    logging_function("First training example:\n\t[SRC] {}\n\t[TRG] {}".format(
        " ".join(vars(train_data[0])['src']),
        " ".join(vars(train_data[0])['trg'])))

    logging_function("First 10 words (src): {}".format(" ".join(
        '(%d) %s' % (i, t) for i, t in enumerate(src_vocab.itos[:10]))))
    logging_function("First 10 words (trg): {}".format(" ".join(
        '(%d) %s' % (i, t) for i, t in enumerate(trg_vocab.itos[:10]))))

    logging_function("Number of Src words (types): {}".format(len(src_vocab)))
    logging_function("Number of Trg words (types): {}".format(len(trg_vocab)))


def load_data(cfg):
    """
    Load train, dev and test data as specified in ccnfiguration.
    :param cfg:
    :return:
    """
    # load data from files
    data_cfg = cfg["data"]
    src_lang = data_cfg["src"]
    trg_lang = data_cfg["trg"]
    train_path = data_cfg["train"]
    dev_path = data_cfg["dev"]
    test_path = data_cfg.get("test", None)
    level = data_cfg["level"]
    lowercase = data_cfg["lowercase"]
    max_sent_length = data_cfg["max_sent_length"]

    if level == "char":
        tok_fun = lambda s: list(s)
    else:  # bpe or word, pre-tokenized
        tok_fun = lambda s: s.split()

    src_field = data.Field(init_token=None, eos_token=EOS_TOKEN,  #FIXME
                           pad_token=PAD_TOKEN, tokenize=tok_fun,
                           batch_first=True, lower=lowercase,
                           unk_token=UNK_TOKEN,
                           include_lengths=True)

    trg_field = data.Field(init_token=BOS_TOKEN, eos_token=EOS_TOKEN,
                           pad_token=PAD_TOKEN, tokenize=tok_fun,
                           unk_token=UNK_TOKEN,
                           batch_first=True, lower=lowercase,
                           include_lengths=True)
    train_data = TranslationDataset(path=train_path,
                                    exts=("." + src_lang, "." + trg_lang),
                                    fields=(src_field, trg_field),
                                    filter_pred=
                                    lambda x: len(vars(x)['src'])
                                              <= max_sent_length and
                                              len(vars(x)['trg'])
                                              <= max_sent_length)
    max_size = data_cfg.get("voc_limit", -1)
    min_freq = data_cfg.get("voc_min_freq", 1)
    src_vocab_file = data_cfg.get("src_vocab", None)
    trg_vocab_file = data_cfg.get("trg_vocab", None)

    src_vocab = build_vocab(field="src", min_freq=min_freq, max_size=max_size,
                            data=train_data, vocab_file=src_vocab_file)
    trg_vocab = build_vocab(field="trg", min_freq=min_freq, max_size=max_size,
                            data=train_data, vocab_file=trg_vocab_file)
    dev_data = TranslationDataset(path=dev_path,
                                  exts=("." + src_lang, "." + trg_lang),
                                  fields=(src_field, trg_field))
    test_data = None
    if test_path is not None:
        # check if target exists
        if os.path.isfile(test_path+"."+trg_lang):
            test_data = TranslationDataset(
                path=test_path, exts=("." + src_lang, "." + trg_lang),
                fields=(src_field, trg_field))
        else:
            # no target is given -> create dataset from src only

            test_data = MonoDataset(path=test_path, ext="." + src_lang,
                                    field=(src_field))
    src_field.vocab = src_vocab
    trg_field.vocab = trg_vocab
    return train_data, dev_data, test_data, src_vocab, trg_vocab


class MonoDataset(TranslationDataset):
    """Defines a dataset for machine translation without targets."""

    def __init__(self, path, ext, field, **kwargs):
        """Create a MonoDataset given path and field.

        Arguments:
            path: Prefix of path to the data file
            ext: Containing the extension to path for this language.
            field: Containing the fields that will be used for data
            Remaining keyword arguments: Passed to the constructor of
                data.Dataset.
        """
        fields = [('src', field)]

        src_path = os.path.expanduser(path + ext)

        examples = []
        with open(src_path) as src_file:
            for src_line in src_file:
                src_line = src_line.strip()
                if src_line != '':
                    examples.append(data.Example.fromlist(
                        [src_line], fields))

        super(TranslationDataset, self).__init__(examples, fields, **kwargs)


def load_config(path="configs/default.yaml"):
    """
    Loads and parses a YAML configuration file.
    :param path:
    :return:
    """
    with open(path, 'r') as ymlfile:
        cfg = yaml.load(ymlfile)
    return cfg


def bpe_postprocess(string):
    """
    Post-processor for BPE output. Recombines BPE-split tokens.
    :param string:
    :return:
    """
    return string.replace("@@ ", "")


def store_attention_plots(attentions, targets, sources, output_prefix,
                          idx):
    """
    Saves attention plots.
    :param attentions:
    :param targets:
    :param sources:
    :param output_prefix:
    :param idx:
    :return:
    """
    for i in idx:
        plot_file = "{}.{}.pdf".format(output_prefix, i)
        src = sources[i]
        trg = targets[i]
        attention_scores = attentions[i].T
        try:
            plot_heatmap(scores=attention_scores, column_labels=trg,
                        row_labels=src, output_path=plot_file)
        except:
            print("Couldn't plot example {}: src len {}, trg len {}, "
                  "attention scores shape {}".format(i, len(src), len(trg),
                                                     attention_scores.shape))
            continue


def get_latest_checkpoint(dir):
    """
    Returns the latest checkpoint (by time) from the given directory.
    :param dir:
    :return:
    """
    list_of_files = glob.glob("{}/*.ckpt".format(dir))
    latest_file = max(list_of_files, key=os.path.getctime)
    return latest_file


def load_model_from_checkpoint(path, use_cuda=True):
    """
    Load model from saved checkpoint.
    :param path:
    :param use_cuda:
    :return:
    """
    assert os.path.isfile(path), "Checkpoint %s not found" % path
    model_checkpoint = torch.load(path,
                                  map_location='cuda' if use_cuda else 'cpu')
    return model_checkpoint


def make_data_iter(dataset, batch_size, train=False, shuffle=False):
    """
    Returns a torchtext iterator for a torchtext dataset.
    :param dataset:
    :param batch_size:
    :param train:
    :param shuffle:
    :return:
    """
    if train:
        # optionally shuffle and sort during training
        data_iter = data.BucketIterator(
            repeat=False, sort=False, dataset=dataset,
            batch_size=batch_size, train=True, sort_within_batch=True,
            sort_key=lambda x: len(x.src), shuffle=shuffle)
    else:
        # don't sort/shuffle for validation/inference
        data_iter = data.Iterator(
            repeat=False, dataset=dataset, batch_size=batch_size,
            train=False, sort=False)

    return data_iter


# from onmt
def tile(x, count, dim=0):
    """
    Tiles x on dimension dim count times. From OpenNMT. Used for beam search.

    :param x:
    :param count:
    :param dim:
    :return:
    """
    if isinstance(x, tuple):
        h, c = x
        return tile(h, count, dim=dim), tile(c, count, dim=dim)

    perm = list(range(len(x.size())))
    if dim != 0:
        perm[0], perm[dim] = perm[dim], perm[0]
        x = x.permute(perm).contiguous()
    out_size = list(x.size())
    out_size[0] *= count
    batch = x.size(0)
    x = x.view(batch, -1) \
         .transpose(0, 1) \
         .repeat(count, 1) \
         .transpose(0, 1) \
         .contiguous() \
         .view(*out_size)
    if dim != 0:
        x = x.permute(perm).contiguous()
    return x