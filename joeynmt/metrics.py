# coding: utf-8
"""
This module holds various MT evaluation metrics.
"""

import sacrebleu
from typing import List


def chrf(hypotheses, references, remove_whitespace=True):
    """
    Character F-score from sacrebleu

    :param hypotheses: list of hypotheses (strings)
    :param references: list of references (strings)
    :param remove_whitespace: (bool)
    :return:
    """
    return sacrebleu.corpus_chrf(hypotheses=hypotheses, references=[references],
                                 remove_whitespace=remove_whitespace).score


def bleu(hypotheses, references, tokenize="13a"):
    """
    Raw corpus BLEU from sacrebleu (without tokenization)

    :param hypotheses: list of hypotheses (strings)
    :param references: list of references (strings)
    :param tokenize: one of {'none', '13a', 'intl', 'zh', 'ja-mecab'}
    :return:
    """
    return sacrebleu.corpus_bleu(sys_stream=hypotheses, ref_streams=[references],
                                 tokenize=tokenize).score


def token_accuracy(hypotheses: List[List[str]], references: List[List[str]]) -> float:
    """
    Compute the accuracy of hypothesis tokens: correct tokens / all tokens
    Tokens are correct if they appear in the same position in the reference.

    :param hypotheses: list of hypotheses (List[str])
    :param references: list of references (List[str])
    :return: token accuracy (float)
    """
    correct_tokens = 0
    all_tokens = 0
    assert len(hypotheses) == len(references)
    for hyp, ref in zip(hypotheses, references):
        all_tokens += len(hyp)
        for h_i, r_i in zip(hyp, ref):
            # min(len(h), len(r)) tokens considered
            if h_i == r_i:
                correct_tokens += 1
    return (correct_tokens / all_tokens)*100 if all_tokens > 0 else 0.0


def _token_accuracy(hypotheses, references, level="word"):
    """
    Compute the accuracy of hypothesis tokens: correct tokens / all tokens
    Tokens are correct if they appear in the same position in the reference.

    :param hypotheses: list of hypotheses (strings)
    :param references: list of references (strings)
    :param level: segmentation level, either "word", "bpe", or "char"
    :return:
    """
    def split_by_space(string):
        """
        Helper method to split the input based on spaces.
        Follows the same structure as list(inp)
        :param string: string
        :return: list of strings
        """
        return string.split(" ")

    correct_tokens = 0
    all_tokens = 0
    split_func = split_by_space if level in ["word", "bpe"] else list
    assert len(hypotheses) == len(references)
    for hyp, ref in zip(hypotheses, references):
        all_tokens += len(hyp)
        for h_i, r_i in zip(split_func(hyp), split_func(ref)):
            # min(len(h), len(r)) tokens considered
            if h_i == r_i:
                correct_tokens += 1
    return (correct_tokens / all_tokens)*100 if all_tokens > 0 else 0.0


def sequence_accuracy(hypotheses, references):
    """
    Compute the accuracy of hypothesis tokens: correct tokens / all tokens
    Tokens are correct if they appear in the same position in the reference.

    :param hypotheses: list of hypotheses (strings)
    :param references: list of references (strings)
    :return:
    """
    assert len(hypotheses) == len(references)
    correct_sequences = sum([1 for (hyp, ref) in zip(hypotheses, references)
                             if hyp == ref])
    return (correct_sequences / len(hypotheses))*100 if hypotheses else 0.0
