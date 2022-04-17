# coding: utf-8
"""
This module holds various MT evaluation metrics.
"""

from typing import List

import sacrebleu


def chrf(
    hypotheses: List[str], references: List[str], remove_whitespace: bool = True
) -> float:
    """
    Character F-score from sacrebleu

    :param hypotheses: list of hypotheses (strings)
    :param references: list of references (strings)
    :param remove_whitespace: (bool)
    :return: character f-score (0 <= chf <= 1)
             see Breaking Change in sacrebleu v2.0
    """
    score = sacrebleu.corpus_chrf(
        hypotheses=hypotheses,
        references=[references],
        remove_whitespace=remove_whitespace,
    ).score
    return score / 100


def bleu(hypotheses: List[str], references: List[str], tokenize: str = "13a") -> float:
    """
    Raw corpus BLEU from sacrebleu (without tokenization)

    :param hypotheses: list of hypotheses (strings)
    :param references: list of references (strings)
    :param tokenize: one of {'none', '13a', 'intl', 'zh', 'ja-mecab'}
    :return: bleu score
    """
    return sacrebleu.corpus_bleu(
        hypotheses=hypotheses, references=[references], tokenize=tokenize
    ).score


def token_accuracy(hypotheses: List[List[str]], references: List[List[str]]) -> float:
    """
    Compute the accuracy of hypothesis tokens: correct tokens / all tokens
    Tokens are correct if they appear in the same position in the reference.
    We lookup the references before one-hot-encoding, that is, UNK generation in
    hypotheses is always evaluated as incorrect.

    :param hypotheses: list of tokenized hypotheses (List[List[str]])
    :param references: list of tokenized references (List[List[str]])
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
    return (correct_tokens / all_tokens) * 100 if all_tokens > 0 else 0.0


def sequence_accuracy(hypotheses: List[str], references: List[str]) -> float:
    """
    Compute the accuracy of hypothesis tokens: correct tokens / all tokens
    Tokens are correct if they appear in the same position in the reference.
    We lookup the references before one-hot-encoding, that is, hypotheses with UNK
    are always evaluated as incorrect.

    :param hypotheses: list of hypotheses (strings)
    :param references: list of references (strings)
    :return:
    """
    assert len(hypotheses) == len(references)
    correct_sequences = sum(
        [1 for (hyp, ref) in zip(hypotheses, references) if hyp == ref]
    )
    return (correct_sequences / len(hypotheses)) * 100 if hypotheses else 0.0
