# coding: utf-8
"""
Evaluation metrics
"""
import logging
from inspect import getfullargspec
from typing import List

from sacrebleu.metrics import BLEU, CHRF

logger = logging.getLogger(__name__)


def chrf(hypotheses: List[str], references: List[str], **sacrebleu_cfg) -> float:
    """
    Character F-score from sacrebleu
    cf. https://github.com/mjpost/sacrebleu/blob/master/sacrebleu/metrics/chrf.py

    :param hypotheses: list of hypotheses (strings)
    :param references: list of references (strings)
    :return: character f-score (0 <= chf <= 1)
             see Breaking Change in sacrebleu v2.0
    """
    kwargs = {}
    if sacrebleu_cfg:
        valid_keys = getfullargspec(CHRF).args
        for k, v in sacrebleu_cfg.items():
            if k in valid_keys:
                kwargs[k] = v

    metric = CHRF(**kwargs)
    score = metric.corpus_score(hypotheses=hypotheses, references=[references]).score

    # log sacrebleu signature
    logger.info(metric.get_signature())
    return score / 100


def bleu(hypotheses: List[str], references: List[str], **sacrebleu_cfg) -> float:
    """
    Raw corpus BLEU from sacrebleu (without tokenization)
    cf. https://github.com/mjpost/sacrebleu/blob/master/sacrebleu/metrics/bleu.py

    :param hypotheses: list of hypotheses (strings)
    :param references: list of references (strings)
    :return: bleu score
    """
    kwargs = {}
    if sacrebleu_cfg:
        valid_keys = getfullargspec(BLEU).args
        for k, v in sacrebleu_cfg.items():
            if k in valid_keys:
                kwargs[k] = v

    metric = BLEU(**kwargs)
    score = metric.corpus_score(hypotheses=hypotheses, references=[references]).score

    # log sacrebleu signature
    logger.info(metric.get_signature())
    return score


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
        [1 for (hyp, ref) in zip(hypotheses, references) if hyp == ref])
    return (correct_sequences / len(hypotheses)) * 100 if hypotheses else 0.0
