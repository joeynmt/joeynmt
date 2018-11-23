# coding: utf-8

import sacrebleu


def chrf(hypotheses, references):
    """
    Character F-score from sacrebleu

    :param hypotheses:
    :param references:
    :return:
    """
    return sacrebleu.corpus_chrf(
                    hypotheses=hypotheses,
                    references=references)


def bleu(hypotheses, references):
    """
    Raw corpus BLEU from sacrebleu (without tokenization)

    :param hypotheses:
    :param references:
    :return:
    """
    return sacrebleu.raw_corpus_bleu(
                    sys_stream=hypotheses,
                    ref_streams=[references]).score


def token_accuracy(hypotheses, references, level="word"):
    """
    Compute the accuracy of hypothesis tokens: correct tokens / all tokens
    Tokens are correct if they appear in the same position in the reference.

    :param hypotheses:
    :param references:
    :return:
    """
    correct_tokens = 0
    all_tokens = 0
    split_char = " " if level in ["word", "bpe"] else ""
    assert len(hypotheses) == len(references)
    for h, r in zip(hypotheses, references):
        all_tokens += len(h)
        for h_i, r_i in zip(h.split(split_char), r.split(split_char)):
            # min(len(h), len(r)) tokens considered
            if h_i == r_i:
                correct_tokens += 1
    return (correct_tokens / all_tokens)*100 if all_tokens > 0 else 0.0


def sequence_accuracy(hypotheses, references):
    """
    Compute the accuracy of hypothesis tokens: correct tokens / all tokens
    Tokens are correct if they appear in the same position in the reference.

    :param hypotheses:
    :param references:
    :return:
    """
    assert len(hypotheses) == len(references)
    correct_sequences = sum([1 for (h, r) in zip(hypotheses, references)
                             if h == r])
    return (correct_sequences / len(hypotheses))*100 if len(hypotheses) > 0 \
        else 0.0
