import unittest

import torch

from joeynmt.data import load_data
from joeynmt.model import build_model
from joeynmt.prediction import parse_test_args, validate_on_data

# TODO make sure rnn also returns the nbest list in the resorted order


class TestPrediction(unittest.TestCase):
    def setUp(self):
        seed = 42
        torch.manual_seed(seed)
        self.cfg = {
            "data": {
                "src": "de",
                "trg": "en",
                "train": "test/data/toy/train",     # needed for vocab
                "test": "test/data/toy/test",
                "level": "word",
                "lowercase": False,
                "max_sent_length": 10
            },
            "testing": {
                "bpe_type": None,
                "beam_size": 5,
                "alpha": 1.0
            },
            "training": {
                "batch_size": 2,
                "batch_type": "sentence",
                "eval_metric": "bleu"
            },
            "model": {
                "tied_embeddings": False,
                "tied_softmax": False,
                "encoder": {
                    "type": "transformer",
                    "hidden_size": 12,
                    "ff_size": 24,
                    "embeddings": {"embedding_dim": 12},
                    "num_layers": 1,
                    "num_heads": 4
                },
                "decoder": {
                    "type": "transformer",
                    "hidden_size": 12,
                    "ff_size": 24,
                    "embeddings": {"embedding_dim": 12},
                    "num_layers": 1,
                    "num_heads": 4
                },
            }
        }

        # load data
        _, _, test_data, src_vocab, trg_vocab = load_data(
            self.cfg["data"], datasets=["train", "test"])
        self.test_data = test_data
        self.parsed_cfg = parse_test_args(self.cfg, mode="translate")

        # build model
        self.model = build_model(self.cfg["model"],
                                 src_vocab=src_vocab, trg_vocab=trg_vocab)

    def _translate(self, n_best):
        (batch_size, batch_type, use_cuda, device, n_gpu, level, eval_metric,
         max_output_length, beam_size, beam_alpha, postprocess, bpe_type,
         sacrebleu, _, _) = self.parsed_cfg

        (score, loss, ppl, sources, sources_raw, references, hypotheses,
         hypotheses_raw, attention_scores) = validate_on_data(
            self.model, data=self.test_data, batch_size=batch_size,
            batch_type=batch_type, level=level, use_cuda=use_cuda,
            max_output_length=max_output_length, eval_metric=None,
            compute_loss=False, beam_size=beam_size, beam_alpha=beam_alpha,
            postprocess=postprocess, bpe_type=bpe_type, sacrebleu=sacrebleu,
            n_gpu=n_gpu, n_best=n_best)
        return sources, hypotheses

    def test_transformer_nbest(self):
        n_best = 1
        sources_1best, hypotheses_1best = self._translate(n_best)
        self.assertEqual(len(self.test_data), len(hypotheses_1best))

        n_best = 5
        sources_5best, hypotheses_5best = self._translate(n_best)
        self.assertEqual(len(self.test_data) * n_best, len(hypotheses_5best))

        for n in range(n_best):
            hyp = [hypotheses_5best[i]
                   for i in range(n, len(hypotheses_5best), n_best)]
            self.assertEqual(len(self.test_data), len(hyp))     # unroll
            if n == 0:
                # hypotheses must match 1best_hypotheses
                self.assertEqual(hypotheses_1best, hyp)

        n_best = 10
        with self.assertRaises(AssertionError) as e:
            self._translate(n_best)
        self.assertEqual('Can only return 5 best hypotheses.', str(e.exception))
