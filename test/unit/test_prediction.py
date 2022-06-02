import unittest

import torch

from joeynmt.data import load_data
from joeynmt.helpers import expand_reverse_index
from joeynmt.model import build_model
from joeynmt.prediction import predict

# TODO make sure rnn also returns the nbest list in the resorted order


class TestHelpers(unittest.TestCase):

    def test_expand_reverse_index(self):
        reverse_index = [1, 0, 2]

        n_best = 1
        reverse_index_1best = expand_reverse_index(reverse_index, n_best)
        self.assertEqual(reverse_index_1best, [1, 0, 2])

        n_best = 2
        reverse_index_2best = expand_reverse_index(reverse_index, n_best)
        self.assertEqual(reverse_index_2best, [2, 3, 0, 1, 4, 5])

        n_best = 3
        reverse_index_3best = expand_reverse_index(reverse_index, n_best)
        self.assertEqual(reverse_index_3best, [3, 4, 5, 0, 1, 2, 6, 7, 8])


class TestPrediction(unittest.TestCase):

    def setUp(self):
        seed = 42
        torch.manual_seed(seed)
        self.cfg = {
            "data": {
                "train": "test/data/toy/train",  # needed for vocab
                "test": "test/data/toy/test",
                "src": {
                    "lang": "de",
                    "level": "word",
                    "lowercase": False,
                    "max_length": 10,
                },
                "trg": {
                    "lang": "en",
                    "level": "word",
                    "lowercase": False,
                    "max_length": 10,
                },
                "dataset_type": "plain",
            },
            "testing": {
                "n_best": 1,
                "batch_size": 2,
                "batch_type": "sentence",
                "beam_size": 5,
                "beam_alpha": 1.0,
                "eval_metrics": ["bleu"],
                "return_prob": "none",
                "sacrebleu_cfg": {
                    "tokenize": "13a"
                },
            },
            "model": {
                "tied_embeddings": False,
                "tied_softmax": False,
                "encoder": {
                    "type": "transformer",
                    "hidden_size": 12,
                    "ff_size": 24,
                    "embeddings": {
                        "embedding_dim": 12
                    },
                    "num_layers": 1,
                    "num_heads": 4,
                    "layer_norm": "pre",
                },
                "decoder": {
                    "type": "transformer",
                    "hidden_size": 12,
                    "ff_size": 24,
                    "embeddings": {
                        "embedding_dim": 12
                    },
                    "num_layers": 1,
                    "num_heads": 4,
                    "layer_norm": "pre",
                },
            },
        }

        # load data
        src_vocab, trg_vocab, _, _, self.test_data = load_data(
            self.cfg["data"], datasets=["train", "test"])

        # build model
        self.model = build_model(self.cfg["model"],
                                 src_vocab=src_vocab,
                                 trg_vocab=trg_vocab)

    def _translate(self, n_best):
        cfg = self.cfg["testing"].copy()
        cfg["n_best"] = n_best
        _, _, hypotheses, _, _, _ = predict(
            self.model,
            data=self.test_data,
            compute_loss=False,
            device=torch.device("cpu"),
            n_gpu=0,
            num_workers=0,
            normalization="none",
            cfg=cfg,
        )
        return hypotheses

    def test_transformer_nbest(self):
        self.assertFalse(self.test_data.has_trg)

        n_best = 1
        hypotheses_1best = self._translate(n_best)
        self.assertEqual(len(self.test_data), len(hypotheses_1best))

        n_best = 5
        hypotheses_5best = self._translate(n_best)
        self.assertEqual(len(self.test_data) * n_best, len(hypotheses_5best))

        for n in range(n_best):
            hyp = [hypotheses_5best[i] for i in range(n, len(hypotheses_5best), n_best)]
            self.assertEqual(len(self.test_data), len(hyp))  # unroll
            if n == 0:
                # hypotheses must match 1best_hypotheses
                self.assertEqual(hypotheses_1best, hyp)

        n_best = 10
        with self.assertRaises(AssertionError) as e:
            self._translate(n_best)
        self.assertEqual("`n_best` must be smaller than or equal to `beam_size`.",
                         str(e.exception))
