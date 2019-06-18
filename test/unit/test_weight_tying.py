from torch.nn import GRU, LSTM
import torch
import numpy as np

from joeynmt.encoders import RecurrentEncoder
from .test_helpers import TensorTestCase
from joeynmt.model import build_model
from joeynmt.vocabulary import Vocabulary
import copy


class TestWeightTying(TensorTestCase):

    def setUp(self):
        self.seed = 42
        vocab_size = 30
        tokens = ["tok{:02d}".format(i) for i in range(vocab_size)]
        self.vocab = Vocabulary(tokens=tokens)

        self.cfg = {
            "model": {
                "tied_embeddings": False,
                "tied_softmax": False,
                "encoder": {
                    "type": "recurrent",
                    "hidden_size": 64,
                    "embeddings": {"embedding_dim": 32},
                    "num_layers": 1,
                },
                "decoder": {
                    "type": "recurrent",
                    "hidden_size": 64,
                    "embeddings": {"embedding_dim": 32},
                    "num_layers": 1,
                },
            }
        }

    def test_tied_src_trg_embeddings(self):

        torch.manual_seed(self.seed)
        cfg = copy.deepcopy(self.cfg)
        cfg["model"]["tied_embeddings"] = True
        cfg["model"]["tied_softmax"] = False

        src_vocab = trg_vocab = self.vocab

        model = build_model(cfg["model"],
                            src_vocab=src_vocab, trg_vocab=trg_vocab)

        self.assertEqual(src_vocab.itos, trg_vocab.itos)
        self.assertEqual(model.src_embed, model.trg_embed)
        self.assertTensorEqual(model.src_embed.lut.weight,
                               model.trg_embed.lut.weight)
        self.assertEqual(model.src_embed.lut.weight.shape,
                         model.trg_embed.lut.weight.shape)

    def test_tied_softmax(self):

        torch.manual_seed(self.seed)

        cfg = copy.deepcopy(self.cfg)
        cfg["model"]["decoder"]["type"] = "transformer"
        cfg["model"]["tied_embeddings"] = False
        cfg["model"]["tied_softmax"] = True
        cfg["model"]["decoder"]["embeddings"]["embedding_dim"] = 64

        src_vocab = trg_vocab = self.vocab

        model = build_model(cfg["model"],
                            src_vocab=src_vocab, trg_vocab=trg_vocab)

        self.assertEqual(model.trg_embed.lut.weight.shape,
                         model.decoder.output_layer.weight.shape)

        self.assertTensorEqual(model.trg_embed.lut.weight,
                               model.decoder.output_layer.weight)

        # test source embedding, target embedding, and softmax tying
        cfg["model"]["tied_embeddings"] = True
        cfg["model"]["encoder"]["embeddings"]["embedding_dim"] = 64
        model = build_model(cfg["model"],
                            src_vocab=src_vocab, trg_vocab=trg_vocab)

        self.assertTensorEqual(model.src_embed.lut.weight,
                               model.trg_embed.lut.weight)
        self.assertEqual(model.src_embed.lut.weight.shape,
                         model.trg_embed.lut.weight.shape)
