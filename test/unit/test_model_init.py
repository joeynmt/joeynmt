from torch.nn import GRU, LSTM
import torch
from torch import nn
import numpy as np

from joeynmt.encoders import RecurrentEncoder
from .test_helpers import TensorTestCase
from joeynmt.model import build_model
from joeynmt.vocabulary import Vocabulary
import copy


class TestModelInit(TensorTestCase):

    def setUp(self):
        self.seed = 42
        vocab_size = 30
        tokens = ["tok{:02d}".format(i) for i in range(vocab_size)]
        self.vocab = Vocabulary(tokens=tokens)
        self.hidden_size = 64

        self.cfg = {
            "model": {
                "tied_embeddings": False,
                "tied_softmax": False,
                "encoder": {
                    "type": "transformer",
                    "hidden_size": self.hidden_size,
                    "embeddings": {"embedding_dim": self.hidden_size},
                    "num_layers": 1,
                },
                "decoder": {
                    "type": "transformer",
                    "hidden_size": self.hidden_size,
                    "embeddings": {"embedding_dim": self.hidden_size},
                    "num_layers": 1,
                },
            }
        }

    def test_transformer_layer_norm_init(self):
        torch.manual_seed(self.seed)
        cfg = copy.deepcopy(self.cfg)

        src_vocab = trg_vocab = self.vocab

        model = build_model(cfg["model"],
                            src_vocab=src_vocab, trg_vocab=trg_vocab)

        def check_layer_norm(m: nn.Module):
            for name, child in m.named_children():
                if isinstance(child, nn.LayerNorm):
                    self.assertTensorEqual(child.weight,
                                           torch.ones([self.hidden_size]))
                    self.assertTensorEqual(child.bias,
                                           torch.zeros([self.hidden_size]))
                else:
                    check_layer_norm(child)

        check_layer_norm(model)
