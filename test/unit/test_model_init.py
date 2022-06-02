import copy
from test.unit.test_helpers import TensorTestCase

import torch
from torch import nn

from joeynmt.model import build_model
from joeynmt.vocabulary import Vocabulary


class TestModelInit(TensorTestCase):

    def setUp(self):
        self.seed = 42
        vocab_size = 30
        tokens = [f"tok{i:02d}" for i in range(vocab_size)]
        self.vocab = Vocabulary(tokens=tokens)
        self.hidden_size = 64

        self.cfg = {
            "model": {
                "tied_embeddings": False,
                "tied_softmax": False,
                "encoder": {
                    "type": "transformer",
                    "hidden_size": self.hidden_size,
                    "embeddings": {
                        "embedding_dim": self.hidden_size
                    },
                    "num_layers": 1,
                    "layer_norm": "pre",
                },
                "decoder": {
                    "type": "transformer",
                    "hidden_size": self.hidden_size,
                    "embeddings": {
                        "embedding_dim": self.hidden_size
                    },
                    "num_layers": 1,
                    "layer_norm": "pre",
                },
            }
        }

    def test_transformer_layer_norm_init(self):
        torch.manual_seed(self.seed)
        cfg = copy.deepcopy(self.cfg)

        src_vocab = trg_vocab = self.vocab

        model = build_model(cfg["model"], src_vocab=src_vocab, trg_vocab=trg_vocab)

        def check_layer_norm(m: nn.Module):
            for _, child in m.named_children():
                if isinstance(child, nn.LayerNorm):
                    self.assertTensorEqual(child.weight, torch.ones([self.hidden_size]))
                    self.assertTensorEqual(child.bias, torch.zeros([self.hidden_size]))
                else:
                    check_layer_norm(child)

        check_layer_norm(model)

    def test_transformer_layers_init(self):
        torch.manual_seed(self.seed)
        cfg = copy.deepcopy(self.cfg)
        cfg["model"]["initializer"] = "xavier_normal"
        cfg["model"]["encoder"]["num_layers"] = 6
        cfg["model"]["decoder"]["num_layers"] = 6

        src_vocab = trg_vocab = self.vocab

        model = build_model(cfg["model"], src_vocab=src_vocab, trg_vocab=trg_vocab)

        self.assertEqual(len(model.encoder.layers), 6)
        self.assertEqual(len(model.decoder.layers), 6)

        for layer in model.encoder.layers:
            self.assertEqual(layer.alpha, 1.417938140685523)
        for layer in model.decoder.layers:
            self.assertEqual(layer.alpha, 2.0597671439071177)

        self.assertTensorAlmostEqual(
            model.encoder.layers[0].src_src_att.q_layer.weight[:5, 0].data,
            torch.Tensor([-0.2093, -0.1066, -0.1455, -0.1146, 0.0760]),
        )
        self.assertTensorAlmostEqual(
            model.decoder.layers[0].src_trg_att.q_layer.weight[:5, 0].data,
            torch.Tensor([0.0072, -0.0241, 0.2873, -0.0417, -0.2752]),
        )
        self.assertTensorAlmostEqual(
            model.decoder.layers[0].trg_trg_att.q_layer.weight[:5, 0].data,
            torch.Tensor([-0.2140, 0.0942, 0.0203, 0.0417, 0.2482]),
        )
