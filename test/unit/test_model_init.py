import copy
import unittest
from types import SimpleNamespace

import torch
from torch import nn

from joeynmt.model import build_model
from joeynmt.vocabulary import Vocabulary


class TestModelInit(unittest.TestCase):

    def setUp(self):
        self.seed = 42
        vocab_size = 30
        tokens = [f"tok{i:02d}" for i in range(vocab_size)]
        special_symbols = SimpleNamespace(
            **{
                "unk_token": "<unk>",
                "pad_token": "<pad>",
                "bos_token": "<s>",
                "eos_token": "</s>",
                "sep_token": "<sep>",
                "unk_id": 0,
                "pad_id": 1,
                "bos_id": 2,
                "eos_id": 3,
                "sep_id": 4,
                "lang_tags": ["<de>", "<en>"],
            }
        )
        self.vocab = Vocabulary(tokens=tokens, cfg=special_symbols)
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
                    "layer_norm": "pre",
                    "activation": "relu",
                },
                "decoder": {
                    "type": "transformer",
                    "hidden_size": self.hidden_size,
                    "embeddings": {"embedding_dim": self.hidden_size},
                    "num_layers": 1,
                    "layer_norm": "pre",
                    "activation": "relu",
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
                    torch.testing.assert_close(
                        child.weight, torch.ones([self.hidden_size])
                    )
                    torch.testing.assert_close(
                        child.bias, torch.zeros([self.hidden_size])
                    )
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

        torch.testing.assert_close(
            model.encoder.layers[0].src_src_att.q_layer.weight[:5, 0].data,
            torch.Tensor([0.1232, 0.1870, -0.1077, 0.0748, -0.0651]),
            rtol=1e-4,
            atol=1e-4,
        )
        torch.testing.assert_close(
            model.decoder.layers[0].src_trg_att.q_layer.weight[:5, 0].data,
            torch.Tensor([-0.1035, 0.2171, 0.1729, -0.0120, -0.1008]),
            rtol=1e-4,
            atol=1e-4,
        )
        torch.testing.assert_close(
            model.decoder.layers[0].trg_trg_att.q_layer.weight[:5, 0].data,
            torch.Tensor([-0.2248, -0.0396, 0.2041, 0.0627, 0.0255]),
            rtol=1e-4,
            atol=1e-4,
        )

    def test_transformer_activation_init(self):
        cfg = copy.deepcopy(self.cfg)
        cfg["model"]["encoder"]["activation"] = "gelu"
        cfg["model"]["decoder"]["activation"] = "swish"

        src_vocab = trg_vocab = self.vocab

        model = build_model(cfg["model"], src_vocab=src_vocab, trg_vocab=trg_vocab)
        self.assertTrue(model.encoder.layers[0].feed_forward.pwff_layer[1], nn.GELU)
        self.assertTrue(model.decoder.layers[0].feed_forward.pwff_layer[1], nn.SiLU)
