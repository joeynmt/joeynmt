import copy
import unittest
from types import SimpleNamespace

import torch

from joeynmt.model import build_model
from joeynmt.vocabulary import Vocabulary


class TestWeightTying(unittest.TestCase):

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

    def test_tied_embeddings(self):

        torch.manual_seed(self.seed)
        cfg = copy.deepcopy(self.cfg)
        cfg["model"]["tied_embeddings"] = True
        cfg["model"]["tied_softmax"] = False

        src_vocab = trg_vocab = self.vocab

        model = build_model(cfg["model"], src_vocab=src_vocab, trg_vocab=trg_vocab)

        self.assertEqual(src_vocab, trg_vocab)
        self.assertEqual(model.src_embed, model.trg_embed)
        torch.testing.assert_close(
            model.src_embed.lut.weight, model.trg_embed.lut.weight
        )
        self.assertEqual(
            model.src_embed.lut.weight.shape, model.trg_embed.lut.weight.shape
        )

    def test_tied_softmax(self):

        torch.manual_seed(self.seed)
        cfg = copy.deepcopy(self.cfg)
        cfg["model"]["decoder"]["type"] = "transformer"
        cfg["model"]["tied_embeddings"] = False
        cfg["model"]["tied_softmax"] = True
        cfg["model"]["decoder"]["embeddings"]["embedding_dim"] = 64

        src_vocab = trg_vocab = self.vocab

        model = build_model(cfg["model"], src_vocab=src_vocab, trg_vocab=trg_vocab)

        self.assertEqual(
            model.trg_embed.lut.weight.shape,
            model.decoder.output_layer.weight.shape,
        )

        torch.testing.assert_close(
            model.trg_embed.lut.weight, model.decoder.output_layer.weight
        )

    def test_tied_src_trg_softmax(self):

        # test source embedding, target embedding, and softmax tying
        torch.manual_seed(self.seed)
        cfg = copy.deepcopy(self.cfg)

        cfg["model"]["decoder"]["type"] = "transformer"
        cfg["model"]["tied_embeddings"] = True
        cfg["model"]["tied_softmax"] = True
        cfg["model"]["decoder"]["embeddings"]["embedding_dim"] = 64
        cfg["model"]["encoder"]["embeddings"]["embedding_dim"] = 64

        src_vocab = trg_vocab = self.vocab
        model = build_model(cfg["model"], src_vocab=src_vocab, trg_vocab=trg_vocab)

        src_weight = model.src_embed.lut.weight
        trg_weight = model.trg_embed.lut.weight
        output_weight = model.decoder.output_layer.weight

        torch.testing.assert_close(src_weight, trg_weight)
        torch.testing.assert_close(src_weight, output_weight)
        self.assertEqual(src_weight.shape, trg_weight.shape)
        self.assertEqual(trg_weight.shape, output_weight.shape)

        output_weight.data.fill_(3.0)
        self.assertEqual(output_weight.sum().item(), 7104)
        self.assertEqual(output_weight.sum().item(), src_weight.sum().item())
        self.assertEqual(output_weight.sum().item(), trg_weight.sum().item())
        self.assertEqual(src_weight.sum().item(), trg_weight.sum().item())
