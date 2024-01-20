import unittest

import torch

from joeynmt.embeddings import Embeddings


class TestEmbeddings(unittest.TestCase):

    def setUp(self):
        self.emb_size = 10
        self.vocab_size = 11
        self.pad_idx = 1
        seed = 42
        torch.manual_seed(seed)

    def test_size(self):
        emb = Embeddings(
            embedding_dim=self.emb_size,
            vocab_size=self.vocab_size,
            padding_idx=self.pad_idx,
        )
        self.assertEqual(
            emb.lut.weight.shape, torch.Size([self.vocab_size, self.emb_size])
        )

    def test_pad_zeros(self):
        emb = Embeddings(
            embedding_dim=self.emb_size,
            vocab_size=self.vocab_size,
            padding_idx=self.pad_idx,
        )
        # pad embedding should be zeros
        torch.testing.assert_close(
            emb.lut.weight[self.pad_idx], torch.zeros([self.emb_size])
        )

    def test_freeze(self):
        encoder = Embeddings(
            embedding_dim=self.emb_size,
            vocab_size=self.vocab_size,
            padding_idx=self.pad_idx,
            freeze=True,
        )
        for _, p in encoder.named_parameters():
            self.assertFalse(p.requires_grad)

    def test_forward(self):
        # fix the embedding weights
        weights = self._get_random_embedding_weights()
        emb = Embeddings(
            embedding_dim=self.emb_size,
            vocab_size=self.vocab_size,
            padding_idx=self.pad_idx,
        )
        emb.lut.weight.data = weights  # fill embeddings
        indices = torch.Tensor([0, 1, self.pad_idx, 9]).long()
        embedded = emb.forward(x=indices)
        # embedding operation is just slicing from weights matrix
        torch.testing.assert_close(
            embedded, torch.index_select(input=weights, index=indices, dim=0)
        )
        # after embedding, representations for PAD should still be zero
        torch.testing.assert_close(embedded[2], torch.zeros([self.emb_size]))

    def test_scale(self):
        # fix the embedding weights
        weights = self._get_random_embedding_weights()
        emb = Embeddings(
            embedding_dim=self.emb_size,
            vocab_size=self.vocab_size,
            padding_idx=self.pad_idx,
            scale=True,
        )
        emb.lut.weight.data = weights
        indices = torch.Tensor([0, 1, self.pad_idx, 9]).long()
        embedded = emb.forward(x=indices)

        # now scaled
        # yapf: disable
        expected = torch.index_select(input=weights, index=indices, dim=0)
        torch.testing.assert_close(
            embedded, expected * (self.emb_size**0.5), rtol=1e-4, atol=1e-4)

    def _get_random_embedding_weights(self):
        weights = torch.rand([self.vocab_size, self.emb_size])
        weights[self.pad_idx] = torch.zeros([self.emb_size])
        return weights
