import torch
import numpy as np

from joeynmt.search import greedy, recurrent_greedy, transformer_greedy
from joeynmt.search import beam_search
from joeynmt.decoders import RecurrentDecoder, TransformerDecoder
from joeynmt.encoders import RecurrentEncoder
from joeynmt.embeddings import Embeddings

from .test_helpers import TensorTestCase


# TODO for transformer and rnn, make sure both return the same result for
# beam_size<2 and greedy decoding

class TestSearch(TensorTestCase):
    def setUp(self):
        self.emb_size = 12
        self.num_layers = 3
        self.hidden_size = 12
        self.ff_size = 24
        self.num_heads = 4
        self.dropout = 0.
        self.encoder_hidden_size = 3
        self.vocab_size = 5
        seed = 42
        torch.manual_seed(seed)
        self.bos_index = 2
        self.pad_index = 1
        self.eos_index = 3

class TestSearchTransformer(TestSearch):

    def _build(self, batch_size):
        src_time_dim = 4
        vocab_size = 7

        emb = Embeddings(embedding_dim=self.emb_size, vocab_size=vocab_size,
                         padding_idx=self.pad_index)

        decoder = TransformerDecoder(
            num_layers=self.num_layers, num_heads=self.num_heads,
            hidden_size=self.hidden_size, ff_size=self.ff_size,
            dropout=self.dropout, emb_dropout=self.dropout,
            vocab_size=vocab_size)

        encoder_output = torch.rand(
            size=(batch_size, src_time_dim, self.hidden_size))

        for p in decoder.parameters():
            torch.nn.init.uniform_(p, -0.5, 0.5)

        src_mask = torch.ones(size=(batch_size, 1, src_time_dim)) == 1

        encoder_hidden = None  # unused
        return src_mask, emb, decoder, encoder_output, encoder_hidden

    def test_transformer_greedy(self):
        batch_size = 2
        max_output_length = 3
        src_mask, embed, decoder, encoder_output, \
        encoder_hidden = self._build(batch_size=batch_size)
        output, attention_scores = transformer_greedy(
            src_mask=src_mask, embed=embed, bos_index=self.bos_index,
            eos_index=self.eos_index,
            max_output_length=max_output_length, decoder=decoder,
            encoder_output=encoder_output, encoder_hidden=encoder_hidden)
        # Transformer greedy doesn't return attention scores
        self.assertIsNone(attention_scores)
        # batch x time
        self.assertEqual(output.shape, (batch_size, max_output_length))
        np.testing.assert_equal(output, [[5, 5, 5], [5, 5, 5]])

    def test_transformer_beam1(self):
        batch_size = 2
        beam_size = 1
        alpha = 1.
        max_output_length = 3
        src_mask, embed, decoder, encoder_output, \
        encoder_hidden = self._build(batch_size=batch_size)
        output, attention_scores = beam_search(
            size=beam_size, eos_index=self.eos_index, pad_index=self.pad_index,
            src_mask=src_mask, embed=embed, bos_index=self.bos_index,
            max_output_length=max_output_length, decoder=decoder, alpha=alpha,
            encoder_output=encoder_output, encoder_hidden=encoder_hidden)
        # Transformer beam doesn't return attention scores
        self.assertIsNone(attention_scores)
        # batch x time
        self.assertEqual(output.shape, (batch_size, max_output_length))
        np.testing.assert_equal(output, [[5, 5, 5], [5, 5, 5]])

        # now compare to greedy, they should be the same for beam=1
        greedy_output, _ = transformer_greedy(
            src_mask=src_mask, embed=embed, bos_index=self.bos_index,
            eos_index=self.eos_index,
            max_output_length=max_output_length, decoder=decoder,
            encoder_output=encoder_output, encoder_hidden=encoder_hidden)
        np.testing.assert_equal(output, greedy_output)

    def test_transformer_beam7(self):
        batch_size = 2
        beam_size = 7
        alpha = 1.
        max_output_length = 3
        src_mask, embed, decoder, encoder_output, \
        encoder_hidden = self._build(batch_size=batch_size)
        output, attention_scores = beam_search(
            size=beam_size, eos_index=self.eos_index, pad_index=self.pad_index,
            src_mask=src_mask, embed=embed, bos_index=self.bos_index, n_best=1,
            max_output_length=max_output_length, decoder=decoder, alpha=alpha,
            encoder_output=encoder_output, encoder_hidden=encoder_hidden)
        # Transformer beam doesn't return attention scores
        self.assertIsNone(attention_scores)
        # batch x time
        # now it produces EOS, so everything after gets cut off
        self.assertEqual(output.shape, (batch_size, 1))
        np.testing.assert_equal(output, [[3], [3]])


class TestSearchRecurrent(TestSearch):
    def _build(self, batch_size):
        src_time_dim = 4
        vocab_size = 7

        emb = Embeddings(embedding_dim=self.emb_size, vocab_size=vocab_size,
                         padding_idx=self.pad_index)

        encoder = RecurrentEncoder(emb_size=self.emb_size,
                                   num_layers=self.num_layers,
                                   hidden_size=self.encoder_hidden_size,
                                   bidirectional=True)

        decoder = RecurrentDecoder(hidden_size=self.hidden_size,
                                   encoder=encoder, attention="bahdanau",
                                   emb_size=self.emb_size,
                                   vocab_size=self.vocab_size,
                                   num_layers=self.num_layers,
                                   init_hidden="bridge",
                                   input_feeding=True)

        encoder_output = torch.rand(
            size=(batch_size, src_time_dim, encoder.output_size))

        for p in decoder.parameters():
            torch.nn.init.uniform_(p, -0.5, 0.5)

        src_mask = torch.ones(size=(batch_size, 1, src_time_dim)) == 1

        encoder_hidden = torch.rand(size=(batch_size, encoder.output_size))

        return src_mask, emb, decoder, encoder_output, encoder_hidden

    def test_recurrent_greedy(self):
        batch_size = 2
        max_output_length = 3
        src_mask, emb, decoder, encoder_output, encoder_hidden = self._build(
            batch_size=batch_size)

        output, attention_scores = recurrent_greedy(
            src_mask=src_mask, embed=emb, bos_index=self.bos_index,
            eos_index=self.eos_index, max_output_length=max_output_length,
            decoder=decoder,
            encoder_output=encoder_output, encoder_hidden=encoder_hidden)

        self.assertEqual(output.shape, (batch_size, max_output_length))
        np.testing.assert_equal(output, [[4, 0, 4], [4, 4, 4]])

        expected_attention_scores = np.array(
            [[[0.22914883, 0.24638498, 0.21247596, 0.3119903],
              [0.22970565, 0.24540883, 0.21261126, 0.31227428],
              [0.22903332, 0.2459198,  0.2110187,  0.3140282]],
             [[0.252522, 0.29074305, 0.257121, 0.19961396],
              [0.2519883,  0.2895494,  0.25718424, 0.201278],
              [0.2523954,  0.28959078, 0.25769445, 0.2003194]]])
        np.testing.assert_array_almost_equal(attention_scores,
                                             expected_attention_scores)
        self.assertEqual(attention_scores.shape, (batch_size, max_output_length,
                                                  4))

    def test_recurrent_beam1(self):
        # beam=1 and greedy should return the same result
        batch_size = 2
        max_output_length = 3
        src_mask, emb, decoder, encoder_output, encoder_hidden = self._build(
            batch_size=batch_size)

        greedy_output, _ = recurrent_greedy(
            src_mask=src_mask, embed=emb, bos_index=self.bos_index,
            eos_index=self.eos_index,
            max_output_length=max_output_length, decoder=decoder,
            encoder_output=encoder_output, encoder_hidden=encoder_hidden)

        beam_size = 1
        alpha = 1.0
        output, _ = beam_search(
            size=beam_size, eos_index=self.eos_index, pad_index=self.pad_index,
            src_mask=src_mask, embed=emb, bos_index=self.bos_index, n_best=1,
            max_output_length=max_output_length, decoder=decoder, alpha=alpha,
            encoder_output=encoder_output, encoder_hidden=encoder_hidden)
        np.testing.assert_array_equal(greedy_output, output)

    def test_recurrent_beam7(self):
        batch_size = 2
        max_output_length = 3
        src_mask, emb, decoder, encoder_output, encoder_hidden = self._build(
            batch_size=batch_size)

        beam_size = 7
        alpha = 1.0
        output, _ = beam_search(
            size=beam_size, eos_index=self.eos_index, pad_index=self.pad_index,
            src_mask=src_mask, embed=emb, bos_index=self.bos_index, n_best=1,
            max_output_length=max_output_length, decoder=decoder, alpha=alpha,
            encoder_output=encoder_output, encoder_hidden=encoder_hidden)

        self.assertEqual(output.shape, (2, 1))
        np.testing.assert_array_equal(output, [[3], [3]])
