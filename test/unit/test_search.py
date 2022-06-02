from test.unit.test_helpers import TensorTestCase

import numpy as np
import torch

from joeynmt.decoders import RecurrentDecoder, TransformerDecoder
from joeynmt.embeddings import Embeddings
from joeynmt.encoders import RecurrentEncoder
from joeynmt.model import Model
from joeynmt.search import beam_search, greedy
from joeynmt.vocabulary import Vocabulary


class TestSearch(TensorTestCase):
    # pylint: disable=too-many-instance-attributes
    def setUp(self):
        self.emb_size = 12
        self.num_layers = 3
        self.hidden_size = 12
        self.ff_size = 24
        self.num_heads = 4
        self.dropout = 0.0
        self.encoder_hidden_size = 3
        self.vocab = Vocabulary(tokens=["word"])
        self.vocab_size = len(self.vocab)  # = 5
        seed = 42
        torch.manual_seed(seed)
        # self.bos_index = 2
        self.pad_index = 1
        # self.eos_index = 3

        self.expected_transformer_ids = [[5, 5, 5], [5, 5, 5]]
        self.expected_transformer_scores = np.array([
            [-1.256322, -1.3881024, -1.4247599],
            [-1.23621, -1.384755, -1.4188296],
        ])

        self.expected_recurrent_ids = [[4, 0, 4], [4, 4, 4]]
        self.expected_recurrent_scores = np.array([
            [-1.1915066, -1.2217927, -1.244617],
            [-1.1754444, -1.2138686, -1.204663],
        ])


class TestSearchTransformer(TestSearch):

    def _build(self, batch_size):
        src_time_dim = 4
        vocab_size = 7

        emb = Embeddings(
            embedding_dim=self.emb_size,
            vocab_size=vocab_size,
            padding_idx=self.pad_index,
        )

        decoder = TransformerDecoder(
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            hidden_size=self.hidden_size,
            ff_size=self.ff_size,
            dropout=self.dropout,
            emb_dropout=self.dropout,
            vocab_size=vocab_size,
            layer_norm="pre",
        )

        encoder_output = torch.rand(size=(batch_size, src_time_dim, self.hidden_size))

        for p in decoder.parameters():
            torch.nn.init.uniform_(p, -0.5, 0.5)

        src_mask = torch.ones(size=(batch_size, 1, src_time_dim)) == 1

        encoder_hidden = None  # unused

        model = Model(
            encoder=None,
            decoder=decoder,
            src_embed=emb,
            trg_embed=emb,
            src_vocab=self.vocab,
            trg_vocab=self.vocab,
        )
        return src_mask, model, encoder_output, encoder_hidden

    def test_transformer_greedy(self):
        batch_size = 2
        max_output_length = 3
        src_mask, model, encoder_output, encoder_hidden = self._build(
            batch_size=batch_size)

        output, scores, attention_scores = greedy(
            src_mask=src_mask,
            max_output_length=max_output_length,
            model=model,
            encoder_output=encoder_output,
            encoder_hidden=encoder_hidden,
            return_prob="hyp",
        )
        # Transformer greedy doesn't return attention scores
        # `return_attention = False` by default
        self.assertIsNone(attention_scores)

        # outputs
        self.assertEqual(output.shape, (batch_size, max_output_length))  # batch x time
        np.testing.assert_equal(output, self.expected_transformer_ids)

        # scores
        self.assertEqual(scores.shape, (batch_size, max_output_length))  # batch x time
        np.testing.assert_allclose(scores, self.expected_transformer_scores, rtol=1e-5)

    def test_transformer_beam1(self):
        batch_size = 2
        beam_size = 1
        alpha = 0.0
        n_best = 1
        max_output_length = 3
        src_mask, model, encoder_output, encoder_hidden = self._build(
            batch_size=batch_size)

        beam_output, beam_scores, attention_scores = beam_search(
            beam_size=beam_size,
            src_mask=src_mask,
            max_output_length=max_output_length,
            model=model,
            alpha=alpha,
            n_best=n_best,
            encoder_output=encoder_output,
            encoder_hidden=encoder_hidden,
            return_prob="hyp",
        )
        # Transformer beam doesn't return attention scores
        self.assertIsNone(attention_scores)

        # batch_size * n_best x hyp_len
        self.assertEqual(beam_output.shape, (batch_size * n_best, max_output_length))
        np.testing.assert_equal(beam_output, self.expected_transformer_ids)
        np.testing.assert_allclose(beam_scores, [[-4.174977], [-4.141973]], rtol=1e-5)

        # now compare to greedy, they should be the same for beam=1
        greedy_output, greedy_scores, _ = greedy(
            src_mask=src_mask,
            max_output_length=max_output_length,
            model=model,
            encoder_output=encoder_output,
            encoder_hidden=encoder_hidden,
            return_prob="hyp",
        )
        np.testing.assert_equal(beam_output, greedy_output)
        np.testing.assert_allclose(
            greedy_scores,
            self.expected_transformer_scores,
            rtol=1e-5,
        )

    def test_transformer_beam7(self):
        batch_size = 2
        beam_size = 7
        n_best = 5
        alpha = 1.0
        max_output_length = 3
        src_mask, model, encoder_output, encoder_hidden = self._build(
            batch_size=batch_size)

        output, scores, attention_scores = beam_search(
            beam_size=beam_size,
            src_mask=src_mask,
            max_output_length=max_output_length,
            model=model,
            alpha=alpha,
            n_best=n_best,
            encoder_output=encoder_output,
            encoder_hidden=encoder_hidden,
            return_prob="hyp",
        )
        # Transformer beam doesn't return attention scores
        self.assertIsNone(attention_scores)

        # batch_size * n_best x hyp_len (= time steps)
        self.assertEqual(output.shape, (batch_size * n_best, max_output_length))
        expected_output = [[5, 5, 5], [0, 5, 5], [0, 0, 5], [5, 5, 0], [5, 0, 5],
                           [5, 5, 5], [0, 5, 5], [5, 0, 5], [5, 5, 0], [0, 0, 5]]
        np.testing.assert_equal(output, expected_output)
        expected_scores = [
            [-3.13123298],
            [-3.29512906],
            [-3.43877649],
            [-3.44861484],
            [-3.45595121],
            [-3.10648012],
            [-3.30023503],
            [-3.43445206],
            [-3.43654943],
            [-3.47406816],
        ]
        np.testing.assert_allclose(scores, expected_scores, rtol=1e-5)

    def test_repetition_penalty_and_generate_unk(self):
        batch_size = 3
        max_output_length = 3
        src_mask, model, encoder_output, encoder_hidden = self._build(
            batch_size=batch_size)

        # no repetition penalty
        output, _, _ = greedy(
            src_mask=src_mask,
            max_output_length=max_output_length,
            model=model,
            encoder_output=encoder_output,
            encoder_hidden=encoder_hidden,
            generate_unk=False,
        )
        expected_output = [[5, 6, 4], [5, 6, 4], [5, 6, 6]]
        np.testing.assert_equal(output, expected_output)
        np.testing.assert_equal(np.count_nonzero(output), 9)  # no unk token generated

        # trg repetition penalty
        output_trg_penalty, _, _ = greedy(
            src_mask=src_mask,
            max_output_length=max_output_length,
            model=model,
            encoder_output=encoder_output,
            encoder_hidden=encoder_hidden,
            encoder_input=None,
            repetition_penalty=1.5,
            generate_unk=False,
        )
        expected_output_trg_penalty = [[5, 6, 4], [5, 6, 4], [5, 6, 4]]
        np.testing.assert_equal(output_trg_penalty, expected_output_trg_penalty)

        # src + trg repetition penalty
        # src_len = 4 (see self._build())
        src_tokens = torch.tensor([[4, 3, 1, 1], [5, 4, 3, 1], [5, 5, 6, 3]])
        src_mask = (src_tokens != 1).unsqueeze(1)
        output_src_penalty, _, attention = greedy(
            src_mask=src_mask,
            max_output_length=max_output_length,
            model=model,
            encoder_output=encoder_output,
            encoder_hidden=encoder_hidden,
            encoder_input=src_tokens,
            repetition_penalty=1.5,
            generate_unk=False,
            return_attention=True,
        )
        expected_output_src_penalty = [[5, 6, 4], [6, 4, 6], [4, 5, 6]]
        np.testing.assert_equal(output_src_penalty, expected_output_src_penalty)

        # Transformer Greedy can return attention probs
        expected_attention = np.array([[[0.50552493, 0.49447513, 0.0, 0.0],
                                        [0.49927852, 0.50072145, 0.0, 0.0],
                                        [0.4978183, 0.5021817, 0.0, 0.0]],
                                       [[0.33627957, 0.3340565, 0.32966393, 0.0],
                                        [0.33195895, 0.34128872, 0.32675233, 0.0],
                                        [0.33139172, 0.33966494, 0.3289433, 0.0]],
                                       [[0.25297716, 0.24552101, 0.26504526, 0.2364566],
                                        [0.25382724, 0.24408112, 0.2637272, 0.23836446],
                                        [0.24418367, 0.2407347, 0.26917717, 0.24590445]]
                                       ])  # (batch_size, trg_len, src_len) = (3, 3, 4)
        np.testing.assert_allclose(attention, expected_attention, rtol=1e-5)

    def test_repetition_penalty_in_beam_search(self):
        batch_size = 2
        beam_size = 7
        n_best = 5
        alpha = 1.0
        max_output_length = 3
        src_mask, model, encoder_output, encoder_hidden = self._build(
            batch_size=batch_size)

        src_tokens = torch.tensor([[5, 5, 4], [5, 6, 6]])
        expected_output_with_penalty = [
            [6, 0, 5],
            [6, 0, 1],
            [6, 1, 0],
            [6, 0, 0],
            [0, 5, 6],
            [5, 1, 0],
            [1, 0, 5],
            [4, 0, 5],
            [0, 5, 1],
            [0, 5, 3],
        ]
        expected_scores_with_penalty = [
            [-3.97227573],
            [-4.04966927],
            [-4.13298225],
            [-4.1490984],
            [-4.20313644],
            [-4.25672293],
            [-4.28554916],
            [-4.31342602],
            [-4.3269372],
            [-4.34846115],
        ]
        output_with_penalty, scores_with_penalty, _ = beam_search(
            beam_size=beam_size,
            src_mask=src_mask,
            max_output_length=max_output_length,
            model=model,
            alpha=alpha,
            n_best=n_best,
            encoder_output=encoder_output,
            encoder_hidden=encoder_hidden,
            encoder_input=src_tokens,
            repetition_penalty=1.5,
            return_prob="hyp",
        )
        np.testing.assert_equal(output_with_penalty, expected_output_with_penalty)
        np.testing.assert_allclose(scores_with_penalty,
                                   expected_scores_with_penalty,
                                   rtol=1e-5)

    def test_ngram_blocker(self):
        batch_size = 2
        max_output_length = 10
        no_repeat_ngram_size = 3
        src_mask, model, encoder_output, encoder_hidden = self._build(
            batch_size=batch_size)

        expected_output = [[5, 5, 5, 0, 5, 5, 0, 0, 5, 0],
                           [5, 5, 5, 0, 5, 5, 0, 0, 5, 0]]
        expected_scores = [[
            -1.256322, -1.3881024, -1.4247599, -1.7823244, -1.3601547, -1.4100374,
            -1.8323467, -1.4702327, -1.3825183, -1.8272306
        ],
                           [
                               -1.2362098, -1.384755, -1.4188296, -1.7987821,
                               -1.3441339, -1.4011306, -1.845073, -1.5097935,
                               -1.3674619, -1.8354796
                           ]]
        output, scores, _ = greedy(
            src_mask=src_mask,
            max_output_length=max_output_length,
            model=model,
            encoder_output=encoder_output,
            encoder_hidden=encoder_hidden,
            encoder_input=None,
            return_prob="hyp",
            no_repeat_ngram_size=no_repeat_ngram_size,
        )
        np.testing.assert_equal(output, expected_output)
        np.testing.assert_allclose(scores, expected_scores, rtol=1e-5)

    def test_ngram_blocker_in_beam_search(self):
        batch_size = 2
        beam_size = 3
        n_best = 3
        alpha = 1.0
        max_output_length = 10
        no_repeat_ngram_size = 3
        src_mask, model, encoder_output, encoder_hidden = self._build(
            batch_size=batch_size)

        expected_output = [
            [5, 5, 5, 0, 0, 0, 0, 0, 0, 5],
            [5, 5, 5, 0, 0, 0, 0, 0, 5, 5],
            [5, 5, 5, 0, 0, 0, 0, 0, 0, 0],
            [5, 5, 5, 0, 0, 5, 5, 0, 0, 0],
            [5, 5, 5, 0, 0, 0, 0, 5, 5, 0],
            [5, 5, 5, 0, 0, 0, 0, 5, 5, 2],
        ]
        expected_scores = [[-5.882798], [-5.884224], [-5.889960], [-5.969066],
                           [-6.015323], [-6.062648]]
        output, scores, _ = beam_search(
            beam_size=beam_size,
            src_mask=src_mask,
            max_output_length=max_output_length,
            model=model,
            alpha=alpha,
            n_best=n_best,
            encoder_output=encoder_output,
            encoder_hidden=encoder_hidden,
            encoder_input=None,
            return_prob="hyp",
            no_repeat_ngram_size=no_repeat_ngram_size,
        )
        np.testing.assert_equal(output, expected_output)
        np.testing.assert_allclose(scores, expected_scores, rtol=1e-5)


class TestSearchRecurrent(TestSearch):

    def _build(self, batch_size):
        src_time_dim = 4
        vocab_size = 7

        emb = Embeddings(
            embedding_dim=self.emb_size,
            vocab_size=vocab_size,
            padding_idx=self.pad_index,
        )

        encoder = RecurrentEncoder(
            emb_size=self.emb_size,
            num_layers=self.num_layers,
            hidden_size=self.encoder_hidden_size,
            bidirectional=True,
        )

        decoder = RecurrentDecoder(
            hidden_size=self.hidden_size,
            encoder=encoder,
            attention="bahdanau",
            emb_size=self.emb_size,
            vocab_size=self.vocab_size,
            num_layers=self.num_layers,
            init_hidden="bridge",
            input_feeding=True,
        )

        encoder_output = torch.rand(size=(batch_size, src_time_dim,
                                          encoder.output_size))

        for p in decoder.parameters():
            torch.nn.init.uniform_(p, -0.5, 0.5)

        src_mask = torch.ones(size=(batch_size, 1, src_time_dim)) == 1

        encoder_hidden = torch.rand(size=(batch_size, encoder.output_size))

        model = Model(
            encoder=encoder,
            decoder=decoder,
            src_embed=emb,
            trg_embed=emb,
            src_vocab=self.vocab,
            trg_vocab=self.vocab,
        )

        return src_mask, model, encoder_output, encoder_hidden

    def test_recurrent_greedy(self):
        batch_size = 2
        max_output_length = 3
        src_mask, model, encoder_output, encoder_hidden = self._build(
            batch_size=batch_size)

        output, scores, attention_scores = greedy(
            src_mask=src_mask,
            max_output_length=max_output_length,
            model=model,
            encoder_output=encoder_output,
            encoder_hidden=encoder_hidden,
            return_prob="hyp",
        )
        self.assertEqual(output.shape, (batch_size, max_output_length))
        np.testing.assert_equal(output, self.expected_recurrent_ids)
        np.testing.assert_allclose(scores, self.expected_recurrent_scores, rtol=1e-5)

        expected_attention_scores = np.array(
            [[[0.22914883, 0.24638498, 0.21247596, 0.3119903],
              [0.22970565, 0.24540883, 0.21261126, 0.31227428],
              [0.22903332, 0.2459198, 0.2110187, 0.3140282]],
             [[0.252522, 0.29074305, 0.257121, 0.19961396],
              [0.2519883, 0.2895494, 0.25718424, 0.201278],
              [0.2523954, 0.28959078, 0.25769445, 0.2003194]]])
        np.testing.assert_array_almost_equal(attention_scores,
                                             expected_attention_scores)
        self.assertEqual(attention_scores.shape, (batch_size, max_output_length, 4))

    def test_recurrent_beam1(self):
        # beam=1 and greedy should return the same result
        batch_size = 2
        max_output_length = 3
        src_mask, model, encoder_output, encoder_hidden = self._build(
            batch_size=batch_size)

        greedy_output, greedy_scores, _ = greedy(
            src_mask=src_mask,
            max_output_length=max_output_length,
            model=model,
            encoder_output=encoder_output,
            encoder_hidden=encoder_hidden,
            return_prob="hyp",
        )
        self.assertEqual(greedy_output.shape, (batch_size, max_output_length))
        np.testing.assert_equal(greedy_output, self.expected_recurrent_ids)
        np.testing.assert_allclose(
            greedy_scores,
            self.expected_recurrent_scores,
            rtol=1e-5,
        )

        beam_size = 1
        alpha = 0.0
        n_best = 1
        beam_output, beam_scores, _ = beam_search(
            beam_size=beam_size,
            src_mask=src_mask,
            n_best=n_best,
            max_output_length=max_output_length,
            model=model,
            alpha=alpha,
            encoder_output=encoder_output,
            encoder_hidden=encoder_hidden,
            return_prob="hyp",
        )
        np.testing.assert_array_equal(greedy_output, beam_output)
        np.testing.assert_allclose(
            beam_scores,
            self.expected_recurrent_scores.sum(axis=1, keepdims=True),
            rtol=1e-5,
        )

    def test_recurrent_beam7(self):
        batch_size = 2
        max_output_length = 3
        src_mask, model, encoder_output, encoder_hidden = self._build(
            batch_size=batch_size)

        beam_size = 7
        n_best = 5
        alpha = 1.0
        output, scores, _ = beam_search(
            beam_size=beam_size,
            src_mask=src_mask,
            max_output_length=max_output_length,
            model=model,
            alpha=alpha,
            n_best=n_best,
            encoder_output=encoder_output,
            encoder_hidden=encoder_hidden,
            return_prob="hyp",
        )

        self.assertEqual(output.shape, (batch_size * n_best, max_output_length))

        # output indices
        expected_output = [[4, 4, 4], [4, 4, 0], [4, 0, 4], [4, 0, 0], [0, 4, 4],
                           [4, 4, 4], [4, 4, 0], [4, 0, 4], [4, 0, 0], [0, 4, 4]]
        np.testing.assert_array_equal(output, expected_output)

        # log probabilities
        expected_scores = [
            [-2.71620679],
            [-2.72217512],
            [-2.74343705],
            [-2.76944518],
            [-2.86219954],
            [-2.69548202],
            [-2.72114182],
            [-2.76927805],
            [-2.82477784],
            [-2.87750268],
        ]
        np.testing.assert_allclose(scores, expected_scores, rtol=1e-5)
