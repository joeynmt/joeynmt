import unittest

import torch

from joeynmt.decoders import RecurrentDecoder, TransformerDecoder
from joeynmt.embeddings import Embeddings
from joeynmt.encoders import RecurrentEncoder
from joeynmt.helpers import set_seed
from joeynmt.model import Model
from joeynmt.search import beam_search, greedy
from joeynmt.vocabulary import Vocabulary


class TestSearch(unittest.TestCase):
    # yapf: disable
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
        set_seed(seed)
        # self.bos_index = 2
        self.pad_index = 1
        # self.eos_index = 3

        self.expected_transformer_ids = torch.tensor([[0, 0, 0], [0, 5, 5]])
        self.expected_transformer_scores = torch.tensor([
            [-0.1825, -0.5574, -0.5695], [-0.6465, -0.3005, -0.1618],
        ])

        self.expected_recurrent_ids = torch.tensor([[1, 1, 0], [1, 0, 0]])
        self.expected_recurrent_scores = torch.tensor([
            [-0.3178, -0.4528, -0.1068], [-0.5680, -0.0671, -0.0612],
        ])


class TestSearchTransformer(TestSearch):
    # yapf: disable
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
            torch.nn.init.trunc_normal_(p, mean=0.0, std=1.0, a=-2.0, b=2.0)

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
            fp16=False,
        )
        # Transformer greedy doesn't return attention scores
        # `return_attention = False` by default
        self.assertIsNone(attention_scores)

        # outputs
        self.assertEqual(output.shape, (batch_size, max_output_length))  # batch x time
        torch.testing.assert_close(
            output, self.expected_transformer_ids, check_dtype=False)

        # scores
        self.assertEqual(scores.shape, (batch_size, max_output_length))  # batch x time
        torch.testing.assert_close(
            scores, self.expected_transformer_scores, rtol=1e-4, atol=1e-4)

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
            fp16=False,
        )
        # Transformer beam doesn't return attention scores
        self.assertIsNone(attention_scores)

        # batch_size * n_best x hyp_len
        self.assertEqual(beam_output.shape, (batch_size * n_best, max_output_length))
        torch.testing.assert_close(
            beam_output, self.expected_transformer_ids, check_dtype=False)
        torch.testing.assert_close(
            beam_scores, torch.tensor([[-1.3221], [-1.1592]]), rtol=1e-4, atol=1e-4)

        # now compare to greedy, they should be the same for beam=1
        greedy_output, greedy_scores, _ = greedy(
            src_mask=src_mask,
            max_output_length=max_output_length,
            model=model,
            encoder_output=encoder_output,
            encoder_hidden=encoder_hidden,
            return_prob="hyp",
            fp16=False,
        )
        torch.testing.assert_close(beam_output, greedy_output, check_dtype=False)
        torch.testing.assert_close(
            greedy_scores, self.expected_transformer_scores, rtol=1e-4, atol=1e-4)

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
            fp16=False,
        )
        # Transformer beam doesn't return attention scores
        self.assertIsNone(attention_scores)

        # batch_size * n_best x hyp_len (= time steps)
        self.assertEqual(output.shape, (batch_size * n_best, max_output_length))

        expected_scores = torch.tensor([
            [-0.9916], [-1.1645], [-1.2123], [-1.6370], [-1.9956],
            [-0.7621], [-0.8694], [-1.8208], [-2.5722], [-2.7833],
        ])
        torch.testing.assert_close(scores, expected_scores, rtol=1e-4, atol=1e-4)

        expected_output = torch.tensor([
            [0, 0, 0], [0, 5, 5], [0, 0, 5], [0, 5, 0], [5, 5, 5],
            [5, 5, 5], [0, 5, 5], [0, 0, 5], [0, 5, 0], [0, 0, 0],
        ])
        torch.testing.assert_close(output, expected_output)

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
            fp16=False,
        )

        expected_output = torch.tensor([[4, 4, 4], [4, 4, 4], [4, 5, 5]])
        torch.testing.assert_close(output, expected_output, check_dtype=False)
        self.assertEqual(torch.count_nonzero(output).item(), 9)  # no unk token

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
            fp16=False,
        )

        expected_output_trg_penalty = torch.tensor([[4, 4, 4], [4, 4, 4], [4, 5, 5]])
        torch.testing.assert_close(
            output_trg_penalty, expected_output_trg_penalty, check_dtype=False)

        # src + trg repetition penalty
        # src_len = 4 (see self._build())
        src_tokens = torch.tensor([[4, 3, 1, 1], [5, 4, 3, 1], [5, 5, 6, 3]]).long()
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
            fp16=False,
        )

        expected_output_src_penalty = torch.tensor([[4, 4, 4], [4, 4, 4], [4, 5, 5]])
        torch.testing.assert_close(
            output_src_penalty, expected_output_src_penalty, check_dtype=False)

        # Transformer Greedy can return attention probs
        # (batch_size, trg_len, src_len) = (3, 3, 4)
        expected_attention = torch.tensor([
            [[0.5196, 0.4804, 0.0000, 0.0000], [0.4938, 0.5062, 0.0000, 0.0000],
             [0.5022, 0.4978, 0.0000, 0.0000]],
            [[0.3077, 0.5314, 0.1609, 0.0000], [0.3028, 0.5456, 0.1516, 0.0000],
             [0.2967, 0.5426, 0.1607, 0.0000]],
            [[0.1679, 0.1129, 0.4732, 0.2460], [0.1669, 0.0946, 0.4861, 0.2523],
             [0.1831, 0.0725, 0.4724, 0.2720]],
        ])
        torch.testing.assert_close(attention, expected_attention, rtol=1e-4, atol=1e-4)

    def test_repetition_penalty_in_beam_search(self):
        batch_size = 2
        beam_size = 7
        n_best = 5
        alpha = 1.0
        max_output_length = 3
        src_mask, model, encoder_output, encoder_hidden = self._build(
            batch_size=batch_size)

        src_tokens = torch.tensor([[5, 5, 4], [5, 6, 6]]).long()
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
            fp16=False,
        )

        expected_scores_with_penalty = torch.tensor([
            [-1.4143], [-1.7453], [-1.9395], [-2.3824], [-3.4464],
            [-1.1363], [-1.2449], [-2.4712], [-3.3064], [-3.5255],
        ])
        torch.testing.assert_close(
            scores_with_penalty, expected_scores_with_penalty, rtol=1e-4, atol=1e-4)

        expected_output_with_penalty = torch.tensor([
            [0, 0, 0], [0, 0, 5], [0, 5, 5], [0, 5, 0], [5, 5, 5],
            [0, 5, 5], [5, 5, 5], [0, 0, 5], [0, 5, 3], [5, 0, 5],
        ])
        torch.testing.assert_close(
            output_with_penalty, expected_output_with_penalty, check_dtype=False)

    def test_ngram_blocker(self):
        batch_size = 2
        max_output_length = 7
        no_repeat_ngram_size = 3
        src_mask, model, encoder_output, encoder_hidden = self._build(
            batch_size=batch_size)

        output, scores, _ = greedy(
            src_mask=src_mask,
            max_output_length=max_output_length,
            model=model,
            encoder_output=encoder_output,
            encoder_hidden=encoder_hidden,
            encoder_input=None,
            return_prob="hyp",
            no_repeat_ngram_size=no_repeat_ngram_size,
            fp16=False,
        )

        expected_output = torch.tensor([[0, 0, 0, 0, 0, 0, 0], [0, 5, 5, 5, 0, 5, 0]])
        torch.testing.assert_close(output, expected_output, check_dtype=False)

        expected_scores = torch.tensor([
            [-0.1825, -0.5574, -0.5695, -0.5473, -0.5315, -0.5637, -0.6012],
            [-0.6465, -0.3005, -0.1618, -0.1710, -2.1467, -0.0350, -3.1589],
        ])
        torch.testing.assert_close(scores, expected_scores, rtol=1e-4, atol=1e-4)

    def test_ngram_blocker_in_beam_search(self):
        batch_size = 2
        beam_size = 3
        n_best = 3
        alpha = 1.0
        max_output_length = 7
        no_repeat_ngram_size = 3
        src_mask, model, encoder_output, encoder_hidden = self._build(
            batch_size=batch_size)

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
            fp16=False,
        )

        expected_output = torch.tensor([
            [0, 0, 0, 0, 5, 5, 5], [0, 0, 0, 0, 0, 5, 5],
            [0, 0, 0, 0, 0, 0, 0], [0, 0, 5, 5, 5, 0, 5],
            [0, 5, 5, 5, 0, 5, 0], [0, 0, 5, 5, 5, 0, 0],
        ])
        torch.testing.assert_close(output, expected_output, check_dtype=False)

        expected_scores = torch.tensor([[-1.5582], [-1.7164], [-1.7834],
                                        [-2.4636], [-3.3369], [-4.0860]])
        torch.testing.assert_close(scores, expected_scores, rtol=1e-4, atol=1e-4)


class TestSearchRecurrent(TestSearch):
    # yapf: disable
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
            torch.nn.init.trunc_normal_(p, mean=0.0, std=1.0, a=-2.0, b=2.0)

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
            fp16=False,
        )

        self.assertEqual(output.shape, (batch_size, max_output_length))
        torch.testing.assert_close(
            output, self.expected_recurrent_ids, check_dtype=False)
        torch.testing.assert_close(
            scores, self.expected_recurrent_scores, rtol=1e-4, atol=1e-4)

        expected_attention_scores = torch.tensor([
            [[0.1721, 0.2611, 0.3236, 0.2432], [0.0765, 0.1171, 0.0370, 0.7695],
             [0.0855, 0.0787, 0.0519, 0.7840]],
            [[0.2405, 0.3703, 0.3079, 0.0813], [0.2084, 0.5028, 0.2404, 0.0483],
             [0.2227, 0.3913, 0.3075, 0.0785]],
        ])
        torch.testing.assert_close(
            attention_scores, expected_attention_scores, rtol=1e-4, atol=1e-4)
        self.assertEqual(
            attention_scores.shape, (batch_size, max_output_length, 4))

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
            fp16=False,
        )

        self.assertEqual(greedy_output.shape, (batch_size, max_output_length))
        torch.testing.assert_close(
            greedy_output, self.expected_recurrent_ids, check_dtype=False)
        torch.testing.assert_close(
            greedy_scores, self.expected_recurrent_scores, rtol=1e-4, atol=1e-4)

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
            fp16=False,
        )

        torch.testing.assert_close(greedy_output, beam_output, check_dtype=False)
        torch.testing.assert_close(
            beam_scores, torch.tensor([[-0.8947], [-0.6968]]), rtol=1e-4, atol=1e-4)

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
            fp16=False,
        )

        self.assertEqual(output.shape, (batch_size * n_best, max_output_length))

        # output indices
        expected_output = torch.tensor([
            [1, 1, 0], [1, 3, 1], [4, 0, 0], [1, 0, 0], [1, 1, 3],
            [1, 0, 0], [4, 0, 0], [4, 4, 0], [1, 4, 0], [1, 0, 4],
        ])
        torch.testing.assert_close(output, expected_output, check_dtype=False)

        # log probabilities
        expected_scores = torch.tensor([
            [-0.6711], [-1.4130], [-1.8920], [-2.0893], [-2.4158],
            [-0.5226], [-0.7926], [-2.7672], [-2.7673], [-2.8261],
        ])
        torch.testing.assert_close(scores, expected_scores, rtol=1e-4, atol=1e-4)
