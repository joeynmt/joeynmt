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
        self.vocab_size = len(self.vocab)  # = 8
        seed = 42
        set_seed(seed)
        # self.bos_index = 2
        self.pad_index = 1
        # self.eos_index = 3

        self.expected_transformer_ids = torch.tensor([[14, 14, 7], [14, 14, 7]])
        self.expected_transformer_scores = torch.tensor([
            [-0.4408, -0.9853, -0.9438], [-0.6358, -0.9394, -0.9752]
        ])

        self.expected_recurrent_ids = torch.tensor([[0, 6, 0], [6, 6, 6]])
        self.expected_recurrent_scores = torch.tensor([
            [-0.8579, -1.1738, -1.1085], [-0.0139, -0.1368, -0.1733],
        ])


class TestSearchTransformer(TestSearch):
    # yapf: disable
    def _build(self, batch_size):
        src_time_dim = 4
        vocab_size = 15

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

    def test_transformer_greedy_with_prompt(self):
        batch_size = 2
        max_output_length = 7
        src_mask, model, encoder_output, encoder_hidden = self._build(
            batch_size=batch_size)

        decoder_prompt = torch.tensor([[2, 11, 12, 4], [2, 13, 4, 1]])
        trg_prompt_mask = torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]])
        output, scores, attention_scores = greedy(
            src_mask=src_mask,
            max_output_length=max_output_length,
            model=model,
            encoder_output=encoder_output,
            encoder_hidden=encoder_hidden,
            return_prob="hyp",
            return_attention=True,
            fp16=False,
            decoder_prompt=decoder_prompt,
            trg_prompt_mask=trg_prompt_mask,
        )
        expected_output = torch.tensor([[11, 12, 4, 14, 14, 14, 14],
                                        [13, 4, 14, 14, 14, 14, 14]])
        # forced decoding
        self.assertEqual(output.shape, (batch_size, max_output_length))  # batch x time
        torch.testing.assert_close(output, expected_output, check_dtype=False)

        # zero log_prob on the forced positions
        expected_score = torch.tensor([
            [0.0, 0.0, 0.0, -0.2913, -0.2711, -0.3030, -0.3384],
            [0.0, 0.0, -0.7620, -0.2798, -0.3660, -0.4551, -0.5006],
        ])
        self.assertEqual(scores.shape, (batch_size, max_output_length))  # batch x time
        torch.testing.assert_close(scores, expected_score, rtol=1e-4, atol=1e-4)

        expected_att = torch.tensor([
            [[0.0000, 0.0000, 0.0000, 0.0000], [0.0000, 0.0000, 0.0000, 0.0000],
             [0.0000, 0.0000, 0.0000, 0.0000], [0.4054, 0.1381, 0.2770, 0.1794],
             [0.3993, 0.1581, 0.1975, 0.2451], [0.4037, 0.1677, 0.1932, 0.2354],
             [0.4096, 0.1826, 0.1874, 0.2203]],
            [[0.0000, 0.0000, 0.0000, 0.0000], [0.0000, 0.0000, 0.0000, 0.0000],
             [0.0722, 0.4177, 0.2564, 0.2537], [0.1450, 0.4187, 0.2538, 0.1826],
             [0.1271, 0.4199, 0.2521, 0.2009], [0.1155, 0.4215, 0.2508, 0.2122],
             [0.1124, 0.4243, 0.2512, 0.2122]],
        ])
        self.assertEqual(attention_scores.shape,  # batch x trg_time_step x src_time_step
                         (batch_size, max_output_length, encoder_output.size(1)))
        torch.testing.assert_close(attention_scores, expected_att, rtol=1e-4, atol=1e-4)

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
            beam_scores, torch.tensor([[-2.7509], [-3.1279]]), rtol=1e-4, atol=1e-4)

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
            [-1.9099], [-1.9674], [-2.0631], [-2.2915], [-2.4354],
            [-2.2932], [-2.3459], [-2.3484], [-2.4797], [-2.6288],
        ])
        torch.testing.assert_close(scores, expected_scores, rtol=1e-4, atol=1e-4)

        expected_output = torch.tensor([
            [7, 14, 14], [14, 7, 7], [14, 14, 7], [14, 14, 14], [14, 3, 1],
            [7, 14, 14], [14, 14, 7], [14, 7, 7], [14, 14, 14], [14, 3, 1],
        ])
        torch.testing.assert_close(output, expected_output)

    def test_transformer_beam7_with_prompt(self):
        batch_size = 2
        beam_size = 7
        n_best = 5
        alpha = 1.0
        max_output_length = 10
        src_mask, model, encoder_output, encoder_hidden = self._build(
            batch_size=batch_size)

        decoder_prompt = torch.tensor([[2, 11, 12, 4], [2, 13, 4, 1]])
        trg_prompt_mask = torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]])
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
            decoder_prompt=decoder_prompt,
            trg_prompt_mask=trg_prompt_mask,
        )
        # Transformer beam doesn't return attention scores
        self.assertIsNone(attention_scores)

        # batch_size * n_best x hyp_len (= time steps)
        self.assertEqual(output.shape, (batch_size * n_best, max_output_length))

        expected_output = torch.tensor(
            [[11, 12, 4, 14, 14, 14, 14, 14, 14, 14],
             [11, 12, 4, 14, 14, 14, 14, 14, 7, 14],
             [11, 12, 4, 14, 14, 14, 14, 7, 14, 14],
             [11, 12, 4, 14, 14, 14, 14, 14, 14, 7],
             [11, 12, 4, 14, 14, 14, 7, 14, 14, 14],
             [13, 4, 3, 1, 1, 1, 1, 1, 1, 1],
             [13, 4, 14, 14, 14, 14, 14, 14, 14, 14],
             [13, 4, 14, 14, 14, 14, 14, 7, 14, 14],
             [13, 4, 14, 14, 14, 14, 7, 14, 14, 14],
             [13, 4, 14, 14, 14, 7, 14, 14, 14, 14]]
        )
        torch.testing.assert_close(output, expected_output)

        expected_scores = torch.tensor([
            [-1.0547], [-1.4467], [-1.4840], [-1.5332], [-1.6025],
            [-1.4108], [-1.7024], [-1.9792], [-1.9953], [-2.0453],
        ])
        torch.testing.assert_close(scores, expected_scores, rtol=1e-4, atol=1e-4)

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

        expected_output = torch.tensor([[10, 10, 10], [10, 10, 10], [10, 10, 10]])
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

        expected_output_trg_penalty = torch.tensor([[10, 10, 10], [10, 10, 10], [10, 10, 10]])
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

        expected_output_src_penalty = torch.tensor([[10, 10, 10], [10, 10, 10], [10, 10, 10]])
        torch.testing.assert_close(
            output_src_penalty, expected_output_src_penalty, check_dtype=False)

        # Transformer Greedy can return attention probs
        # (batch_size, trg_len, src_len) = (3, 3, 4)
        expected_attention = torch.tensor([
            [[0.3082, 0.6918, 0.0000, 0.0000], [0.3221, 0.6779, 0.0000, 0.0000],
             [0.3282, 0.6718, 0.0000, 0.0000]],
            [[0.1683, 0.5097, 0.3220, 0.0000], [0.2011, 0.5101, 0.2888, 0.0000],
             [0.1968, 0.5143, 0.2889, 0.0000]],
            [[0.2811, 0.3088, 0.1084, 0.3017], [0.1903, 0.3827, 0.1060, 0.3210],
             [0.1730, 0.3840, 0.1118, 0.3313]],
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
            [-2.1110], [-2.3053], [-2.4354], [-2.4654], [-2.8679],
            [-2.5615], [-2.6288], [-2.7265], [-2.7542], [-2.9029],
        ])
        torch.testing.assert_close(
            scores_with_penalty, expected_scores_with_penalty, rtol=1e-4, atol=1e-4)

        expected_output_with_penalty = torch.tensor([
            [7, 14, 14], [14, 7, 7], [14, 3, 1], [14, 14, 7], [14, 7, 3],
            [7, 14, 14], [14, 3, 1], [14, 7, 7], [14, 14, 7], [14, 7, 4],
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

        expected_output = torch.tensor([[14, 14, 7, 7, 7, 3, 3], [14, 14, 7, 7, 7, 14, 7]])
        torch.testing.assert_close(output, expected_output, check_dtype=False)

        expected_scores = torch.tensor([
            [-0.4408, -0.9853, -0.9438, -0.8608, -0.8863, -1.5302, -1.1206],
            [-0.6358, -0.9394, -0.9752, -0.8336, -0.8651, -1.2007, -0.8916],
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
            [7, 14, 14, 14, 7, 7, 14], [7, 14, 14, 14, 7, 14, 7],
            [7, 14, 14, 14, 7, 7, 7], [7, 14, 14, 14, 7, 7, 14],
            [7, 14, 14, 14, 7, 14, 7], [14, 14, 7, 7, 7, 14, 7],
        ])
        torch.testing.assert_close(output, expected_output, check_dtype=False)

        expected_scores = torch.tensor([[-3.3003], [-3.3707], [-3.7550],
                                        [-3.6104], [-3.6620], [-3.8789]])
        torch.testing.assert_close(scores, expected_scores, rtol=1e-4, atol=1e-4)


class TestSearchRecurrent(TestSearch):
    # yapf: disable
    def _build(self, batch_size):
        src_time_dim = 4
        vocab_size = 15

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
            [[0.3436, 0.2531, 0.1345, 0.2688], [0.3064, 0.2516, 0.1752, 0.2669],
             [0.3096, 0.2619, 0.1753, 0.2532]],
            [[0.3520, 0.2047, 0.1948, 0.2485], [0.2398, 0.2544, 0.1958, 0.3100],
             [0.2602, 0.2233, 0.2199, 0.2966]],
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
            beam_scores, torch.tensor([[-3.1402], [-0.3241]]), rtol=1e-4, atol=1e-4)

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
            [0, 3, 1], [0, 6, 0], [4, 0, 6], [0, 6, 6], [0, 5, 0],
            [6, 6, 6], [6, 1, 6], [6, 6, 3], [6, 6, 5], [6, 3, 1],
        ])
        torch.testing.assert_close(output, expected_output, check_dtype=False)

        # log probabilities
        expected_scores = torch.tensor([
            [-2.1393], [-2.3551], [-2.4183], [-2.4237], [-2.5244],
            [-0.2431], [-2.4729], [-2.5563], [-2.6377], [-2.7709],
        ])
        torch.testing.assert_close(scores, expected_scores, rtol=1e-4, atol=1e-4)
