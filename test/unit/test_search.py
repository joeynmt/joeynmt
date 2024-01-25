import unittest
from types import SimpleNamespace

import torch

from joeynmt.decoders import RecurrentDecoder, TransformerDecoder
from joeynmt.embeddings import Embeddings
from joeynmt.encoders import RecurrentEncoder
from joeynmt.helpers import set_seed
from joeynmt.model import Model
from joeynmt.search import beam_search, greedy
from joeynmt.vocabulary import Vocabulary


class TestSearch(unittest.TestCase):

    # pylint: disable=too-many-instance-attributes
    def setUp(self):
        self.emb_size = 12
        self.num_layers = 3
        self.hidden_size = 12
        self.ff_size = 24
        self.num_heads = 4
        self.dropout = 0.0
        self.encoder_hidden_size = 3

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
        self.vocab = Vocabulary(tokens=["word"], cfg=special_symbols)
        self.vocab_size = len(self.vocab)  # = 8
        seed = 42
        set_seed(seed)
        self.pad_index = 1
        self.autocat = {"device_type": "cpu", "enabled": False}

        self.expected_transformer_ids = torch.tensor([[0, 0, 0], [0, 0, 7]])
        self.expected_transformer_scores = torch.tensor([
            [-0.5425, -0.4908, -0.5439], [-0.8726, -0.9898, -0.9668],
        ])  # yapf: disable

        self.expected_recurrent_ids = torch.tensor([[0, 0, 0], [0, 0, 0]])
        self.expected_recurrent_scores = torch.tensor([
            [-2.7310, -1.1748, -0.8349], [-1.8635, -1.6904, -1.4866],
        ])  # yapf: disable


class TestSearchTransformer(TestSearch):

    def _build(self, batch_size):
        src_time_dim = 4

        emb = Embeddings(
            embedding_dim=self.emb_size,
            vocab_size=self.vocab_size,
            padding_idx=self.pad_index,
        )

        decoder = TransformerDecoder(
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            hidden_size=self.hidden_size,
            ff_size=self.ff_size,
            dropout=self.dropout,
            emb_dropout=self.dropout,
            vocab_size=self.vocab_size,
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
            batch_size=batch_size
        )

        output, scores, attention_scores = greedy(
            src_mask=src_mask,
            max_output_length=max_output_length,
            model=model,
            encoder_output=encoder_output,
            encoder_hidden=encoder_hidden,
            return_prob="hyp",
            autocast=self.autocat,
        )
        # Transformer greedy doesn't return attention scores
        # `return_attention = False` by default
        self.assertIsNone(attention_scores)

        # outputs
        self.assertEqual(output.shape, (batch_size, max_output_length))  # batch x time
        torch.testing.assert_close(
            output, self.expected_transformer_ids, check_dtype=False
        )

        # scores
        self.assertEqual(scores.shape, (batch_size, max_output_length))  # batch x time
        torch.testing.assert_close(
            scores, self.expected_transformer_scores, rtol=1e-4, atol=1e-4
        )

    def test_transformer_greedy_with_prompt(self):
        batch_size = 2
        max_output_length = 7
        src_mask, model, encoder_output, encoder_hidden = self._build(
            batch_size=batch_size
        )

        decoder_prompt = torch.tensor([[2, 7, 7, 4], [0, 7, 4, 1]])
        trg_prompt_mask = torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]])
        output, scores, attention_scores = greedy(
            src_mask=src_mask,
            max_output_length=max_output_length,
            model=model,
            encoder_output=encoder_output,
            encoder_hidden=encoder_hidden,
            return_prob="hyp",
            return_attention=True,
            autocast=self.autocat,
            decoder_prompt=decoder_prompt,
            trg_prompt_mask=trg_prompt_mask,
        )

        expected_output = torch.tensor([[7, 7, 4, 0, 0, 0, 0], [7, 4, 0, 0, 0, 0, 0]])
        # forced decoding
        self.assertEqual(output.shape, (batch_size, max_output_length))  # batch x time
        torch.testing.assert_close(output, expected_output, check_dtype=False)

        # zero log_prob on the forced positions
        expected_score = torch.tensor([
            [0.0000, 0.0000, 0.0000, -0.4631, -0.3289, -0.3042, -0.3404],
            [0.0000, 0.0000, -0.7532, -0.6751, -0.5730, -0.4957, -0.5718],
        ])
        self.assertEqual(scores.shape, (batch_size, max_output_length))  # batch x time
        torch.testing.assert_close(scores, expected_score, rtol=1e-4, atol=1e-4)

        expected_att = torch.tensor([
            [[0.0000, 0.0000, 0.0000, 0.0000], [0.0000, 0.0000, 0.0000, 0.0000],
             [0.0000, 0.0000, 0.0000, 0.0000], [0.3926, 0.2844, 0.3191, 0.0039],
             [0.4019, 0.2798, 0.3156, 0.0027], [0.4072, 0.2809, 0.3093, 0.0026],
             [0.4004, 0.2799, 0.3169, 0.0027]],
            [[0.0000, 0.0000, 0.0000, 0.0000], [0.0000, 0.0000, 0.0000, 0.0000],
             [0.3194, 0.0042, 0.4271, 0.2492], [0.3523, 0.0036, 0.3957, 0.2484],
             [0.3335, 0.0034, 0.4143, 0.2488], [0.3135, 0.0031, 0.4346, 0.2488],
             [0.3322, 0.0034, 0.4158, 0.2486]],
        ])
        self.assertEqual(
            attention_scores.shape,  # batch x trg_len x src_len
            (batch_size, max_output_length, encoder_output.size(1))
        )
        torch.testing.assert_close(attention_scores, expected_att, rtol=1e-4, atol=1e-4)

    def test_transformer_beam1(self):
        batch_size = 2
        beam_size = 1
        alpha = 0.0
        n_best = 1
        max_output_length = 3
        src_mask, model, encoder_output, encoder_hidden = self._build(
            batch_size=batch_size
        )

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
            autocast=self.autocat,
        )
        # Transformer beam doesn't return attention scores
        self.assertIsNone(attention_scores)

        # batch_size * n_best x hyp_len
        self.assertEqual(beam_output.shape, (batch_size * n_best, max_output_length))
        torch.testing.assert_close(
            beam_output, self.expected_transformer_ids, check_dtype=False
        )
        torch.testing.assert_close(
            beam_scores, torch.tensor([[-1.5772], [-2.8292]]), rtol=1e-4, atol=1e-4
        )

        # now compare to greedy, they should be the same for beam=1
        greedy_output, greedy_scores, _ = greedy(
            src_mask=src_mask,
            max_output_length=max_output_length,
            model=model,
            encoder_output=encoder_output,
            encoder_hidden=encoder_hidden,
            return_prob="hyp",
            autocast=self.autocat,
        )
        torch.testing.assert_close(beam_output, greedy_output, check_dtype=False)
        torch.testing.assert_close(
            greedy_scores, self.expected_transformer_scores, rtol=1e-4, atol=1e-4
        )

    def test_transformer_beam7(self):
        batch_size = 2
        beam_size = 7
        n_best = 5
        alpha = 1.0
        max_output_length = 3
        src_mask, model, encoder_output, encoder_hidden = self._build(
            batch_size=batch_size
        )

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
            autocast=self.autocat,
        )
        # Transformer beam doesn't return attention scores
        self.assertIsNone(attention_scores)

        # batch_size * n_best x hyp_len (= time steps)
        self.assertEqual(output.shape, (batch_size * n_best, max_output_length))

        expected_output = torch.tensor([
            [0, 0, 0], [0, 0, 7], [0, 7, 0], [7, 0, 0], [7, 0, 7],
            [0, 0, 7], [7, 0, 0], [0, 0, 0], [0, 7, 0], [7, 0, 7],
        ])  # yapf: disable
        torch.testing.assert_close(output, expected_output)

        expected_scores = torch.tensor([
            [-1.1829], [-1.6948], [-1.7128], [-2.0805], [-2.9899],
            [-2.1219], [-2.1881], [-2.1931], [-2.3707], [-2.4195],
        ])  # yapf: disable
        torch.testing.assert_close(scores, expected_scores, rtol=1e-4, atol=1e-4)

    def test_transformer_beam7_with_prompt(self):
        batch_size = 2
        beam_size = 7
        n_best = 5
        alpha = 1.0
        max_output_length = 10
        src_mask, model, encoder_output, encoder_hidden = self._build(
            batch_size=batch_size
        )

        decoder_prompt = torch.tensor([[2, 7, 7, 4], [0, 7, 4, 1]])
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
            autocast=self.autocat,
            decoder_prompt=decoder_prompt,
            trg_prompt_mask=trg_prompt_mask,
        )
        # Transformer beam doesn't return attention scores
        self.assertIsNone(attention_scores)

        # batch_size * n_best x hyp_len (= time steps)
        self.assertEqual(output.shape, (batch_size * n_best, max_output_length))

        expected_output = torch.tensor([
            [7, 7, 4, 0, 0, 0, 0, 0, 0, 0], [7, 7, 4, 0, 0, 0, 0, 0, 7, 0],
            [7, 7, 4, 0, 0, 0, 0, 0, 0, 7], [7, 7, 4, 0, 0, 0, 0, 0, 0, 0],
            [7, 7, 4, 0, 0, 0, 0, 7, 0, 0], [7, 4, 0, 0, 0, 0, 0, 0, 0, 0],
            [7, 4, 0, 0, 0, 0, 0, 0, 0, 7], [7, 4, 0, 0, 0, 0, 0, 0, 7, 7],
            [7, 4, 0, 0, 0, 0, 0, 0, 7, 0], [7, 4, 0, 0, 0, 0, 0, 7, 7, 0],
        ])  # yapf: disable
        torch.testing.assert_close(output, expected_output)

        expected_scores = torch.tensor([
            [-1.2273], [-1.3972], [-1.3999], [-1.4480], [-1.6088],
            [-2.2729], [-2.2850], [-2.3435], [-2.4353], [-2.4680],
        ])  # yapf: disable
        torch.testing.assert_close(scores, expected_scores, rtol=1e-4, atol=1e-4)

    def test_repetition_penalty_and_generate_unk(self):
        batch_size = 3
        max_output_length = 3
        src_mask, model, encoder_output, encoder_hidden = self._build(
            batch_size=batch_size
        )

        # no repetition penalty
        output, _, _ = greedy(
            src_mask=src_mask,
            max_output_length=max_output_length,
            model=model,
            encoder_output=encoder_output,
            encoder_hidden=encoder_hidden,
            generate_unk=False,
            autocast=self.autocat,
        )

        expected_output = torch.tensor([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
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
            autocast=self.autocat,
        )

        expected_output_trg_penalty = torch.tensor([[1, 1, 1], [1, 1, 1], [1, 1, 3]])
        torch.testing.assert_close(
            output_trg_penalty, expected_output_trg_penalty, check_dtype=False
        )

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
            autocast=self.autocat,
        )

        expected_output_src_penalty = torch.tensor([[1, 7, 3], [1, 7, 1], [1, 1, 1]])
        torch.testing.assert_close(
            output_src_penalty, expected_output_src_penalty, check_dtype=False
        )

        # Transformer Greedy can return attention probs
        # (batch_size, trg_len, src_len) = (3, 3, 4)
        expected_attention = torch.tensor([
            [[0.5292, 0.4708, 0.0000, 0.0000], [0.5269, 0.4731, 0.0000, 0.0000],
             [0.5264, 0.4736, 0.0000, 0.0000]],
            [[0.3075, 0.6322, 0.0602, 0.0000], [0.2890, 0.6350, 0.0760, 0.0000],
             [0.3343, 0.6314, 0.0343, 0.0000]],
            [[0.2648, 0.1326, 0.5174, 0.0852], [0.2642, 0.1167, 0.5365, 0.0825],
             [0.2646, 0.1125, 0.5421, 0.0809]],
        ])
        torch.testing.assert_close(attention, expected_attention, rtol=1e-4, atol=1e-4)

    def test_repetition_penalty_in_beam_search(self):
        batch_size = 2
        beam_size = 7
        n_best = 5
        alpha = 1.0
        max_output_length = 3
        src_mask, model, encoder_output, encoder_hidden = self._build(
            batch_size=batch_size
        )

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
            autocast=self.autocat,
        )

        expected_output_with_penalty = torch.tensor([
            [0, 0, 0], [0, 7, 0], [0, 0, 7], [7, 0, 0], [7, 0, 7],
            [7, 0, 0], [0, 0, 7], [7, 0, 7], [0, 7, 0], [0, 0, 0],
        ])  # yapf: disable
        torch.testing.assert_close(
            output_with_penalty, expected_output_with_penalty, check_dtype=False
        )

        expected_scores_with_penalty = torch.tensor([
            [-1.5709], [-1.8617], [-1.8788], [-2.2284], [-3.5925],
            [-2.4791], [-2.4931], [-2.8261], [-2.8357], [-2.9624],
        ])  # yapf: disable
        torch.testing.assert_close(
            scores_with_penalty, expected_scores_with_penalty, rtol=1e-4, atol=1e-4
        )

    def test_ngram_blocker(self):
        batch_size = 2
        max_output_length = 7
        no_repeat_ngram_size = 3
        src_mask, model, encoder_output, encoder_hidden = self._build(
            batch_size=batch_size
        )

        output, scores, _ = greedy(
            src_mask=src_mask,
            max_output_length=max_output_length,
            model=model,
            encoder_output=encoder_output,
            encoder_hidden=encoder_hidden,
            encoder_input=None,
            return_prob="hyp",
            no_repeat_ngram_size=no_repeat_ngram_size,
            autocast=self.autocat,
        )

        expected_output = torch.tensor([[0, 0, 0, 0, 0, 0, 0], [0, 0, 7, 0, 1, 0, 0]])
        torch.testing.assert_close(output, expected_output, check_dtype=False)

        expected_scores = torch.tensor([
            [-0.5425, -0.4908, -0.5439, -0.7328, -0.6922, -0.6422, -0.6464],
            [-0.8726, -0.9898, -0.9668, -1.3988, -0.6783, -1.0269, -0.6804],
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
            batch_size=batch_size
        )

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
            autocast=self.autocat,
        )

        expected_output = torch.tensor([
            [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 7], [0, 0, 0, 0, 0, 7, 7],
            [7, 0, 0, 0, 0, 0, 7], [0, 0, 0, 7, 3, 1, 1], [7, 0, 0, 0, 0, 0, 0],
        ])  # yapf: disable
        torch.testing.assert_close(output, expected_output, check_dtype=False)

        expected_scores = torch.tensor([
            [-2.1454], [-2.4287], [-2.4680], [-3.2931], [-3.3489], [-3.4080]
        ])  # yapf: disable
        torch.testing.assert_close(scores, expected_scores, rtol=1e-4, atol=1e-4)


class TestSearchRecurrent(TestSearch):

    def _build(self, batch_size):
        src_time_dim = 4

        emb = Embeddings(
            embedding_dim=self.emb_size,
            vocab_size=self.vocab_size,
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

        encoder_output = torch.rand(
            size=(batch_size, src_time_dim, encoder.output_size)
        )

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
            batch_size=batch_size
        )

        output, scores, attention_scores = greedy(
            src_mask=src_mask,
            max_output_length=max_output_length,
            model=model,
            encoder_output=encoder_output,
            encoder_hidden=encoder_hidden,
            return_prob="hyp",
            autocast=self.autocat,
        )
        self.assertEqual(output.shape, (batch_size, max_output_length))
        torch.testing.assert_close(
            output, self.expected_recurrent_ids, check_dtype=False
        )
        torch.testing.assert_close(
            scores, self.expected_recurrent_scores, rtol=1e-4, atol=1e-4
        )

        expected_attention_scores = torch.tensor([
            [[0.2598, 0.2550, 0.3179, 0.1673], [0.1861, 0.2468, 0.4017, 0.1654],
             [0.1468, 0.1553, 0.6240, 0.0740]],
            [[0.0854, 0.1041, 0.7020, 0.1085], [0.2067, 0.1528, 0.5042, 0.1363],
             [0.2170, 0.1495, 0.4908, 0.1426]],
        ])
        torch.testing.assert_close(
            attention_scores, expected_attention_scores, rtol=1e-4, atol=1e-4
        )
        self.assertEqual(attention_scores.shape, (batch_size, max_output_length, 4))

    def test_recurrent_beam1(self):
        # beam=1 and greedy should return the same result
        batch_size = 2
        max_output_length = 3
        src_mask, model, encoder_output, encoder_hidden = self._build(
            batch_size=batch_size
        )

        greedy_output, greedy_scores, _ = greedy(
            src_mask=src_mask,
            max_output_length=max_output_length,
            model=model,
            encoder_output=encoder_output,
            encoder_hidden=encoder_hidden,
            return_prob="hyp",
            autocast=self.autocat,
        )

        self.assertEqual(greedy_output.shape, (batch_size, max_output_length))
        torch.testing.assert_close(
            greedy_output, self.expected_recurrent_ids, check_dtype=False
        )
        torch.testing.assert_close(
            greedy_scores, self.expected_recurrent_scores, rtol=1e-4, atol=1e-4
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
            autocast=self.autocat,
        )

        torch.testing.assert_close(greedy_output, beam_output, check_dtype=False)
        torch.testing.assert_close(
            beam_scores, torch.tensor([[-4.7406], [-5.0405]]), rtol=1e-4, atol=1e-4
        )

    def test_recurrent_beam7(self):
        batch_size = 2
        max_output_length = 3
        src_mask, model, encoder_output, encoder_hidden = self._build(
            batch_size=batch_size
        )

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
            autocast=self.autocat,
        )

        self.assertEqual(output.shape, (batch_size * n_best, max_output_length))

        # output indices
        expected_output = torch.tensor([
            [0, 0, 0], [7, 0, 0], [0, 7, 0], [0, 3, 1], [0, 0, 3],
            [0, 0, 0], [7, 0, 0], [0, 7, 0], [0, 0, 7], [0, 3, 1],
        ])  # yapf: disable
        torch.testing.assert_close(output, expected_output, check_dtype=False)

        # log probabilities
        expected_scores = torch.tensor([
            [-3.5555], [-4.2165], [-5.1052], [-5.1194], [-5.5383],
            [-3.7804], [-4.0709], [-4.7656], [-5.0124], [-5.4101],
        ])  # yapf: disable
        torch.testing.assert_close(scores, expected_scores, rtol=1e-4, atol=1e-4)
