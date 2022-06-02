from test.unit.test_helpers import TensorTestCase

import torch

from joeynmt.transformer_layers import MultiHeadedAttention, PositionalEncoding


class TestTransformerUtils(TensorTestCase):

    def setUp(self):
        seed = 42
        torch.manual_seed(seed)

    def test_position_encoding(self):
        batch_size = 2
        max_time = 3
        emb_size = hidden_size = 12

        x = torch.zeros([batch_size, max_time, emb_size])
        pe = PositionalEncoding(emb_size)
        output = pe(x)
        self.assertEqual(pe.pe.size(2), hidden_size)
        self.assertTensorAlmostEqual(output, pe.pe[:, :x.size(1)])

    def test_src_src_attention(self):
        num_heads = 2
        hidden_size = 8
        att = MultiHeadedAttention(num_heads=num_heads, size=hidden_size)

        batch_size = 1
        src_len = 4
        x = torch.randn(batch_size, src_len, hidden_size)

        output, attention_weights = att(x, x, x)
        self.assertEqual(output.size(), (batch_size, src_len, hidden_size))
        self.assertIsNone(attention_weights)  # return_weights = False by default

    def test_src_trg_attention(self):
        num_heads = 2
        hidden_size = 8
        att = MultiHeadedAttention(num_heads=num_heads, size=hidden_size)

        batch_size = 1
        src_len = 4
        trg_len = 5
        x = torch.randn(batch_size, src_len, hidden_size)
        m = torch.randn(batch_size, trg_len, hidden_size)

        output, attention_weights = att(x, x, m, return_weights=True)
        self.assertEqual(output.size(), (batch_size, trg_len, hidden_size))
        self.assertEqual(attention_weights.size(), (batch_size, trg_len, src_len))
