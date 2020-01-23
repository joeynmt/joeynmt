import torch

from joeynmt.transformer_layers import PositionalEncoding
from .test_helpers import TensorTestCase


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
